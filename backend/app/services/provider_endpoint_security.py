"""Endpoint validation and DNS-pinned destination selection for explicit tests."""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from app.config.settings import settings
from app.services.provider_errors import ProviderManagementError
from app.services.provider_registry import get_provider


_BLOCKED_METADATA_NAMES = frozenset({
    "metadata",
    "metadata.google.internal",
    "metadata.goog",
    "instance-data",
})
_BLOCKED_METADATA_IPS = frozenset({
    "168.63.129.16",       # Azure platform virtual IP
    "169.254.169.254",     # Common cloud metadata endpoint
    "fd00:ec2::254",       # AWS IPv6 metadata endpoint
})


@dataclass(frozen=True)
class PinnedEndpoint:
    base_url: str
    host_header: str
    sni_hostname: str | None
    scheme: str
    pinned_ip: str


def _parsed_endpoint(provider_id: str, endpoint_url: str):
    definition = get_provider(provider_id)
    if definition is None:
        raise ProviderManagementError(
            "unsupported_provider", "Select a provider from the supported catalog.", 422,
        )
    if not isinstance(endpoint_url, str) or not endpoint_url.strip():
        raise ProviderManagementError(
            "endpoint_required", "This provider requires an endpoint URL.", 422,
        )
    if len(endpoint_url) > 2048 or any(ord(char) < 32 for char in endpoint_url):
        raise ProviderManagementError(
            "invalid_endpoint", "The endpoint URL is invalid.", 422,
        )
    try:
        parsed = urlsplit(endpoint_url.strip())
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise ProviderManagementError(
            "invalid_endpoint", "The endpoint URL is invalid.", 422,
        ) from None

    if parsed.scheme.casefold() not in {"http", "https"}:
        raise ProviderManagementError(
            "unsupported_endpoint_scheme", "Endpoint URLs must use HTTP or HTTPS.", 422,
        )
    if not hostname or parsed.username is not None or parsed.password is not None:
        raise ProviderManagementError(
            "invalid_endpoint", "Endpoint URLs must not contain user or password information.", 422,
        )
    if parsed.query or parsed.fragment:
        raise ProviderManagementError(
            "endpoint_query_not_allowed",
            "Endpoint URLs must not contain query parameters or fragments.",
            422,
        )

    host = hostname.rstrip(".").casefold()
    if host in _BLOCKED_METADATA_NAMES or host.endswith((".internal", ".localhost", ".local")):
        if definition.endpoint_mode == "custom_openai_compatible":
            raise ProviderManagementError(
                "unsafe_endpoint", "The endpoint resolves to a reserved internal destination.", 422,
            )

    local_mode = definition.endpoint_mode == "local_openai_compatible"
    if not local_mode and parsed.scheme.casefold() != "https":
        raise ProviderManagementError(
            "https_required", "Remote custom endpoints must use HTTPS.", 422,
        )

    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        address = None

    if address is not None:
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        if str(address) in _BLOCKED_METADATA_IPS:
            raise ProviderManagementError(
                "unsafe_endpoint", "Cloud metadata destinations are not allowed.", 422,
            )
        if local_mode and not address.is_loopback:
            raise ProviderManagementError(
                "private_destination", "Local provider endpoints must use a loopback address.", 422,
            )
        if not local_mode and not address.is_global:
            raise ProviderManagementError(
                "private_destination", "Private, reserved, and metadata destinations are not allowed.", 422,
            )
    elif local_mode and host != "localhost":
        raise ProviderManagementError(
            "local_endpoint_host_not_allowed",
            "Local provider endpoints must use localhost or a loopback IP address.",
            422,
        )

    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise ProviderManagementError(
            "invalid_endpoint", "The endpoint port is invalid.", 422,
        )
    return parsed, host, port, address, local_mode


def validate_endpoint_syntax(provider_id: str, endpoint_url: str | None) -> str | None:
    """Validate and normalize endpoint syntax without making a DNS/network request."""
    definition = get_provider(provider_id)
    if definition is None:
        raise ProviderManagementError(
            "unsupported_provider", "Select a provider from the supported catalog.", 422,
        )
    if definition.endpoint_requirement == "managed":
        if endpoint_url is not None:
            raise ProviderManagementError(
                "endpoint_not_allowed",
                "This provider uses its managed endpoint; custom URLs are not accepted.",
                422,
            )
        return None
    if endpoint_url is None:
        raise ProviderManagementError(
            "endpoint_required", "This provider requires an endpoint URL.", 422,
        )
    parsed, _host, _port, _address, _local = _parsed_endpoint(provider_id, endpoint_url)
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.casefold(), parsed.netloc, path, "", ""))


def _destination_is_allowed(address: ipaddress.IPv4Address | ipaddress.IPv6Address, local_mode: bool) -> bool:
    normalized = address
    if isinstance(address, ipaddress.IPv6Address):
        normalized = address.ipv4_mapped or address
    if str(normalized) in _BLOCKED_METADATA_IPS:
        return False
    if local_mode:
        return normalized.is_loopback
    return normalized.is_global


def resolve_and_pin_endpoint(provider_id: str, endpoint_url: str) -> PinnedEndpoint:
    """Resolve once, reject unsafe answers, and return a numeric-IP endpoint.

    The HTTP client must use this numeric URL, preserve Host/SNI, disable proxy
    environment variables, and refuse redirects. This prevents a second DNS
    lookup (and DNS-rebinding bypass) between validation and connection.
    """
    parsed, host, port, literal_address, local_mode = _parsed_endpoint(
        provider_id, endpoint_url,
    )
    effective_port = port or (443 if parsed.scheme.casefold() == "https" else 80)

    if literal_address is not None:
        addresses = [literal_address]
    else:
        try:
            results = socket.getaddrinfo(
                host, effective_port, type=socket.SOCK_STREAM,
            )
        except (OSError, socket.gaierror):
            raise ProviderManagementError(
                "endpoint_dns_failed", "The endpoint hostname could not be resolved.", 422,
            ) from None
        addresses = []
        for result in results:
            try:
                resolved = ipaddress.ip_address(result[4][0].split("%", 1)[0])
                if isinstance(resolved, ipaddress.IPv6Address) and resolved.ipv4_mapped is not None:
                    resolved = resolved.ipv4_mapped
                addresses.append(resolved)
            except (ValueError, IndexError, TypeError):
                continue
        addresses = list(dict.fromkeys(addresses))
        if not addresses:
            raise ProviderManagementError(
                "endpoint_dns_failed", "The endpoint hostname has no usable IP address.", 422,
            )

    if any(not _destination_is_allowed(address, local_mode) for address in addresses):
        raise ProviderManagementError(
            "private_destination",
            "The endpoint resolves to a private, reserved, or metadata destination.",
            422,
        )

    pinned = addresses[0]
    rendered_ip = f"[{pinned}]" if pinned.version == 6 else str(pinned)
    default_port = 443 if parsed.scheme.casefold() == "https" else 80
    pinned_netloc = rendered_ip
    if effective_port != default_port:
        pinned_netloc += f":{effective_port}"
    pinned_url = urlunsplit((
        parsed.scheme.casefold(), pinned_netloc, parsed.path.rstrip("/"), "", "",
    ))

    original_host = f"[{host}]" if ":" in host else host
    host_header = original_host
    if effective_port != default_port:
        host_header += f":{effective_port}"
    try:
        ipaddress.ip_address(host)
        sni_hostname = None
    except ValueError:
        sni_hostname = host if parsed.scheme.casefold() == "https" else None

    return PinnedEndpoint(
        base_url=pinned_url,
        host_header=host_header,
        sni_hostname=sni_hostname,
        scheme=parsed.scheme.casefold(),
        pinned_ip=str(pinned),
    )


async def resolve_and_pin_endpoint_async(
    provider_id: str, endpoint_url: str,
) -> PinnedEndpoint:
    """Bound DNS work without blocking the API event loop."""
    timeout_seconds = min(5.0, float(settings.PROVIDER_TEST_TIMEOUT_SECONDS))
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                resolve_and_pin_endpoint,
                provider_id,
                endpoint_url,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise ProviderManagementError(
            "endpoint_dns_timeout", "Endpoint DNS resolution exceeded its timeout.", 422,
        ) from None
