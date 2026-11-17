"""Versioned authenticated encryption for provider credentials."""
from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

from app.config.settings import settings


ACTIVE_CREDENTIAL_FORMAT = "v1"


class ProviderCredentialConfigurationError(RuntimeError):
    """Raised when the backend cannot safely encrypt or decrypt a credential."""


def _fernet() -> Fernet:
    configured_key = settings.PROVIDER_CREDENTIAL_ENCRYPTION_KEY
    if not isinstance(configured_key, str) or not configured_key.strip():
        raise ProviderCredentialConfigurationError(
            "Provider credential encryption is not configured. Set "
            "PROVIDER_CREDENTIAL_ENCRYPTION_KEY in the backend environment."
        )
    try:
        return Fernet(configured_key.strip().encode("ascii"))
    except (ValueError, TypeError, UnicodeEncodeError):
        raise ProviderCredentialConfigurationError(
            "Provider credential encryption is misconfigured. Set a valid "
            "Fernet key in PROVIDER_CREDENTIAL_ENCRYPTION_KEY."
        ) from None


def encrypt_provider_credential(credential: str) -> tuple[str, str]:
    if not isinstance(credential, str) or not credential.strip():
        raise ProviderCredentialConfigurationError(
            "A non-empty provider credential is required."
        )
    try:
        token = _fernet().encrypt(credential.encode("utf-8")).decode("ascii")
    except ProviderCredentialConfigurationError:
        raise
    except (UnicodeEncodeError, ValueError, TypeError):
        raise ProviderCredentialConfigurationError(
            "The provider credential could not be encrypted."
        ) from None
    return f"enc:{ACTIVE_CREDENTIAL_FORMAT}:{token}", ACTIVE_CREDENTIAL_FORMAT


def decrypt_provider_credential(ciphertext: str, key_version: str | None) -> str:
    expected_prefix = f"enc:{ACTIVE_CREDENTIAL_FORMAT}:"
    if key_version != ACTIVE_CREDENTIAL_FORMAT or not isinstance(ciphertext, str):
        raise ProviderCredentialConfigurationError(
            "The stored provider credential uses an unsupported encryption version."
        )
    if not ciphertext.startswith(expected_prefix):
        raise ProviderCredentialConfigurationError(
            "The stored provider credential could not be decrypted."
        )
    token = ciphertext[len(expected_prefix):]
    try:
        return _fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except ProviderCredentialConfigurationError:
        raise
    except (InvalidToken, UnicodeDecodeError, UnicodeEncodeError, ValueError, TypeError):
        raise ProviderCredentialConfigurationError(
            "The stored provider credential could not be decrypted."
        ) from None
