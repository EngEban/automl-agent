"""Ownership-safe and atomic provider assignments for Agent roles."""
from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DBSession

from app.db.models import (
    AIProviderProfile,
    User,
    UserAgentProviderAssignment,
)
from app.services.provider_errors import ProviderManagementError
from app.services.provider_profiles import get_owned_profile
from app.services.provider_registry import (
    AGENT_ROLES,
    ROLE_REQUIRED_CAPABILITIES,
    get_provider,
)


def _validate_role(agent_role: str) -> None:
    if agent_role not in AGENT_ROLES:
        raise ProviderManagementError(
            "invalid_agent_role",
            "Agent role must be pipeline_agent or report_agent.",
            422,
        )


def list_assignments(
    db: DBSession, *, user_id: str,
) -> list[tuple[UserAgentProviderAssignment, AIProviderProfile, list[str]]]:
    rows = db.execute(
        select(UserAgentProviderAssignment, AIProviderProfile)
        .join(
            AIProviderProfile,
            (AIProviderProfile.id == UserAgentProviderAssignment.profile_id)
            & (AIProviderProfile.user_id == UserAgentProviderAssignment.user_id),
        )
        .where(UserAgentProviderAssignment.user_id == user_id)
        .order_by(UserAgentProviderAssignment.agent_role)
    ).all()
    return [
        (assignment, profile, _unverified_capabilities(profile, assignment.agent_role))
        for assignment, profile in rows
    ]


def get_assignment(
    db: DBSession, *, user_id: str, agent_role: str,
) -> tuple[UserAgentProviderAssignment, AIProviderProfile, list[str]] | None:
    _validate_role(agent_role)
    row = db.execute(
        select(UserAgentProviderAssignment, AIProviderProfile)
        .join(
            AIProviderProfile,
            (AIProviderProfile.id == UserAgentProviderAssignment.profile_id)
            & (AIProviderProfile.user_id == UserAgentProviderAssignment.user_id),
        )
        .where(
            UserAgentProviderAssignment.user_id == user_id,
            UserAgentProviderAssignment.agent_role == agent_role,
        )
    ).first()
    if row is None:
        return None
    assignment, profile = row
    return assignment, profile, _unverified_capabilities(profile, agent_role)


def _unverified_capabilities(
    profile: AIProviderProfile, agent_role: str,
) -> list[str]:
    statuses = profile.capability_status_json or {}
    return [
        capability
        for capability in ROLE_REQUIRED_CAPABILITIES[agent_role]
        if statuses.get(capability, "unknown") == "unknown"
    ]


def assign_profile(
    db: DBSession,
    *,
    user_id: str,
    agent_role: str,
    profile_id: str,
) -> tuple[UserAgentProviderAssignment, AIProviderProfile, list[str]]:
    _validate_role(agent_role)
    profile = get_owned_profile(db, user_id, profile_id)
    definition = get_provider(profile.provider_id)
    if definition is None:
        raise ProviderManagementError(
            "unsupported_provider", "The provider profile uses an unsupported provider.", 422,
        )
    if definition.api_key_requirement == "required" and not profile.credential_configured:
        raise ProviderManagementError(
            "credential_required",
            "Configure a provider credential before assigning this profile.",
            409,
        )
    if profile.connection_status != "connected":
        raise ProviderManagementError(
            "connection_test_required",
            "Test this provider profile's connection before assigning it.",
            409,
        )

    statuses = profile.capability_status_json or {}
    required = ROLE_REQUIRED_CAPABILITIES[agent_role]
    unsupported = [
        capability for capability in required
        if statuses.get(capability, "unknown") == "unsupported"
    ]
    if unsupported:
        raise ProviderManagementError(
            "incompatible_capabilities",
            "The tested model does not support the capabilities required by this Agent role.",
            409,
        )

    try:
        # Serializing on the owner row makes insert/update assignment changes
        # atomic for this user, including when the role has no row yet.
        owner_id = db.scalar(
            select(User.id).where(User.id == user_id).with_for_update()
        )
        if owner_id is None:
            raise ProviderManagementError(
                "user_not_found", "Authenticated user was not found.", 404,
            )
        assignment = db.scalar(
            select(UserAgentProviderAssignment).where(
                UserAgentProviderAssignment.user_id == user_id,
                UserAgentProviderAssignment.agent_role == agent_role,
            )
        )
        if assignment is None:
            assignment = UserAgentProviderAssignment(
                user_id=user_id,
                agent_role=agent_role,
                profile_id=profile.id,
            )
            db.add(assignment)
        else:
            assignment.profile_id = profile.id
        db.commit()
        db.refresh(assignment)
    except IntegrityError:
        db.rollback()
        raise ProviderManagementError(
            "assignment_conflict",
            "The provider assignment changed concurrently. Retry the operation.",
            409,
        ) from None
    except ProviderManagementError:
        db.rollback()
        raise

    return assignment, profile, _unverified_capabilities(profile, agent_role)


def clear_assignment(
    db: DBSession, *, user_id: str, agent_role: str,
) -> None:
    _validate_role(agent_role)
    db.execute(delete(UserAgentProviderAssignment).where(
        UserAgentProviderAssignment.user_id == user_id,
        UserAgentProviderAssignment.agent_role == agent_role,
    ))
    db.commit()
