"""Authenticated APIs for user-owned AI provider profiles and assignments."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

from app.api.deps import CurrentUser, DB
from app.db.models import AIProviderProfile
from app.schemas.ai_provider import (
    ProviderAssignmentRead,
    ProviderAssignmentSet,
    ProviderCatalogRead,
    ProviderConnectionTestRead,
    ProviderCredentialReplace,
    ProviderProfileCreate,
    ProviderProfileRead,
    ProviderProfileUpdate,
)
from app.services import provider_assignments, provider_profiles
from app.services.provider_credentials import ProviderCredentialConfigurationError
from app.services.provider_errors import ProviderManagementError
from app.services.provider_registry import provider_catalog
from app.services.provider_validation import test_provider_profile

router = APIRouter()


def _raise_http_error(exc: ProviderManagementError) -> None:
    raise HTTPException(
        status_code=exc.status_code,
        detail={"code": exc.code, "message": exc.message},
    ) from None


def _profile_response(profile: AIProviderProfile) -> ProviderProfileRead:
    return ProviderProfileRead(
        id=profile.id,
        display_name=profile.display_name,
        provider_id=profile.provider_id,
        model_identifier=profile.model_identifier,
        endpoint_url=profile.endpoint_url,
        configuration=profile.configuration_json or {},
        credential_configured=profile.credential_configured,
        capability_status=profile.capability_status_json or {},
        connection_status=profile.connection_status,
        connection_error_code=profile.connection_error_code,
        last_connection_checked_at=profile.last_connection_checked_at,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


def _assignment_response(
    assignment, profile: AIProviderProfile, unverified_capabilities: list[str],
) -> ProviderAssignmentRead:
    return ProviderAssignmentRead(
        agent_role=assignment.agent_role,
        profile_id=profile.id,
        created_at=assignment.created_at,
        updated_at=assignment.updated_at,
        profile=_profile_response(profile),
        unverified_capabilities=unverified_capabilities,
    )


@router.get("/catalog", response_model=ProviderCatalogRead)
def get_catalog(current_user: CurrentUser) -> ProviderCatalogRead:
    del current_user
    return ProviderCatalogRead(providers=provider_catalog())


@router.get("/profiles", response_model=list[ProviderProfileRead])
def list_provider_profiles(db: DB, current_user: CurrentUser):
    return [
        _profile_response(profile)
        for profile in provider_profiles.list_profiles(db, str(current_user.id))
    ]


@router.post(
    "/profiles",
    response_model=ProviderProfileRead,
    status_code=status.HTTP_201_CREATED,
)
def create_provider_profile(
    payload: ProviderProfileCreate,
    db: DB,
    current_user: CurrentUser,
) -> ProviderProfileRead:
    try:
        profile = provider_profiles.create_profile(
            db,
            user_id=str(current_user.id),
            display_name=payload.display_name,
            provider_id=payload.provider_id,
            model_identifier=payload.model_identifier,
            endpoint_url=payload.endpoint_url,
            configuration=payload.configuration,
            credential=(
                payload.credential.get_secret_value()
                if payload.credential is not None
                else None
            ),
        )
    except ProviderManagementError as exc:
        _raise_http_error(exc)
    except ProviderCredentialConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "credential_encryption_unavailable", "message": str(exc)},
        ) from None
    return _profile_response(profile)


@router.get("/profiles/{profile_id}", response_model=ProviderProfileRead)
def get_provider_profile(
    profile_id: str,
    db: DB,
    current_user: CurrentUser,
) -> ProviderProfileRead:
    try:
        profile = provider_profiles.get_owned_profile(
            db, str(current_user.id), profile_id,
        )
    except ProviderManagementError as exc:
        _raise_http_error(exc)
    return _profile_response(profile)


@router.patch("/profiles/{profile_id}", response_model=ProviderProfileRead)
def update_provider_profile(
    profile_id: str,
    payload: ProviderProfileUpdate,
    db: DB,
    current_user: CurrentUser,
) -> ProviderProfileRead:
    try:
        profile = provider_profiles.update_profile(
            db,
            user_id=str(current_user.id),
            profile_id=profile_id,
            updates=payload.model_dump(exclude_unset=True),
        )
    except ProviderManagementError as exc:
        _raise_http_error(exc)
    return _profile_response(profile)


@router.delete("/profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_provider_profile(
    profile_id: str,
    db: DB,
    current_user: CurrentUser,
) -> Response:
    try:
        provider_profiles.delete_profile(
            db, user_id=str(current_user.id), profile_id=profile_id,
        )
    except ProviderManagementError as exc:
        _raise_http_error(exc)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put(
    "/profiles/{profile_id}/credential",
    response_model=ProviderProfileRead,
)
def replace_provider_credential(
    profile_id: str,
    payload: ProviderCredentialReplace,
    db: DB,
    current_user: CurrentUser,
) -> ProviderProfileRead:
    try:
        profile = provider_profiles.replace_credential(
            db,
            user_id=str(current_user.id),
            profile_id=profile_id,
            credential=payload.credential.get_secret_value(),
        )
    except ProviderManagementError as exc:
        _raise_http_error(exc)
    except ProviderCredentialConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "credential_encryption_unavailable", "message": str(exc)},
        ) from None
    return _profile_response(profile)


@router.delete(
    "/profiles/{profile_id}/credential",
    response_model=ProviderProfileRead,
)
def remove_provider_credential(
    profile_id: str,
    db: DB,
    current_user: CurrentUser,
) -> ProviderProfileRead:
    try:
        profile = provider_profiles.remove_credential(
            db, user_id=str(current_user.id), profile_id=profile_id,
        )
    except ProviderManagementError as exc:
        _raise_http_error(exc)
    return _profile_response(profile)


@router.post(
    "/profiles/{profile_id}/test-connection",
    response_model=ProviderConnectionTestRead,
)
async def test_provider_connection(
    profile_id: str,
    db: DB,
    current_user: CurrentUser,
) -> ProviderConnectionTestRead:
    try:
        profile = provider_profiles.get_owned_profile(
            db, str(current_user.id), profile_id,
        )
    except ProviderManagementError as exc:
        _raise_http_error(exc)

    outcome = await test_provider_profile(profile)
    try:
        provider_profiles.record_connection_result(
            db,
            profile,
            connected=outcome.status == "connected",
            error_code=None if outcome.status == "connected" else outcome.category,
            capability_status=outcome.capabilities,
            checked_at=outcome.checked_at,
        )
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "connection_result_save_failed",
                "message": "The provider test result could not be saved.",
            },
        ) from None
    return ProviderConnectionTestRead(
        status=outcome.status,
        category=outcome.category,
        message=outcome.message,
        capabilities=outcome.capabilities,
        checked_at=outcome.checked_at,
    )


@router.get("/assignments", response_model=list[ProviderAssignmentRead])
def get_provider_assignments(db: DB, current_user: CurrentUser):
    return [
        _assignment_response(assignment, profile, unverified)
        for assignment, profile, unverified in provider_assignments.list_assignments(
            db, user_id=str(current_user.id),
        )
    ]


@router.get(
    "/assignments/{agent_role}",
    response_model=ProviderAssignmentRead | None,
)
def get_provider_assignment(
    agent_role: str,
    db: DB,
    current_user: CurrentUser,
):
    try:
        result = provider_assignments.get_assignment(
            db, user_id=str(current_user.id), agent_role=agent_role,
        )
    except ProviderManagementError as exc:
        _raise_http_error(exc)
    if result is None:
        return None
    assignment, profile, unverified = result
    return _assignment_response(assignment, profile, unverified)


@router.put(
    "/assignments/{agent_role}",
    response_model=ProviderAssignmentRead,
)
def set_provider_assignment(
    agent_role: str,
    payload: ProviderAssignmentSet,
    db: DB,
    current_user: CurrentUser,
) -> ProviderAssignmentRead:
    try:
        assignment, profile, unverified = provider_assignments.assign_profile(
            db,
            user_id=str(current_user.id),
            agent_role=agent_role,
            profile_id=payload.profile_id,
        )
    except ProviderManagementError as exc:
        _raise_http_error(exc)
    return _assignment_response(assignment, profile, unverified)


@router.delete(
    "/assignments/{agent_role}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def clear_provider_assignment(
    agent_role: str,
    db: DB,
    current_user: CurrentUser,
) -> Response:
    try:
        provider_assignments.clear_assignment(
            db, user_id=str(current_user.id), agent_role=agent_role,
        )
    except ProviderManagementError as exc:
        _raise_http_error(exc)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
