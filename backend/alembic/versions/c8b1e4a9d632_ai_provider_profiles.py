"""Add encrypted user AI provider profiles and Agent assignments.

Revision ID: c8b1e4a9d632
Revises: f7a4c12d8e90
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c8b1e4a9d632"
down_revision: Union[str, None] = "f7a4c12d8e90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ai_provider_profiles",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("user_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=False),
        sa.Column("provider_id", sa.String(length=50), nullable=False),
        sa.Column("model_identifier", sa.String(length=255), nullable=False),
        sa.Column("endpoint_url", sa.String(length=2048), nullable=True),
        sa.Column("configuration_json", sa.JSON(), nullable=False),
        sa.Column("credential_ciphertext", sa.Text(), nullable=True),
        sa.Column("credential_key_version", sa.String(length=16), nullable=True),
        sa.Column("capability_status_json", sa.JSON(), nullable=False),
        sa.Column("connection_status", sa.String(length=20), nullable=True),
        sa.Column("connection_error_code", sa.String(length=64), nullable=True),
        sa.Column("last_connection_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"],
            name="fk_ai_provider_profiles_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_ai_provider_profiles"),
        sa.UniqueConstraint(
            "id", "user_id", name="uq_ai_provider_profiles_id_user",
        ),
    )
    op.create_index(
        "ix_ai_provider_profiles_user_id",
        "ai_provider_profiles",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_ai_provider_profiles_user_updated",
        "ai_provider_profiles",
        ["user_id", "updated_at"],
        unique=False,
    )

    op.create_table(
        "user_agent_provider_assignments",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("user_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("profile_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("agent_role", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "agent_role IN ('pipeline_agent', 'report_agent')",
            name="ck_user_agent_provider_assignments_role",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"],
            name="fk_user_agent_provider_assignments_user_id_users",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["profile_id", "user_id"],
            ["ai_provider_profiles.id", "ai_provider_profiles.user_id"],
            name="fk_user_agent_provider_assignments_owned_profile",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_user_agent_provider_assignments"),
        sa.UniqueConstraint(
            "user_id", "agent_role",
            name="uq_user_agent_provider_assignments_user_role",
        ),
    )
    op.create_index(
        "ix_user_agent_provider_assignments_profile",
        "user_agent_provider_assignments",
        ["profile_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_user_agent_provider_assignments_profile",
        table_name="user_agent_provider_assignments",
    )
    op.drop_table("user_agent_provider_assignments")
    op.drop_index(
        "ix_ai_provider_profiles_user_updated",
        table_name="ai_provider_profiles",
    )
    op.drop_index(
        "ix_ai_provider_profiles_user_id",
        table_name="ai_provider_profiles",
    )
    op.drop_table("ai_provider_profiles")
