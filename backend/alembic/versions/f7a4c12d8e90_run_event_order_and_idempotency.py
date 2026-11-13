"""Index run idempotency keys and stable event pagination."""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f7a4c12d8e90"
down_revision: Union[str, None] = "18cd42a0a3f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "uq_analysis_runs_owner_idempotency",
        "analysis_runs",
        ["requested_by", "idempotency_key"],
        unique=True,
    )
    op.add_column(
        "run_events",
        sa.Column("sequence", sa.Integer(), nullable=True),
    )
    op.execute(sa.text(
        """
        WITH ranked AS (
            SELECT id, row_number() OVER (
                PARTITION BY run_id ORDER BY created_at, id
            ) AS event_sequence
            FROM run_events
        )
        UPDATE run_events AS events
        SET sequence = ranked.event_sequence
        FROM ranked
        WHERE events.id = ranked.id
        """
    ))
    op.alter_column("run_events", "sequence", nullable=False)
    op.create_index(
        "uq_run_events_run_sequence",
        "run_events",
        ["run_id", "sequence"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_run_events_run_sequence", table_name="run_events")
    op.drop_column("run_events", "sequence")
    op.drop_index(
        "uq_analysis_runs_owner_idempotency",
        table_name="analysis_runs",
    )
