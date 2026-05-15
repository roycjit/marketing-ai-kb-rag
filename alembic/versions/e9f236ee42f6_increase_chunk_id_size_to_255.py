"""Increase chunk_id size to 255

Revision ID: e9f236ee42f6
Revises: 001
Create Date: 2026-05-15 10:34:17.520639

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e9f236ee42f6'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        'document_chunks',
        'chunk_id',
        existing_type=sa.VARCHAR(length=36),
        type_=sa.String(length=255),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        'document_chunks',
        'chunk_id',
        existing_type=sa.String(length=255),
        type_=sa.VARCHAR(length=36),
        existing_nullable=False,
    )
