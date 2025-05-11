"""Create model configurations table

Revision ID: create_model_config_table
Revises: initial_migration
Create Date: 2024-03-20 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime

# revision identifiers, used by Alembic.
revision = 'create_model_config_table'
down_revision = 'initial_migration'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('model_configurations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('active_model', sa.String(length=50), default='lstm'),
        sa.Column('created_at', sa.DateTime(), default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade():
    op.drop_table('model_configurations') 