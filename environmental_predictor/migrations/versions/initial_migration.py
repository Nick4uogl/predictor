"""Initial migration

Revision ID: initial_migration
Revises: 
Create Date: 2024-03-20 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'initial_migration'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Mark existing tables as present without trying to create them
    # This is a no-op migration that just marks the tables as present
    pass

def downgrade():
    # Don't drop any tables in downgrade since they already exist
    pass 