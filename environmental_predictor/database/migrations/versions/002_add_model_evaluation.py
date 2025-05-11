"""add model_evaluations table

Revision ID: 002
Revises: 001
Create Date: 2024-04-27 20:10:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('model_evaluations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('city', sa.String(length=100), nullable=False),
        sa.Column('model_type', sa.String(length=50), nullable=False),
        sa.Column('rmse', sa.Float(), nullable=True),
        sa.Column('mae', sa.Float(), nullable=True),
        sa.Column('r2', sa.Float(), nullable=True),
        sa.Column('evaluation_date', sa.DateTime(), nullable=False),
        sa.Column('metrics', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade():
    op.drop_table('model_evaluations') 