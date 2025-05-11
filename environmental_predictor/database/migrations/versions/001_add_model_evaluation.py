"""add model evaluation

Revision ID: 001
Revises: 
Create Date: 2024-03-20 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create model_evaluations table
    op.create_table('model_evaluations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('city', sa.String(length=50), nullable=False),
        sa.Column('model_type', sa.String(length=50), nullable=False),
        sa.Column('rmse', sa.Float(), nullable=False),
        sa.Column('mae', sa.Float(), nullable=False),
        sa.Column('r2', sa.Float(), nullable=False),
        sa.Column('evaluation_date', sa.DateTime(), nullable=False),
        sa.Column('metrics', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Add index for faster queries
    op.create_index('ix_model_evaluations_city_model', 'model_evaluations', ['city', 'model_type'])
    op.create_index('ix_model_evaluations_evaluation_date', 'model_evaluations', ['evaluation_date'])

def downgrade():
    # Drop indexes
    op.drop_index('ix_model_evaluations_evaluation_date')
    op.drop_index('ix_model_evaluations_city_model')
    
    # Drop table
    op.drop_table('model_evaluations') 