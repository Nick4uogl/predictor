import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from sqlalchemy.orm import sessionmaker
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from database.db_manager import get_db_session, ModelEvaluation

def plot_mse_history(city: str, save_path: str = None):
    """
    Plot MSE history for all models in a specific city
    
    Args:
        city (str): City name to plot metrics for
        save_path (str, optional): Path to save the plot. If None, plot will be displayed
    """
    # Get database session
    session = get_db_session()
    
    try:
        # Get all evaluations for the city
        evaluations = session.query(ModelEvaluation).filter(
            ModelEvaluation.city == city
        ).order_by(
            ModelEvaluation.evaluation_date
        ).all()
        
        if not evaluations:
            print(f"No evaluation data found for {city}")
            return
            
        # Prepare data for plotting
        data = []
        for eval in evaluations:
            data.append({
                'date': eval.evaluation_date,
                'model_type': eval.model_type,
                'mse': eval.rmse ** 2  # Convert RMSE to MSE
            })
            
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        
        # Plot MSE for each model type
        for model_type in df['model_type'].unique():
            model_data = df[df['model_type'] == model_type]
            plt.plot(model_data['date'], model_data['mse'], 
                    marker='o', label=model_type, linewidth=2)
        
        plt.title(f'MSE History for {city}', fontsize=14, pad=20)
        plt.xlabel('Evaluation Date', fontsize=12)
        plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
        plt.legend(title='Model Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error plotting MSE history: {str(e)}")
        
    finally:
        session.close()

if __name__ == "__main__":
    # Example usage
    city = "Kyiv"  # Replace with your city
    save_path = "evaluation_results/mse_history.png"
    plot_mse_history(city, save_path) 