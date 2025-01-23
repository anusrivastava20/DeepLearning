from fastbook import *
from fastai.collab import *
import pandas as pd
import numpy as np
from datetime import datetime
import os

def save_predictions(learn, dls, output_dir='predictions'):
    """
    Save predictions and actual ratings to an Excel file in a specified directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get all validation predictions
        val_preds, val_targets = learn.get_preds(dl=dls.valid)
        
        # Get user and movie IDs from validation set
        val_dl = dls.valid
        batch = val_dl.one_batch()
        users = batch[0][:, 0].cpu().numpy().flatten()  # Ensure 1D array
        movies = batch[0][:, 1].cpu().numpy().flatten()  # Ensure 1D array
        
        # Convert predictions and targets to numpy and ensure 1D
        predictions = val_preds.cpu().numpy().flatten()
        actual_ratings = val_targets.cpu().numpy().flatten()
        
        # Ensure all arrays have the same length
        min_length = min(len(users), len(movies), len(predictions), len(actual_ratings))
        users = users[:min_length]
        movies = movies[:min_length]
        predictions = predictions[:min_length]
        actual_ratings = actual_ratings[:min_length]
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'User ID': users,
            'Movie ID': movies,
            'Actual Rating': actual_ratings,
            'Predicted Rating': np.round(predictions, 2),
            'Difference': np.round(np.abs(actual_ratings - predictions), 2)
        })
        
        # Sort by absolute difference
        results_df = results_df.sort_values('Difference', ascending=False)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - actual_ratings))
        rmse = np.sqrt(np.mean((predictions - actual_ratings)**2))
        
        # Create a metrics DataFrame
        metrics_df = pd.DataFrame({
            'Metric': ['Mean Absolute Error (MAE)', 'Root Mean Square Error (RMSE)'],
            'Value': [round(mae, 3), round(rmse, 3)]
        })
        
        # Generate timestamp for the filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(output_dir, f'movie_predictions_{timestamp}.xlsx')
        
        print(f"\nSaving results to: {filename}")
        
        # Save both DataFrames to different sheets in the Excel file
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Predictions', index=False)
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Create a summary sheet
            summary_df = pd.DataFrame({
                'Metric': [
                    'Total Predictions',
                    'Average Actual Rating',
                    'Min Actual Rating',
                    'Max Actual Rating',
                    'Average Predicted Rating',
                    'Min Predicted Rating',
                    'Max Predicted Rating',
                    'Predictions within 0.5 stars',
                    'Predictions within 1 star'
                ],
                'Value': [
                    len(results_df),
                    round(results_df['Actual Rating'].mean(), 2),
                    results_df['Actual Rating'].min(),
                    results_df['Actual Rating'].max(),
                    round(results_df['Predicted Rating'].mean(), 2),
                    round(results_df['Predicted Rating'].min(), 2),
                    round(results_df['Predicted Rating'].max(), 2),
                    f"{(results_df['Difference'] <= 0.5).mean() * 100:.1f}%",
                    f"{(results_df['Difference'] <= 1.0).mean() * 100:.1f}%"
                ]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print("\nFile saved successfully!")
        print("\nSample of predictions (top 5 largest differences):")
        print(results_df.head())
        print("\nMetrics:")
        print(f"Mean Absolute Error (MAE): {mae:.3f}")
        print(f"Root Mean Square Error (RMSE): {rmse:.3f}")
        
    except Exception as e:
        print(f"Error during prediction and saving: {str(e)}")
        raise

def main():
    try:
        # Set up path
        path = Path('movie_lens_sample')
        print(f"Loading data from: {path}")
        
        # Load data
        dls = CollabDataLoaders.from_csv(path/'ratings.csv',
                                        user_name='userId',
                                        item_name='movieId',
                                        rating_name='rating',
                                        valid_pct=0.2)
        
        print(f"Training batches: {len(dls.train)}")
        print(f"Validation batches: {len(dls.valid)}")
        
        # Create and train model
        learn = collab_learner(dls, y_range=(0.5,5.5))
        learn.fine_tune(10)
        
        # Show built-in results
        learn.show_results()
        
        # Save predictions to Excel
        save_predictions(learn, dls)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        
        if 'dls' in locals():
            batch = dls.valid.one_batch()
            print("\nDebug information:")
            print("Batch structure:")
            print(f"Input shape: {batch[0].shape}")
            print(f"Target shape: {batch[1].shape}")

if __name__ == '__main__':
    main()