import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib 
import matplotlib.pyplot as plt

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates Linear Regression and Ridge regression models.
    Saves performance metrics (RMSE, R2, MAE) for both train and test sets to a text file.
    Saves trained models to the 'models' directory.
    """
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True) 
    metrics_file_path = 'outputs/model_performance_metrics.txt'

    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(random_state=42),
    }

    with open(metrics_file_path, 'w') as f:
        for name, model in models.items():
            print(f"\nTraining and evaluating {name}...")
            model.fit(X_train, y_train)
            
            model_path = f'models/{name}.joblib'
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")

            train_predictions = model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
            train_r2 = r2_score(y_train, train_predictions)
            train_mae = mean_absolute_error(y_train, train_predictions)

            plt.figure(figsize=(6,6))
            plt.scatter(y_train, train_predictions, alpha=0.5)
            plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
            plt.xlabel('Gerçek Değerler')
            plt.ylabel('Tahmin Edilen Değerler')
            plt.title(f'{name} - Train Seti: Gerçek vs Tahmin')
            plt.tight_layout()
            plt.savefig(f'outputs/{name}_train_true_vs_pred.png')
            plt.close()

            test_predictions = model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
            test_r2 = r2_score(y_test, test_predictions)
            test_mae = mean_absolute_error(y_test, test_predictions)

            plt.figure(figsize=(6,6))
            plt.scatter(y_test, test_predictions, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel('Gerçek Değerler')
            plt.ylabel('Tahmin Edilen Değerler')
            plt.title(f'{name} - Test Seti: Gerçek vs Tahmin')
            plt.tight_layout()
            plt.savefig(f'outputs/{name}_test_true_vs_pred.png')
            plt.close()

            f.write(f"\nModel: {name}\n")
            f.write(f"  --- Train Set Metrics ---\n")
            f.write(f"    RMSE: {train_rmse:.4f}\n")
            f.write(f"    R2 Score: {train_r2:.4f}\n")
            f.write(f"    MAE: {train_mae:.4f}\n")
            f.write(f"  --- Test Set Metrics ---\n")
            f.write(f"    RMSE: {test_rmse:.4f}\n")
            f.write(f"    R2 Score: {test_r2:.4f}\n")
            f.write(f"    MAE: {test_mae:.4f}\n")

            print(f"  --- Train Set Metrics ---")
            print(f"    RMSE: {train_rmse:.4f}")
            print(f"    R2 Score: {train_r2:.4f}")
            print(f"    MAE: {train_mae:.4f}")
            print(f"  --- Test Set Metrics ---")
            print(f"    RMSE: {test_rmse:.4f}")
            print(f"    R2 Score: {test_r2:.4f}")
            print(f"    MAE: {test_mae:.4f}")

    print("\nModel training and evaluation completed. Metrics saved to outputs/model_performance_metrics.txt")
