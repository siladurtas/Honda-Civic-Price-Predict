from DataLoader import DataLoader
from Preprocessing import process_all
from FeatureSelection import select_features
from Train import train_and_evaluate_models
import pandas as pd

def main():
    civic_data = DataLoader.load_civic_data()
    
    if civic_data is not None:
        print("Successfully loaded civic.json data")
        
        df = pd.DataFrame(civic_data)
    
        X_train, X_test, y_train, y_test = process_all(df)
        
        print("Data preprocessing completed successfully")
    
        print("\nStarting feature selection...")
        selected_features = select_features(X_train, y_train)
        
        if selected_features:
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]
            print(f"X_train and X_test filtered to selected features. New X_train shape: {X_train.shape}")
        else:
            print("No features were selected. Proceeding with all available features.")

        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        

        X_train.to_csv('outputs/X_train_selected.csv', index=False)
        X_test.to_csv('outputs/X_test_selected.csv', index=False)
        y_train.to_csv('outputs/y_train.csv', index=False)
        y_test.to_csv('outputs/y_test.csv', index=False)

        print("\nStarting model training and evaluation...")
        train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
    else:
        print("Failed to load civic.json data")

if __name__ == "__main__":
    main()
