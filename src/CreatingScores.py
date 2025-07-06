import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance

def create_damage_scores(data):
    damage_columns = [col for col in data.columns if col.endswith(('_changed', '_painted'))]
    
    if not damage_columns:
        print("No damage columns found in the dataset")
        return data
    
    X = data[damage_columns]
    y = data['price']
    
    freq = X.mean()
    weights = freq.apply(lambda x: 1/(x+1e-5))  
    sample_weights = X.dot(weights)
    

    model = XGBRegressor(
        n_estimators=100,
        random_state=42,
        learning_rate=0.1,
        max_depth=5,
        objective='reg:squarederror'
    )
    model.fit(X, y, sample_weight=sample_weights)
    
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': perm_importance.importances_mean
    }).sort_values(by='importance', ascending=False)
    
    os.makedirs('outputs', exist_ok=True)
    
    feature_importance.to_csv('outputs/ScoreImportant.csv', index=False)

    plt.figure(figsize=(12, 6))
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Permutation Feature Importance for Damage Scores (XGBoost)')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance_permutation_xgb.png')
    plt.close()
    
    importance_dict = feature_importance.set_index('feature')['importance'].to_dict()
    damage_scores = []
    for _, row in X.iterrows():
        score = sum(row[col] * importance_dict.get(col, 0) for col in damage_columns)
        damage_scores.append(score)
    
    data['damage_score'] = damage_scores

    data = data.drop(columns=damage_columns)
    
    print("Damage scores created successfully with XGBoost and permutation importance.")
    return data
