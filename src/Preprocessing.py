import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from CreatingScores import create_damage_scores


def drop_unnecessary_columns(data):
    if 'title' in data.columns:
        data = data.drop('title', axis=1)
    print("Dropped unnecessary columns successfully.")
    return data

def process_engine_features(data):
    if 'enginePower' in data.columns:
        def clean_engine_power(x):
            if pd.isna(x): return x
            x = str(x).lower().replace('hp', '').strip()
            if '-' in x:
                try:
                    return np.mean([float(i.strip()) for i in x.split('-')])
                except:
                    return np.nan
            try:
                return float(x)
            except:
                return np.nan
        data['enginePower'] = data['enginePower'].apply(clean_engine_power)

    if 'engineCapacity' in data.columns:
        def clean_engine_capacity(x):
            if pd.isna(x): return x
            x = str(x).lower().replace('cm3', '').strip()
            if '-' in x:
                try:
                    return np.mean([float(i.strip()) for i in x.split('-')])
                except:
                    return np.nan
            try:
                return float(x)
            except:
                return np.nan
        data['engineCapacity'] = data['engineCapacity'].apply(clean_engine_capacity)

    print("Processed engine features successfully.")
    return data


def clean_price(data):
    if 'price' in data.columns:
        data['price'] = data['price'].str.replace('TL', '', regex=False).str.replace('.', '', regex=False).str.strip()
        data['price'] = pd.to_numeric(data['price'], errors='coerce')
        data['price'] = np.log1p(data['price'])
    print("Cleaned price column successfully.")
    return data


def clean_mileage(data):
    if 'mileage' in data.columns:
        data['mileage'] = data['mileage'].str.replace('.', '', regex=False).str.strip()
        data['mileage'] = pd.to_numeric(data['mileage'], errors='coerce')
    print("Cleaned mileage column successfully.")
    return data


def clean_year(data):
    if 'year' in data.columns:
        data['year'] = pd.to_numeric(data['year'], errors='coerce')
    print("Cleaned year column successfully.")
    return data


def calculate_vehicle_age(data):
    if 'year' in data.columns:
        data['vehicle_age'] = 2025 - data['year']
        data = data.drop('year', axis=1)
    print("Calculated vehicle age successfully.")
    return data


def handle_outliers(data):
    numerical_cols = ['price', 'mileage', 'enginePower', 'engineCapacity', 'vehicle_age']
    for col in numerical_cols:
        if col in data.columns and data[col].dtype != 'object':
            mean_val, std_val = data[col].mean(), data[col].std()
            if std_val == 0:
                print(f"Column '{col}' has zero std dev. Skipping.")
                continue
            z_scores = np.abs((data[col] - mean_val) / std_val)
            data[col] = np.where(z_scores > 3, np.nan, data[col])
            print(f"Handled outliers for column '{col}'.")
    return data


def fill_missing_values(data):
    numerical_cols = ['price', 'mileage', 'enginePower', 'engineCapacity', 'vehicle_age']
    numerical_medians = {col: data[col].median() for col in numerical_cols if col in data.columns}
    for col, val in numerical_medians.items():
        data[col] = data[col].fillna(val)
    joblib.dump(numerical_medians, 'models/numerical_medians.joblib')

    categorical_cols = ['transmission', 'fuelType', 'color', 'model', 'location']
    categorical_modes = {col: data[col].mode()[0] for col in categorical_cols if col in data.columns}
    for col, val in categorical_modes.items():
        data[col] = data[col].fillna(val)
    joblib.dump(categorical_modes, 'models/categorical_modes.joblib')

    print("Filled missing values and saved statistics.")
    return data


def visualize_categorical(data):
    if 'location' in data.columns:
        data['location'].value_counts().nlargest(10).plot(kind='bar')
        plt.title('Top 10 Locations')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('outputs/location_distribution.png')
        plt.close()

    if 'model' in data.columns:
        data['model'].value_counts().nlargest(12).plot(kind='bar')
        plt.title('Top 12 Models')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('outputs/model_distribution.png')
        plt.close()

    print("Visualized categorical distributions.")
    

def process_damage_info(data):
    if 'damageInfo' not in data.columns:
        return data

    parts = set()
    for damage_info in data['damageInfo']:
        items = damage_info if isinstance(damage_info, list) else [damage_info]
        for item in items:
            if isinstance(item, dict):
                for key in ['changed', 'painted']:
                    val = item.get(key)
                    if isinstance(val, dict):
                        parts.update(val.keys())
                    elif isinstance(val, list):
                        parts.update(val)

    joblib.dump(list(parts), 'models/damage_parts.joblib')

    for part in parts:
        data[f'{part}_changed'] = data['damageInfo'].apply(
            lambda x: int(any(
                isinstance(item, dict) and (
                    (isinstance(item.get('changed'), dict) and item['changed'].get(part, False)) or
                    (isinstance(item.get('changed'), list) and part in item['changed'])
                )
                for item in (x if isinstance(x, list) else [x])
            )) if isinstance(x, (list, dict)) else 0
        )
        data[f'{part}_painted'] = data['damageInfo'].apply(
            lambda x: int(any(
                isinstance(item, dict) and (
                    (isinstance(item.get('painted'), dict) and item['painted'].get(part, False)) or
                    (isinstance(item.get('painted'), list) and part in item['painted'])
                )
                for item in (x if isinstance(x, list) else [x])
            )) if isinstance(x, (list, dict)) else 0
        )
    data = data.drop('damageInfo', axis=1)
    print("Processed damage info successfully.")
    return data


def encode_categorical(data):
    for col in ['transmission', 'fuelType', 'color', 'model', 'location']:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            joblib.dump(le, f'models/label_encoder_{col}.joblib')
    print("Encoded categorical features.")
    return data


def add_derived_features(data):
    if 'enginePower' in data.columns and 'engineCapacity' in data.columns:
        data['power_to_capacity_ratio'] = data['enginePower'] / data['engineCapacity']
        print("Added power-to-capacity ratio.")

    if 'vehicle_age' in data.columns and 'mileage' in data.columns:
        data['age_to_mileage_ratio'] = data['vehicle_age'] / (data['mileage'] + 1)
        print("Added age-to-mileage ratio.")

    return data


def visualize_price_distribution(data):
    if 'price' in data.columns:
        transformed = data['price']
        plt.hist(transformed, bins='auto', edgecolor='black')
        plt.title('Price Distribution (Log Scale)')
        plt.axvline(transformed.mean(), color='red', linestyle='--', label='Mean')
        plt.axvline(transformed.median(), color='green', linestyle='--', label='Median')
        plt.legend()
        plt.tight_layout()
        plt.savefig('outputs/price_distribution_log_transformed.png')
        plt.close()

        with open('outputs/price_statistics_log_transformed.txt', 'w') as f:
            f.write(f"Mean: {transformed.mean():.2f}\n")
            f.write(f"Median: {transformed.median():.2f}\n")
            f.write(f"Min: {transformed.min():.2f}\n")
            f.write(f"Max: {transformed.max():.2f}\n")
            f.write(f"Std: {transformed.std():.2f}\n")

        print("Visualized price distribution.")
    return data


def scale_features(data):
    features = [col for col in data.columns if col != 'price']
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    joblib.dump(scaler, 'models/scaler.joblib')
    print("Scaled numerical features.")
    return data


def save_processed_data(data, filename='processed_data.csv'):
    data.to_csv(f'outputs/{filename}', index=False)
    print("Saved processed data.")


def plot_price_correlations(data):
    if 'price' in data.columns:
        corr = data.corr(numeric_only=True)['price'].sort_values(ascending=False)
        plt.figure(figsize=(10, 8))
        sns.heatmap(data[corr.index].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Feature Correlation with Price")
        plt.tight_layout()
        plt.savefig('outputs/price_correlations.png')
        plt.close()
        print("Plotted price correlation heatmap.")

def visualize_damage_counts(data):
    """Visualize and save counts of changed and painted parts"""
    if 'damageInfo' not in data.columns:
        return data

    changed_counts, painted_counts = {}, {}

    for damage_info in data['damageInfo']:
        items = damage_info if isinstance(damage_info, list) else [damage_info]
        for item in items:
            if isinstance(item, dict):
                for key, counter in [('changed', changed_counts), ('painted', painted_counts)]:
                    val = item.get(key)
                    if isinstance(val, dict):
                        for part, is_true in val.items():
                            if is_true:
                                counter[part] = counter.get(part, 0) + 1
                    elif isinstance(val, list):
                        for part in val:
                            counter[part] = counter.get(part, 0) + 1

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    pd.Series(changed_counts).sort_values(ascending=False).plot(kind='bar')
    plt.title('Count of Changed Parts')
    plt.xticks(rotation=45, ha='right')
    plt.subplot(1, 2, 2)
    pd.Series(painted_counts).sort_values(ascending=False).plot(kind='bar')
    plt.title('Count of Painted Parts')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('outputs/damage_part_counts.png')
    plt.close()

    with open('outputs/damage_part_counts.txt', 'w', encoding='utf-8') as f:
        f.write("Changed Parts Counts:\n===================\n")
        for part, count in sorted(changed_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{part}: {count}\n")
        f.write("\nPainted Parts Counts:\n===================\n")
        for part, count in sorted(painted_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{part}: {count}\n")

    print("Visualized and saved damage part counts successfully.")
    return data

def prepare_train_test(data, test_size=0.2, random_state=42):
    X = data.drop('price', axis=1)
    y = data['price']
    print("Prepared train-test split.")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def process_all(data):
    """Tüm veri ön işleme adımlarını çalıştırır."""
    os.makedirs('outputs', exist_ok=True)
    print("Output directory created successfully.")

    data = visualize_damage_counts(data)
    data = drop_unnecessary_columns(data)
    data = process_engine_features(data)
    data = clean_price(data)
    data = clean_mileage(data)
    data = clean_year(data)
    data = calculate_vehicle_age(data)
    data = handle_outliers(data)
    data = fill_missing_values(data)
    visualize_categorical(data)
    data = process_damage_info(data)
    data = encode_categorical(data)
    data = create_damage_scores(data)
    print("Created damage scores successfully.")
    data = add_derived_features(data)
    data = visualize_price_distribution(data)
    data = scale_features(data)
    save_processed_data(data)
    plot_price_correlations(data)

    return prepare_train_test(data)
