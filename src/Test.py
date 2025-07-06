import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

MODELS_DIR = 'models'
OUTPUTS_DIR = 'outputs'

def load_artifacts():
    artifacts = {}
    try:
        artifacts['linear_model'] = joblib.load(os.path.join(MODELS_DIR, 'LinearRegression.joblib'))
        artifacts['ridge_model'] = joblib.load(os.path.join(MODELS_DIR, 'Ridge.joblib'))
        artifacts['scaler'] = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        artifacts['numerical_medians'] = joblib.load(os.path.join(MODELS_DIR, 'numerical_medians.joblib'))
        artifacts['categorical_modes'] = joblib.load(os.path.join(MODELS_DIR, 'categorical_modes.joblib'))
        artifacts['damage_parts'] = joblib.load(os.path.join(MODELS_DIR, 'damage_parts.joblib'))
        artifacts['score_important'] = pd.read_csv(os.path.join(OUTPUTS_DIR, 'ScoreImportant.csv'))

        artifacts['label_encoders'] = {}
        for col in ['transmission', 'fuelType', 'color', 'model', 'location']:
            encoder_path = os.path.join(MODELS_DIR, f'label_encoder_{col}.joblib')
            if os.path.exists(encoder_path):
                artifacts['label_encoders'][col] = joblib.load(encoder_path)
            else:
                print(f"Uyarı: {col} için LabelEncoder bulunamadı.")

        with open(os.path.join(OUTPUTS_DIR, 'SelectedFeatures.txt'), 'r') as f:
            artifacts['selected_features'] = [line.strip() for line in f if line.strip()]

        print("Tüm modeller ve önişleme yapıtları başarıyla yüklendi.")
    except Exception as e:
        print(f"Yapıtlar yüklenirken hata oluştu: {e}")
        return None
    return artifacts

def preprocess_input_data(input_df, artifacts):
    df = input_df.copy()

    def clean_engine_power(x):
        if pd.isna(x): return x
        x = str(x).lower().replace('hp', '').strip()
        if '-' in x:
            try:
                return np.mean([float(i.strip()) for i in x.split('-')])
            except: return np.nan
        try: return float(x)
        except: return np.nan

    def clean_engine_capacity(x):
        if pd.isna(x): return x
        x = str(x).lower().replace('cm3', '').strip()
        if '-' in x:
            try:
                return np.mean([float(i.strip()) for i in x.split('-')])
            except: return np.nan
        try: return float(x)
        except: return np.nan

    if 'title' in df.columns:
        df.drop('title', axis=1, inplace=True)

    if 'enginePower' in df.columns:
        df['enginePower'] = df['enginePower'].apply(clean_engine_power)
    if 'engineCapacity' in df.columns:
        df['engineCapacity'] = df['engineCapacity'].apply(clean_engine_capacity)

    if 'mileage' in df.columns:
        df['mileage'] = df['mileage'].astype(str).str.replace('.', '', regex=False).str.strip()
        df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')

    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['vehicle_age'] = 2025 - df['year']
        df.drop('year', axis=1, inplace=True)
    else:
        df['vehicle_age'] = np.nan

    required_cols = ['mileage', 'vehicle_age', 'enginePower', 'engineCapacity',
                     'transmission', 'fuelType', 'color', 'model', 'location', 'damageInfo']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    for col, val in artifacts['numerical_medians'].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    for col, val in artifacts['categorical_modes'].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    if 'damageInfo' in df.columns:
        for part in artifacts['damage_parts']:
            df[f'{part}_changed'] = df['damageInfo'].apply(
                lambda x: int(any(isinstance(i, dict) and ((isinstance(i.get('changed'), dict) and i['changed'].get(part)) or
                                                           (isinstance(i.get('changed'), list) and part in i['changed']))
                                  for i in (x if isinstance(x, list) else [x]))) if isinstance(x, (list, dict)) else 0)
            df[f'{part}_painted'] = df['damageInfo'].apply(
                lambda x: int(any(isinstance(i, dict) and ((isinstance(i.get('painted'), dict) and i['painted'].get(part)) or
                                                           (isinstance(i.get('painted'), list) and part in i['painted']))
                                  for i in (x if isinstance(x, list) else [x]))) if isinstance(x, (list, dict)) else 0)
        df.drop('damageInfo', axis=1, inplace=True)

    for col in [f'{p}_changed' for p in artifacts['damage_parts']] + [f'{p}_painted' for p in artifacts['damage_parts']]:
        if col not in df.columns:
            df[col] = 0

    damage_cols = [col for col in df.columns if col.endswith('_changed') or col.endswith('_painted')]
    importance = artifacts['score_important'].set_index('feature')['importance'].to_dict()
    df['damage_score'] = df[damage_cols].apply(lambda row: sum(row[c] * importance.get(c, 0) for c in damage_cols), axis=1)
    df.drop(columns=damage_cols, inplace=True)

    df['power_to_capacity_ratio'] = df['enginePower'] / df['engineCapacity']
    df['age_to_mileage_ratio'] = df['vehicle_age'] / (df['mileage'] + 1)

    for col in ['power_to_capacity_ratio', 'age_to_mileage_ratio']:
        if df[col].isnull().any():
            df[col] = df[col].fillna(artifacts['numerical_medians'].get(col, 0))

    for col, encoder in artifacts['label_encoders'].items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
            df[col] = encoder.transform(df[col])

    scale_cols = list(artifacts['scaler'].feature_names_in_)
    df_scaled = pd.DataFrame(index=df.index)
    for col in scale_cols:
        df_scaled[col] = pd.to_numeric(df[col], errors='coerce').fillna(artifacts['numerical_medians'].get(col, 0))

    df_scaled[scale_cols] = artifacts['scaler'].transform(df_scaled[scale_cols])
    df[scale_cols] = df_scaled[scale_cols]

    final_features = artifacts['selected_features']
    return df.reindex(columns=final_features, fill_value=0)

def make_predictions(preprocessed_data, artifacts):
    linear_pred_log = artifacts['linear_model'].predict(preprocessed_data)[0]
    ridge_pred_log = artifacts['ridge_model'].predict(preprocessed_data)[0]
    avg_pred_log = (linear_pred_log + ridge_pred_log) / 2

    predicted_price_linear = np.expm1(linear_pred_log)
    predicted_price_ridge = np.expm1(ridge_pred_log)
    predicted_price_avg = np.expm1(avg_pred_log)

    print(f"\nLinear Regression Tahmini (Log Ölçeği): {linear_pred_log:.4f}")
    print(f"Ridge Regression Tahmini (Log Ölçeği): {ridge_pred_log:.4f}")
    print(f"Lineer Regresyon Tahmini (TL): {predicted_price_linear:,.2f} TL")
    print(f"Ridge Regresyon Tahmini (TL): {predicted_price_ridge:,.2f} TL")
    print(f"\nTahmini Fiyat: {predicted_price_avg:,.2f} TL\n")

    return predicted_price_avg

def get_user_input():
    user_input = {}
    print("\nLütfen aşağıdaki araç detaylarını giriniz:")

    questions = {
        'mileage': 'Kilometre (örn. 150000): ',
        'year': 'Yıl (örn. 2018): ',
        'enginePower': 'Motor Gücü (örn. 120 HP): ',
        'engineCapacity': 'Motor Hacmi (örn. 1500 cm3): ',
        'transmission': 'Şanzıman (örn. Otomatik): ',
        'fuelType': 'Yakıt Tipi (örn. Benzin): ',
        'color': 'Renk (örn. Beyaz): ',
        'model': 'Model (örn. Civic): ',
        'location': 'Konum (örn. İstanbul): '
    }

    for feature, prompt in questions.items():
        val = input(prompt).strip()
        user_input[feature] = val if val else np.nan

    painted_input = input("Boyalı Parçalar (örn. Kaput, Tavan): ").strip()
    changed_input = input("Değişen Parçalar (örn. Sağ Kapı, Sol Çamurluk): ").strip()

    painted_parts = [p.strip().title() for p in painted_input.split(',') if p.strip()]
    changed_parts = [p.strip().title() for p in changed_input.split(',') if p.strip()]

    damage_list = []
    for part in painted_parts:
        damage_list.append({'painted': {part: True}})
    for part in changed_parts:
        damage_list.append({'changed': {part: True}})

    user_input['damageInfo'] = damage_list if damage_list else np.nan

    return user_input

def main():
    artifacts = load_artifacts()
    if not artifacts:
        print("Yapıtlar yüklenemedi. Çıkılıyor...")
        return

    print("\n--- Araç Fiyat Tahmin Aracı ---")
    while True:
        user_input_dict = get_user_input()
        input_df = pd.DataFrame([user_input_dict])

        try:
            processed = preprocess_input_data(input_df, artifacts)
            make_predictions(processed, artifacts)
        except Exception as e:
            print(f"Tahmin sırasında hata oluştu: {e}")

        again = input("Yeni bir tahmin yapmak ister misiniz? (evet/hayır): ").strip().lower()
        if again != 'evet':
            break

if __name__ == '__main__':
    main()
