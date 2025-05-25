import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, mean_squared_error
from datetime import datetime
import xgboost as xgb
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# --- Fonction pour calculer la durée de vie restante (RUL) en kilomètres ---
max_kilometrage_dict = {
    'Corolla': 300000, '3 Series': 250000, 'Civic': 280000, 'X5': 240000,
    'Focus': 260000, 'Mustang': 220000, 'Camry': 290000, 'A4': 250000,
    'Golf': 270000, 'Model 3': 350000
}

def calculate_rul(model_name, year, engine_replacement_year, mileage_km, co2_emission_g_per_km, temperature, fuel_consumption):
    current_year = datetime.now().year
    engine_age = current_year - engine_replacement_year if engine_replacement_year != 0 else current_year - year
    
    alpha, beta, gamma, delta, epsilon = 1.0, 0.05, 0.1, 0.05, 0.2
    C = 5 if fuel_consumption == 'Élevée' else 0
    T_penalty = max(0, temperature - 40)
    
    penalite_annees = (alpha * engine_age + beta * (mileage_km / 1000) +
                       gamma * (co2_emission_g_per_km / 100) + delta * T_penalty + epsilon * C)
    
    duree_vie_attendue = 20
    vie_utile_restante_annees = max(0, duree_vie_attendue - penalite_annees)
    
    max_km = max_kilometrage_dict.get(model_name, 250000)
    km_per_year = max_km / duree_vie_attendue
    RUL = max(0, round(vie_utile_restante_annees * km_per_year - mileage_km, 2))
    
    return RUL

# --- Fonction pour assigner la durée de vie maximale (pour XGBoost) ---
def assign_max_lifespan(row):
    if row['fuel_type'] == 'Electric':
        return 25
    elif row['fuel_type'] == 'Diesel':
        return 22
    elif row['fuel_type'] == 'Hybrid':
        return 18
    else:
        if row['engine_type'] in ['V8', 'V6']:
            return 15
        else:
            return 18

# --- Charger et préparer les données ---
file_path = r"C:\Users\Pc\Desktop\nasri\ccc.csv"
df = pd.read_csv(file_path)
print("Columns in dataset:", df.columns.tolist())

# Sauvegarder les valeurs catégoriques originales pour les dropdowns
categorical_originals = {col: df[col].unique().tolist() for col in ['brand', 'model', 'engine_type', 'fuel_type', 'maintenance_frequency', 'usage_type', 'engine_replaced', 'fault_type'] if col in df.columns}

# Ajouter max_lifespan et remaining_engine_lifespan pour XGBoost
df['max_lifespan'] = df.apply(assign_max_lifespan, axis=1)
df['remaining_engine_lifespan'] = df['max_lifespan'] - df['engine_age']

# --- Modèle RandomForestClassifier (prédiction de l'état de la voiture) ---
categorical_columns = ['brand', 'model', 'engine_type', 'fuel_type', 'maintenance_frequency', 'usage_type', 'engine_replaced', 'fault_type']
le_dict = {}
for col in categorical_columns + ['car_state']:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    else:
        print(f"Warning: Column {col} not found in dataset")

features_rf = ['brand', 'model', 'year', 'mileage_km', 'engine_type', 'fuel_type', 'horsepower',
               'maintenance_frequency', 'usage_type', 'engine_replaced', 'engine_replacement_year',
               'engine_age', 'co2_emission_g_per_km', 'fault_type']
features_rf = [col for col in features_rf if col in df.columns]
print("RandomForest features:", features_rf)  # Debug

X_rf = df[features_rf]
y_rf = df['car_state']

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_rf, y_train_rf)

y_pred_rf = model_rf.predict(X_test_rf)
print("RandomForest Accuracy:", accuracy_score(y_test_rf, y_pred_rf))
print(classification_report(y_test_rf, y_pred_rf, target_names=le_dict['car_state'].classes_))

# --- Modèle XGBoost (prédiction de la durée de vie restante en années) ---
X_xgb = df.drop(columns=['car_id', 'engine_age', 'remaining_engine_lifespan', 'max_lifespan', 'car_state'])
y_xgb = df['remaining_engine_lifespan']

categorical_cols = ['brand', 'model', 'engine_type', 'fuel_type', 'maintenance_frequency', 'usage_type', 'engine_replaced']
numerical_cols = ['year', 'mileage_km', 'horsepower', 'engine_replacement_year', 'co2_emission_g_per_km']
categorical_cols = [col for col in categorical_cols if col in df.columns]
numerical_cols = [col for col in numerical_cols if col in df.columns]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])

pipeline_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
])

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42)
pipeline_xgb.fit(X_train_xgb, y_train_xgb)

model_path = r"C:\Users\Pc\Desktop\nasri\engine_lifespan_model.pkl"
joblib.dump(pipeline_xgb, model_path)

y_pred_xgb = pipeline_xgb.predict(X_test_xgb)
mae = mean_absolute_error(y_test_xgb, y_pred_xgb)
rmse = np.sqrt(mean_squared_error(y_test_xgb, y_pred_xgb))
print(f"XGBoost MAE: {mae:.2f} years")
print(f"XGBoost RMSE: {rmse:.2f} years")

# --- Routes Flask ---
@app.route('/')
def home():
    return render_template('index.html', 
                         brands=categorical_originals.get('brand', []),
                         models=categorical_originals.get('model', []),
                         engine_types=categorical_originals.get('engine_type', []),
                         fuel_types=categorical_originals.get('fuel_type', []),
                         maintenance_frequencies=categorical_originals.get('maintenance_frequency', []),
                         usage_types=categorical_originals.get('usage_type', []),
                         engine_replaced_options=categorical_originals.get('engine_replaced', []),
                         fault_types=categorical_originals.get('fault_type', []))

@app.route('/predict', methods=['POST'])
def predict():
    # Charger le modèle XGBoost
    model_xgb = joblib.load(model_path)
    
    # Obtenir les données du formulaire
    car_data = {
        'brand': request.form['brand'],
        'model': request.form['model'],
        'year': int(request.form['year']),
        'mileage_km': float(request.form['mileage_km']),
        'engine_type': request.form['engine_type'],
        'fuel_type': request.form['fuel_type'],
        'horsepower': float(request.form['horsepower']),
        'maintenance_frequency': request.form['maintenance_frequency'],
        'usage_type': request.form['usage_type'],
        'engine_replaced': request.form['engine_replaced'],
        'engine_replacement_year': int(request.form['engine_replacement_year']) if request.form.get('engine_replacement_year') else 0,
        'co2_emission_g_per_km': float(request.form['co2_emission_g_per_km']),
        'fault_type': request.form['fault_type']
    }
    
    temperature = float(request.form['temperature'])
    radiator_leak = request.form['radiator_leak']
    noise = request.form['noise']
    fuel_consumption = request.form['fuel_consumption']
    
    # Calculer engine_age pour RandomForest
    current_year = datetime.now().year
    car_data['engine_age'] = current_year - car_data['engine_replacement_year'] if car_data['engine_replacement_year'] != 0 else current_year - car_data['year']
    
    # --- Prédiction XGBoost (durée de vie en années) ---
    input_df_xgb = pd.DataFrame([{k: v for k, v in car_data.items() if k in X_xgb.columns}])
    lifespan_years = model_xgb.predict(input_df_xgb)[0]
    
    # --- Prédiction RandomForest (état de la voiture) ---
    car_data_rf = car_data.copy()
    for col in categorical_columns:
        if col in le_dict and car_data_rf[col] in le_dict[col].classes_:
            car_data_rf[col] = le_dict[col].transform([car_data_rf[col]])[0]
        elif col in le_dict:
            car_data_rf[col] = le_dict[col].transform([le_dict[col].classes_[0]])[0]
        else:
            car_data_rf[col] = 0
    print("car_data_rf keys:", list(car_data_rf.keys()))  # Debug
    input_df_rf = pd.DataFrame({col: [car_data_rf.get(col, 0)] for col in features_rf})
    print("input_df_rf columns:", input_df_rf.columns.tolist())  # Debug
    pred_rf = model_rf.predict(input_df_rf)[0]
    car_state = le_dict['car_state'].inverse_transform([pred_rf])[0]
    
    # --- Calcul RUL (en kilomètres) ---
    RUL = calculate_rul(car_data['model'], car_data['year'], car_data['engine_replacement_year'],
                        car_data['mileage_km'], car_data['co2_emission_g_per_km'], temperature, fuel_consumption)
    
    # --- Messages d'avertissement ---
    warning_messages = []
    if temperature > 50 or radiator_leak == 'Oui':
        warning_messages.append("Avertissement : Problème de refroidissement - Température élevée ou fuite dans le radiateur détectée.")
    if noise == 'Oui' and fuel_consumption == 'Élevée':
        warning_messages.append("Avertissement : Manque de graissage - Bruit et consommation de carburant élevée détectés.")
    if car_data['co2_emission_g_per_km'] > 200:
        warning_messages.append("Avertissement : Problème de combustion - Émissions de CO2 élevées détectées.")
    
    # Limiter à deux avertissements maximum
    warning_messages = warning_messages[:2]
    
    # Déterminer l'état de la voiture
    if len(warning_messages) == 0:
        car_condition = "Bon état"
    elif len(warning_messages) == 1:
        car_condition = "Moyen état"
    elif len(warning_messages) == 2:
        car_condition = "Très mauvais état"
    
    return render_template('result.html', 
                         lifespan_years=round(float(lifespan_years), 2),
                         car_state=car_state,
                         RUL=RUL,
                         warning_messages=warning_messages,
                         car_condition=car_condition)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)