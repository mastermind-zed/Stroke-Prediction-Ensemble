import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

# 1. Load Data
print("Loading healthcare-dataset-stroke-data.csv...")
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
print(f"Data shape: {df.shape}")

# 2. Preprocessing
print("Preprocessing data...")
data = df.copy()
data.drop('id', axis=1, inplace=True, errors='ignore')
data['bmi'] = pd.to_numeric(data['bmi'], errors='coerce')
data['bmi'] = data['bmi'].fillna(data['bmi'].median())

# Drop rows with any remaining NaNs just in case (should be 0)
data.dropna(inplace=True)

categorical_cols = data.select_dtypes(include='object').columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

X = data.drop('stroke', axis=1)
y = data['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
numerical_cols = ['age', 'avg_glucose_level', 'bmi']
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# 3. Define Resamplers
oversampling_techniques = {
    'Original': None,
    'Random Oversampling': RandomOverSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42, k_neighbors=5),
    'ADASYN': ADASYN(random_state=42),
    'SMOTE-Tomek': SMOTETomek(random_state=42),
    'Borderline-SMOTE': BorderlineSMOTE(random_state=42, kind='borderline-1')
}

# 4. Define Models
def get_models():
    models = {}
    models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    models['XGBoost'] = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)
    models['LightGBM'] = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
    models['CatBoost'] = CatBoostClassifier(iterations=100, learning_rate=0.1, random_state=42, verbose=0)
    models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    models['AdaBoost'] = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)),
        ('lgbm', LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)),
        ('catboost', CatBoostClassifier(iterations=100, learning_rate=0.1, random_state=42, verbose=0))
    ]
    models['Voting Classifier'] = VotingClassifier(estimators=base_models, voting='soft')
    models['Stacking Classifier'] = StackingClassifier(estimators=base_models[:-1], final_estimator=LogisticRegression(random_state=42), cv=5)
    return models

# 5. Execute Pipeline
results = []
models_dict = get_models()

print("\nStarting Experimental Pipeline (48 combinations)...")
for sampling_name, sampler in oversampling_techniques.items():
    print(f"Applying {sampling_name}...")
    if sampler is None:
        X_resampled, y_resampled = X_train_scaled.values, y_train.values
    else:
        X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)
    
    for model_name, model in models_dict.items():
        start_time = time.time()
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test_scaled)
        try:
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        except:
            y_prob = model.decision_function(X_test_scaled)
            
        elapsed = time.time() - start_time
        
        results.append({
            'Sampling Technique': sampling_name,
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob),
            'Avg Precision': average_precision_score(y_test, y_prob),
            'Inference Time (s)': elapsed
        })

# 6. Save and Report
results_df = pd.DataFrame(results)
results_df.to_csv('final_experimental_results.csv', index=False)
print("\n✅ Experiments complete. Results saved to 'final_experimental_results.csv'.")

# Final Ranking (Objective iii)
print("\nTop 5 Combinations (Ranked by F1-Score):")
print(results_df.sort_values(by='F1-Score', ascending=False).head(5).to_string(index=False))
