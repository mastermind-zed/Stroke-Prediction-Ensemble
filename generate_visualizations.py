import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Load and Preprocess again for the best model plots
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
data = df.copy()
data.drop('id', axis=1, inplace=True, errors='ignore')
data['bmi'] = pd.to_numeric(data['bmi'], errors='coerce')
data['bmi'] = data['bmi'].fillna(data['bmi'].median())
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

# Best Model: CatBoost + ROS (High Recall & decent F1)
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train_scaled, y_train)

model = CatBoostClassifier(iterations=100, learning_rate=0.1, random_state=42, verbose=0)
model.fit(X_res, y_res)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Figure 4.5: ROC Curve for Best Performing Model (CatBoost + ROS)')
plt.legend(loc="lower right")
plt.savefig('figure_4_5_roc_curve.png', dpi=300)
plt.close()

# 2. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Figure 4.6: Precision-Recall Curve (CatBoost + ROS)')
plt.savefig('figure_4_6_pr_curve.png', dpi=300)
plt.close()

# 3. Summary Results Comparison Plot
results_df = pd.read_csv('final_experimental_results.csv')
plt.figure(figsize=(12, 6))
top_10 = results_df.sort_values(by='F1-Score', ascending=False).head(10)
sns.barplot(x='F1-Score', y='Model', hue='Sampling Technique', data=top_10)
plt.title('Figure 4.7: Top 10 Model-Sampling Combinations by F1-Score')
plt.savefig('figure_4_7_top_models.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualizations saved successfully.")
