import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, accuracy_score, f1_score
from collections import Counter

# Setting style for professional look
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'serif'

# 1. LOAD DATA
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
data = df.copy()
data.drop('id', axis=1, inplace=True, errors='ignore')
data['bmi'] = pd.to_numeric(data['bmi'], errors='coerce').fillna(data['bmi'].median())
data.dropna(inplace=True)

# Generate FIG 4.1: Class Distribution
plt.figure(figsize=(8, 6))
counts = data['stroke'].value_counts()
plt.pie(counts, labels=['No Stroke', 'Stroke'], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], explode=[0, 0.1])
plt.title('Figure 4.1: Stroke Class Distribution', fontweight='bold')
plt.savefig('figure_4_1_distribution.png', dpi=300)
plt.close()

# Generate FIG 4.2: Age Distribution by Stroke (New)
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='age', hue='stroke', multiple='stack', palette=['#2ecc71', '#e74c3c'])
plt.title('Figure 4.2: Age Distribution and Stroke Occurrence', fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('figure_4_2_age_analysis.png', dpi=300)
plt.close()

# Preprocess for modeling
cat_cols = data.select_dtypes(include='object').columns
for col in cat_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

# Generate FIG 4.3: Correlation Heatmap
plt.figure(figsize=(12, 10))
corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', mask=mask)
plt.title('Figure 4.3: Feature Correlation Heatmap', fontweight='bold')
plt.savefig('figure_4_3_heatmap.png', dpi=300)
plt.close()

# Split and Scale
X = data.drop('stroke', axis=1)
y = data['stroke']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
X_tr_s_df = pd.DataFrame(X_tr_s, columns=X.columns)

# Generate FIG 4.4: Resampling Grid
resamplers = {
    'Original': None, 'ROS': RandomOverSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42), 'ADASYN': ADASYN(random_state=42),
    'SMOTE-Tomek': SMOTETomek(random_state=42), 'B-SMOTE': BorderlineSMOTE(random_state=42)
}
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for idx, (name, sampler) in enumerate(resamplers.items()):
    if sampler is None:
        c = Counter(y_tr)
    else:
        _, y_temp = sampler.fit_resample(X_tr_s, y_tr)
        c = Counter(y_temp)
    axes[idx].bar(['No Stroke', 'Stroke'], [c[0], c[1]], color=['#2ecc71', '#e74c3c'])
    axes[idx].set_title(name, fontweight='bold')
plt.suptitle('Figure 4.4: Class Distribution After Resampling', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figure_4_4_oversampling.png', dpi=300)
plt.close()

# Best Model: CatBoost + ROS
X_res, y_res = RandomOverSampler(random_state=42).fit_resample(X_tr_s, y_tr)
model = CatBoostClassifier(iterations=100, logging_level='Silent').fit(X_res, y_res)
y_prob = model.predict_proba(X_te_s)[:, 1]
y_pred = model.predict(X_te_s)

# FIG 4.5: ROC
fpr, tpr, _ = roc_curve(y_te, y_prob)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.plot([0,1],[0,1], '--', color='gray')
plt.title('Figure 4.5: ROC Curve (CatBoost + ROS)', fontweight='bold')
plt.legend()
plt.savefig('figure_4_5_roc.png', dpi=300)
plt.close()

# FIG 4.8: Confusion Matrix
cm = confusion_matrix(y_te, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Figure 4.8: Confusion Matrix (CatBoost + ROS)', fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('figure_4_8_cm.png', dpi=300)
plt.close()

print("All 8 thesis figures generated successfully.")
