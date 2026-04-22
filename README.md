# Optimizing Imbalanced Clinical Data for Stroke Prediction
## A Comparative Study of Oversampling Techniques and Ensemble Learning Models

### 🎓 MSc Thesis Project Overview
This repository contains the complete experimental framework and results for my MSc thesis. The study addresses the critical challenge of **class imbalance in clinical healthcare data**, specifically applied to the prediction of strokes.

### 🎯 Research Objectives
1.  **Analyze** the impact of 5 oversampling techniques (ROS, SMOTE, ADASYN, etc.) on class distribution.
2.  **Evaluate** 8 ensemble learning algorithms (XGBoost, CatBoost, Random Forest, etc.) across 48 experimental combinations.
3.  **Identify** the optimal model-sampling configuration for clinical sensitivity (Recall) and balanced performance (F1-score).
4.  **Provide** evidence-based recommendations for handling healthcare data imbalance.

### 📊 Key Findings
- **Highest Recall**: CatBoost + Random Oversampling (**0.72**)
- **Highest F1-Score**: XGBoost + Random Oversampling (**0.295**)
- **Best Overall AUC**: **0.84**

### 📁 Repository Structure
- `OPTIMIZING...complete.ipynb`: Main research implementation and pipeline.
- `healthcare-dataset-stroke-data.csv`: Full clinical dataset (5,110 records).
- `final_experimental_results.csv`: Complete performance matrix for all 48 model combinations.
- `img/`: High-resolution figures used in Chapters 3 (Methodology) and 4 (Results).

### 🛠️ Technology Stack
- **Languages**: Python
- **Libraries**: Scikit-Learn, Imbalanced-Learn, XGBoost, CatBoost, LightGBM, Matplotlib, Seaborn.
