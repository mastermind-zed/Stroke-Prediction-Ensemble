import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

def draw_final_framework():
    fig, ax = plt.subplots(figsize=(20, 24))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 110)
    ax.axis('off')

    # Colors
    c_input = '#1A3355' # Dark Blue
    c_iv1 = '#8D6E63'   # Brown
    c_iv2 = '#673AB7'   # Purple
    c_dv = '#B18904'    # Golden/Yellow
    c_out = '#1B5E20'   # Dark Green
    c_highlight = '#C62828' # Red for Clinical Priority

    # Helper function for boxes
    def draw_rounded_box(x, y, w, h, text, color, text_color='white', lw=1.5, fontsize=12, bold=True):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                     linewidth=lw, edgecolor=color, facecolor=color)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, color=text_color, fontweight=weight, wrap=True)

    def draw_outline_box(x, y, w, h, text, color, fontsize=10):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                     linewidth=1.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, color='black', wrap=True)

    # 1. INPUT SECTION (TOP)
    draw_rounded_box(20, 102, 60, 5, "Imbalanced Clinical Stroke Dataset\n5110 records | 11 features | Binary target (stroke: 0/1)", c_input)
    draw_outline_box(10, 92, 25, 7, "Class Distribution\nNon-Stroke: 95.1% | Stroke: 4.9%", 'gray')
    draw_outline_box(37.5, 92, 25, 7, "Clinical Features\nAge, BMI, Glucose, Hypertension, etc.", 'gray')
    draw_outline_box(65, 92, 25, 7, "Core Problem\nImbalance ratio ~ 19:1 biases classifiers", '#C62828')

    # Interaction Bar
    draw_rounded_box(30, 83, 40, 3, "INTERACTION EFFECT (IV₁ × IV₂ → DV)", '#0D1B2A', fontsize=11)

    # 2. INDEPENDENT VARIABLES
    # IV1 Track (Left)
    draw_rounded_box(10, 78, 38, 3, "Independent Variable 1 (IV₁)", c_iv1)
    ax.text(29, 76, "Oversampling Techniques\n(Data-Level Processing)", ha='center', fontsize=11, style='italic', color=c_iv1)
    # Sampling boxes
    draw_rounded_box(12, 70, 16, 4, "Original (Baseline)", c_iv1, fontsize=10)
    draw_rounded_box(30, 70, 16, 4, "Random Oversampling", c_iv1, fontsize=10)
    draw_rounded_box(12, 65, 16, 4, "SMOTE", c_iv1, fontsize=10)
    draw_rounded_box(30, 65, 16, 4, "ADASYN", c_iv1, fontsize=10)
    draw_rounded_box(12, 60, 16, 4, "SMOTE-Tomek", c_iv1, fontsize=10)
    draw_rounded_box(30, 60, 16, 4, "Borderline-SMOTE", c_iv1, fontsize=10)
    draw_outline_box(12, 52, 34, 7, "Mechanism:\nModify training distribution to balance classes\nApplied only to training set (~3,888 samples)", c_iv1, fontsize=9)

    # IV2 Track (Right)
    draw_rounded_box(52, 78, 38, 3, "Independent Variable 2 (IV₂)", c_iv2)
    ax.text(71, 76, "Ensemble Learning Models\n(Algorithm-Level Processing)", ha='center', fontsize=11, style='italic', color=c_iv2)
    # Model boxes
    models = ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "Gradient Boosting", "AdaBoost", "Voting Classifier", "Stacking Classifier"]
    for i, m in enumerate(models):
        row = i // 2
        col = i % 2
        x_pos = 54 + (col * 18)
        y_pos = 70 - (row * 5)
        draw_rounded_box(x_pos, y_pos, 16, 4, m, c_iv2, fontsize=10)
    draw_outline_box(54, 48, 34, 5, "Mechanism:\nCombine multiple learners for robust prediction", c_iv2, fontsize=9)

    # X Symbol in middle
    ax.text(50, 62, "X", ha='center', va='center', fontsize=20, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle'))
    ax.text(50, 58, "6 × 8 = 48\ncombinations", ha='center', va='center', fontsize=9)

    # 3. EVALUATION SECTION
    draw_rounded_box(30, 38, 40, 3, "Dependent Variable (DV): Classification Performance Metrics", c_dv)
    metrics = [
        (15, 30, "Accuracy", "Overall correctness"),
        (32, 30, "Precision", "Positive predictive value"),
        (49, 30, "Recall", "Clinical priority metric", c_highlight),
        (66, 30, "F1-Score", "Balanced measure", c_highlight)
    ]
    for x, y, name, desc, *color in metrics:
        col = color[0] if color else c_dv
        draw_rounded_box(x, y, 15, 5, f"{name}\n{desc}", col, fontsize=10)
    
    draw_rounded_box(25, 24, 23, 5, "ROC-AUC\nDiscriminative ability", c_dv, fontsize=10)
    draw_rounded_box(52, 24, 23, 5, "Average Precision\nPrecision-Recall curve area", c_dv, fontsize=10)

    # 4. OUTPUT SECTION
    draw_rounded_box(30, 16, 40, 3, "Research Output: Evidence-Based Optimal Pipeline", c_out)
    draw_rounded_box(15, 7, 34, 7, "Best Balanced Performance (F1)\nXGBoost + Random Oversampling\nF1=0.295 | AUC=0.81 | Recall=0.38", c_out, fontsize=11)
    draw_rounded_box(51, 7, 34, 7, "Best Clinical Sensitivity (Recall)\nCatBoost + Random Oversampling\nRecall=0.720 | F1=0.279 | AUC=0.84", c_out, fontsize=11)
    
    draw_outline_box(25, 1, 50, 5, "Contribution: Evidence-based guidance for clinicians\non optimal resampling + ensemble combination", c_out, fontsize=10)

    # Connectors
    # Input to Interac
    ax.add_line(lines.Line2D([50, 50], [92, 86], color=c_input, lw=2))
    # IVs to DV
    ax.add_line(lines.Line2D([30, 50], [52, 41], color=c_iv1, lw=2))
    ax.add_line(lines.Line2D([70, 50], [48, 41], color=c_iv2, lw=2))

    plt.tight_layout()
    plt.savefig('figure_conceptual_framework_final.png', dpi=300, bbox_inches='tight')
    plt.close()

draw_final_framework()
print("Final custom conceptual framework generated successfully.")
