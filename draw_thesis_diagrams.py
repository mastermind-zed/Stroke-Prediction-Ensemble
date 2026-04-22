import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 140)
    ax.axis('off')

    # Helper function for boxes
    def draw_box(x, y, w, h, text, color='#E8F0FE'):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                     linewidth=1.5, edgecolor='#1967D2', facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=11, fontweight='bold', wrap=True)

    # Helper for arrows
    def draw_arrow(x, y, dx, dy):
        ax.arrow(x, y, dx, dy, head_width=1.5, head_length=2, fc='#1967D2', ec='#1967D2')

    # Draw Boxes
    draw_box(30, 130, 40, 7, "1. DATA SOURCE\nStroke Dataset (5,110 records)")
    draw_arrow(50, 130, 0, -8)
    
    draw_box(30, 115, 40, 7, "2. PREPROCESSING\nImputation, Encoding, Scaling")
    draw_arrow(50, 115, 0, -8)

    draw_box(30, 100, 40, 7, "3. DATA SPLITTING\n80% Train / 20% Test")
    draw_arrow(50, 100, 0, -8)

    draw_box(20, 80, 60, 12, "4. DATA OPTIMIZATION (OVERSAMPLING)\nROS | SMOTE | ADASYN | S-Tomek | B-SMOTE", color='#FFF4E5')
    draw_arrow(50, 80, 0, -8)

    draw_box(20, 60, 60, 12, "5. ENSEMBLE MODELING\nRF, XGB, LGBM, CatB, GBM, AdaB, Voting, Stacking", color='#FFF4E5')
    draw_arrow(50, 60, 0, -8)

    draw_box(30, 45, 40, 7, "6. EVALUATION\nRecall, F1, ROC-AUC")
    draw_arrow(50, 45, 0, -8)

    draw_box(30, 30, 40, 7, "7. COMPARATIVE ANALYSIS")
    draw_arrow(50, 30, 0, -8)

    draw_box(30, 15, 40, 7, "8. FINAL RECOMMENDATIONS", color='#E6FFED')

    plt.title("Figure 3.1: Methodological Flowchart of the Study", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('figure_3_1_flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()

def draw_conceptual_framework():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.axis('off')

    # IV Box
    rect_iv = patches.FancyBboxPatch((10, 20), 30, 40, boxstyle="round,pad=0.2", facecolor='#F0F0F0', edgecolor='black')
    ax.add_patch(rect_iv)
    ax.text(25, 62, "INDEPENDENT VARIABLES\n(Input)", ha='center', fontsize=12, fontweight='bold')
    ax.text(25, 40, "- Clinical Factors\n- Imbalance Severity\n- Oversampling Choice", ha='center', va='center', fontsize=10)

    # Mediating Box
    rect_med = patches.FancyBboxPatch((45, 25), 30, 30, boxstyle="round,pad=0.2", facecolor='#E8F0FE', edgecolor='#1967D2')
    ax.add_patch(rect_med)
    ax.text(60, 57, "MEDIATING VARIABLES\n(Process)", ha='center', fontsize=12, fontweight='bold')
    ax.text(60, 40, "- Preprocessing Steps\n- Ensemble Algorithms", ha='center', va='center', fontsize=10)

    # DV Box
    rect_dv = patches.FancyBboxPatch((80, 20), 30, 40, boxstyle="round,pad=0.2", facecolor='#E6FFED', edgecolor='#28A745')
    ax.add_patch(rect_dv)
    ax.text(95, 62, "DEPENDENT VARIABLES\n(Output)", ha='center', fontsize=12, fontweight='bold')
    ax.text(95, 40, "- Recall (Sensitivity)\n- F1-Score\n- Accuracy\n- ROC-AUC", ha='center', va='center', fontsize=10)

    # Arrows
    ax.arrow(40, 40, 4, 0, head_width=2, head_length=3, fc='black', ec='black')
    ax.arrow(75, 40, 4, 0, head_width=2, head_length=3, fc='black', ec='black')

    plt.title("Figure 3.2: Conceptual Framework of the Research", fontsize=14, fontweight='bold', pad=30)
    plt.tight_layout()
    plt.savefig('figure_3_2_conceptual_framework.png', dpi=300, bbox_inches='tight')
    plt.close()

draw_flowchart()
draw_conceptual_framework()
print("Conceptual diagrams generated successfully.")
