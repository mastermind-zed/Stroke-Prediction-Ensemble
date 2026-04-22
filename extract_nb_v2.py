import json
import sys

def extract_notebook_code(ipynb_path, output_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                f.write('\n# --- Code Cell ---\n')
                f.write(''.join(cell.get('source', [])))
                f.write('\n')
            elif cell.get('cell_type') == 'markdown':
                f.write('\n### Markdown: ')
                f.write(''.join(cell.get('source', [])))
                f.write('\n')

if __name__ == "__main__":
    path = "d:\\Machine Learning\\Strokeprediction\\OPTIMIZING IMBALANCED CLINICAL DATA FOR STROKE PREDICTION complete.ipynb"
    out = "notebook_content_utf8.txt"
    extract_notebook_code(path, out)
    print(f"Extraction complete to {out}")
