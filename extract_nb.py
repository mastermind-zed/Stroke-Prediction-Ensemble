import json
import sys

def extract_notebook_code(ipynb_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            code_cells.append(''.join(cell.get('source', [])))
        elif cell.get('cell_type') == 'markdown':
            code_cells.append('\n### Markdown: ' + ''.join(cell.get('source', [])) + '\n')
            
    return '\n'.join(code_cells)

if __name__ == "__main__":
    path = "d:\\Machine Learning\\Strokeprediction\\OPTIMIZING IMBALANCED CLINICAL DATA FOR STROKE PREDICTION complete.ipynb"
    code = extract_notebook_code(path)
    print(code[:5000]) # Print first 5000 chars to see structure
