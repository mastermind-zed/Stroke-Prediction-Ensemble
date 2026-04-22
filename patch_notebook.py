import json
import os

notebook_path = "OPTIMIZING IMBALANCED CLINICAL DATA FOR STROKE PREDICTION complete.ipynb"

if os.path.exists(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            
            # 1. Update data loading to full dataset
            if 'full_filled_stroke_data (1).csv' in source_str:
                cell['source'] = [
                    "# Load the FULL dataset (5,110 records)\n",
                    "df = pd.read_csv('healthcare-dataset-stroke-data.csv')\n",
                    "\n",
                    "print(f\"Dataset Shape: {df.shape}\")\n",
                    "print(f\"Total Records: {df.shape[0]}\")\n",
                    "print(f\"Total Features: {df.shape[1]}\")\n",
                    "df.head()"
                ]
            
            # 2. Uncomment 'id' column drop
            if 'data.drop(\'id\', axis=1, inplace=True)' in source_str:
                new_source = []
                for line in cell['source']:
                    if "data.drop('id', axis=1, inplace=True)" in line:
                        new_source.append("data.drop('id', axis=1, inplace=True)\n")
                    elif "Dropped 'id' column. Remaining features:" in line:
                        new_source.append("print(f\"\\nDropped 'id' column. Remaining features: {data.shape[1]-1}\")\n")
                    else:
                        new_source.append(line)
                cell['source'] = new_source

            # 3. Ensure BMI numeric conversion is robust
            if "data['bmi'] = pd.to_numeric(data['bmi'], errors='coerce')" in source_str:
                # This is already good, just keeping it.
                pass

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully with full dataset configuration.")
else:
    print("Notebook file not found.")
