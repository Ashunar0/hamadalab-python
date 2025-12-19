
import json
import os

notebook_path = '/Users/ryugohanai/Documents/mlb_war_project/hamadalab-python/pitch_classification_hanai2copy.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find index of the cell with "# 8.2 裁量モデル（Discretionary Model）"
index_to_remove = -1
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if "# 8.2 裁量モデル（Discretionary Model）" in source:
            index_to_remove = i
            break

if index_to_remove != -1:
    print(f"Found old Section 8.2 at index {index_to_remove}. Removing it and the following code cell.")
    # Remove the markdown cell and the code cell immediately following it
    # We remove index_to_remove twice because after the first removal, the next cell shifts to that index
    notebook['cells'].pop(index_to_remove)
    notebook['cells'].pop(index_to_remove) 
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print("Successfully cleanup up notebook.")
else:
    print("Old Section 8.2 not found. No changes made.")
