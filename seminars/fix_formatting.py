#!/usr/bin/env python3
"""
Исправление форматирования кода в ноутбуке.
Разбивает код на отдельные строки для правильного отображения в Jupyter.
"""

import json
from pathlib import Path

def fix_cell_formatting(cell):
    """Fix formatting of a single cell."""
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])

        # If source is a list with single string, split it
        if isinstance(source, list) and len(source) == 1:
            # Split by newlines and add \n to each line
            lines = source[0].split('\n')
            # Add \n to each line except the last one
            cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
        elif isinstance(source, str):
            # If source is a string, split it
            lines = source.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

    elif cell['cell_type'] == 'markdown':
        source = cell.get('source', [])

        # Same for markdown
        if isinstance(source, list) and len(source) == 1:
            lines = source[0].split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
        elif isinstance(source, str):
            lines = source.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

    return cell

def fix_notebook():
    """Fix formatting in the notebook."""
    notebook_path = Path("01_seminar_mlp_autograd.ipynb")

    print(f"📖 Loading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"🔧 Fixing cell formatting...")
    print(f"   Total cells: {len(nb['cells'])}")

    # Fix each cell
    for i, cell in enumerate(nb['cells']):
        nb['cells'][i] = fix_cell_formatting(cell)

    # Save
    print(f"💾 Saving fixed notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print("✅ Done! Notebook formatting fixed.")

    # Verify
    print("\n🔍 Verification:")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_check = json.load(f)

    code_cells = [c for c in nb_check['cells'] if c['cell_type'] == 'code']
    print(f"   Code cells: {len(code_cells)}")

    # Check first code cell
    if code_cells:
        first_code = code_cells[0]
        num_lines = len(first_code.get('source', []))
        print(f"   First code cell has {num_lines} lines")
        if num_lines > 1:
            print("   ✓ Code is properly split into lines")
        else:
            print("   ⚠ Code might still be in one line")

if __name__ == "__main__":
    fix_notebook()
