#!/usr/bin/env python3
"""Final formatting fixes for Python files."""

import ast
from pathlib import Path


def organize_imports_properly(filepath: Path) -> bool:
    """Properly organize imports following PEP 8."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        # Separate different parts of the file
        docstring = None
        imports = {'stdlib': [], 'third_party': [], 'local': []}
        other_code = []
        
        # Track line numbers
        import_end_line = 0
        
        for i, node in enumerate(tree.body):
            if i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
                # Module docstring
                docstring = node
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                import_end_line = node.end_lineno or node.lineno
                # Categorize import
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name
                        if module in ['os', 'sys', 'time', 'json', 'gc', 'logging', 'subprocess', 'ast', 're']:
                            imports['stdlib'].append(node)
                        elif module.startswith('.'):
                            imports['local'].append(node)
                        else:
                            imports['third_party'].append(node)
                else:  # ImportFrom
                    module = node.module or ''
                    if module in ['pathlib', 'typing', 'collections', 'functools', 'itertools']:
                        imports['stdlib'].append(node)
                    elif module.startswith('.'):
                        imports['local'].append(node)
                    else:
                        imports['third_party'].append(node)
            else:
                other_code.append(node)
        
        # Get the original lines
        lines = content.split('\n')
        
        # Reconstruct the file
        new_lines = []
        
        # Add docstring if present
        if docstring:
            end_line = docstring.end_lineno or docstring.lineno
            new_lines.extend(lines[:end_line])
            new_lines.append('')  # Blank line after docstring
        
        # Function to get import string
        def get_import_lines(node, original_lines):
            start = node.lineno - 1
            end = (node.end_lineno or node.lineno)
            return original_lines[start:end]
        
        # Add imports in order
        if imports['stdlib']:
            for node in sorted(imports['stdlib'], key=lambda n: ast.unparse(n)):
                new_lines.extend(get_import_lines(node, lines))
        
        if imports['stdlib'] and imports['third_party']:
            new_lines.append('')  # Blank line between groups
        
        if imports['third_party']:
            for node in sorted(imports['third_party'], key=lambda n: ast.unparse(n)):
                new_lines.extend(get_import_lines(node, lines))
        
        if (imports['stdlib'] or imports['third_party']) and imports['local']:
            new_lines.append('')  # Blank line between groups
        
        if imports['local']:
            for node in sorted(imports['local'], key=lambda n: ast.unparse(n)):
                new_lines.extend(get_import_lines(node, lines))
        
        if any(imports.values()):
            new_lines.append('')  # Blank line after all imports
        
        # Add the rest of the code
        if import_end_line > 0 and import_end_line < len(lines):
            # Skip empty lines right after imports
            rest_start = import_end_line
            while rest_start < len(lines) and not lines[rest_start].strip():
                rest_start += 1
            new_lines.extend(lines[rest_start:])
        
        # Join and clean up
        new_content = '\n'.join(new_lines)
        
        # Ensure file ends with newline
        if not new_content.endswith('\n'):
            new_content += '\n'
        
        # Only write if changed
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error organizing imports in {filepath}: {e}")
        return False


def remove_trailing_whitespace(filepath: Path) -> bool:
    """Remove trailing whitespace from all lines."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        modified = False
        clean_lines = []
        
        for line in lines:
            clean_line = line.rstrip()
            if line != clean_line + '\n' and line != clean_line:
                modified = True
            clean_lines.append(clean_line)
        
        # Join with newlines and ensure final newline
        content = '\n'.join(clean_lines)
        if content and not content.endswith('\n'):
            content += '\n'
        
        if modified or not content.endswith('\n'):
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error cleaning whitespace in {filepath}: {e}")
        return False


def fix_file_completely(filepath: Path) -> bool:
    """Apply all fixes to a file."""
    fixed = False
    
    # First, remove trailing whitespace
    if remove_trailing_whitespace(filepath):
        print(f"  ✓ Removed trailing whitespace from {filepath}")
        fixed = True
    
    # Then organize imports
    if organize_imports_properly(filepath):
        print(f"  ✓ Organized imports in {filepath}")
        fixed = True
    
    return fixed


def main():
    """Fix all Python files with issues."""
    files_to_fix = [
        'features/gpu_preprocessing.py',
        'features/extract_features_gpu_enhanced.py',
        'feature/extract_features_enhanced.py',
        'validate_pipeline.py',
    ]
    
    print("Applying final formatting fixes...\n")
    
    for file_path in files_to_fix:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"Fixing {file_path}...")
            if fix_file_completely(full_path):
                print(f"  ✅ Fixed {file_path}")
            else:
                print(f"  ℹ️  No changes needed for {file_path}")
        else:
            print(f"  ❌ File not found: {file_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()