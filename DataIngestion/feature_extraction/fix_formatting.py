#!/usr/bin/env python3
"""Fix common Python formatting issues."""

import re
from pathlib import Path


def fix_file(filepath: Path) -> bool:
    """Fix common formatting issues in a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        modified = False
        fixed_lines = []
        
        for line in lines:
            original = line
            
            # Remove trailing whitespace
            line = line.rstrip() + '\n' if line.strip() else line.rstrip()
            
            # Replace tabs with 4 spaces
            if '\t' in line:
                line = line.replace('\t', '    ')
            
            # Fix long lines by adding continuation
            if len(line) > 120 and '#' not in line and '"' not in line and "'" not in line:
                # This is a simple fix - in practice, you'd want more sophisticated line breaking
                if ',' in line and line.count('(') == line.count(')'):
                    # Try to break at a comma
                    parts = line.split(',')
                    if len(parts) > 1:
                        indent = len(line) - len(line.lstrip())
                        new_lines = [parts[0] + ',\n']
                        for part in parts[1:]:
                            new_lines.append(' ' * (indent + 4) + part.strip())
                        line = ''.join(new_lines)
                        if not line.endswith('\n'):
                            line += '\n'
            
            if line != original:
                modified = True
            
            fixed_lines.append(line)
        
        # Ensure file ends with newline
        if fixed_lines and not fixed_lines[-1].endswith('\n'):
            fixed_lines[-1] += '\n'
            modified = True
        
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
        
        return modified
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def organize_imports(filepath: Path) -> bool:
    """Organize imports in a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple import organization - just ensure blank lines between import groups
        lines = content.split('\n')
        
        # Find import section
        import_start = -1
        import_end = -1
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                if import_start == -1:
                    import_start = i
                import_end = i
            elif import_start != -1 and line.strip() and not line.strip().startswith('#'):
                break
        
        if import_start == -1:
            return False
        
        # Extract imports
        imports = []
        for i in range(import_start, import_end + 1):
            if lines[i].strip():
                imports.append(lines[i])
        
        # Sort imports (simple alphabetical)
        imports.sort()
        
        # Reconstruct file
        new_lines = lines[:import_start] + imports + lines[import_end + 1:]
        new_content = '\n'.join(new_lines)
        
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error organizing imports in {filepath}: {e}")
        return False


def main():
    """Fix formatting in all Python files."""
    root = Path(__file__).parent
    
    # Get the newly created files first
    new_files = [
        'features/gpu_preprocessing.py',
        'features/extract_features_gpu_enhanced.py',
        'feature/extract_features_enhanced.py',
        'validate_pipeline.py',
    ]
    
    files_fixed = 0
    
    for file_path in new_files:
        full_path = root / file_path
        if full_path.exists():
            print(f"Fixing {file_path}...")
            if fix_file(full_path):
                files_fixed += 1
                print(f"  ✅ Fixed formatting issues")
            if organize_imports(full_path):
                print(f"  ✅ Organized imports")
    
    print(f"\nFixed {files_fixed} files")


if __name__ == "__main__":
    main()