#!/usr/bin/env python3
"""Check Python syntax and common issues in the codebase."""

import ast
import sys
from pathlib import Path
from typing import List, Tuple


def check_python_file(filepath: Path) -> List[Tuple[int, str]]:
    """Check a Python file for syntax errors and common issues."""
    errors = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check syntax by parsing the AST
        try:
            ast.parse(content)
        except SyntaxError as e:
            errors.append((e.lineno or 0, f"Syntax error: {e.msg}"))
            return errors
        
        # Check for common issues
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            # Check line length (PEP 8 recommends 79, we use 120)
            if len(line) > 120:
                errors.append((i, f"Line too long ({len(line)} > 120 characters)"))
            
            # Check for trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                errors.append((i, "Trailing whitespace"))
            
            # Check for tabs (should use spaces)
            if '\t' in line:
                errors.append((i, "Tab character found (use spaces)"))
        
        # Check imports
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['requests', 'urllib']:
                        errors.append((node.lineno, f"Security: Consider using safer alternatives to {alias.name}"))
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and 'subprocess' in node.module:
                    errors.append((node.lineno, "Security: Be careful with subprocess usage"))
        
    except Exception as e:
        errors.append((0, f"Error reading file: {e}"))
    
    return errors


def check_imports(filepath: Path) -> List[Tuple[int, str]]:
    """Check if imports are properly organized."""
    errors = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Group imports
        stdlib_imports = []
        third_party_imports = []
        local_imports = []
        
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['os', 'sys', 'time', 'logging', 'json', 'pathlib']:
                        stdlib_imports.append((node.lineno, alias.name))
                    elif alias.name.startswith('.'):
                        local_imports.append((node.lineno, alias.name))
                    else:
                        third_party_imports.append((node.lineno, alias.name))
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if module in ['os', 'sys', 'typing', 'pathlib', 'collections']:
                    stdlib_imports.append((node.lineno, module))
                elif module.startswith('.'):
                    local_imports.append((node.lineno, module))
                else:
                    third_party_imports.append((node.lineno, module))
        
        # Check import order
        all_imports = stdlib_imports + third_party_imports + local_imports
        if all_imports and len(all_imports) > 1:
            last_line = 0
            for line, name in all_imports:
                if line < last_line:
                    errors.append((line, "Imports not in order"))
                    break
                last_line = line
    
    except Exception as e:
        errors.append((0, f"Error checking imports: {e}"))
    
    return errors


def main():
    """Check all Python files in the feature extraction directory."""
    root = Path(__file__).parent
    
    # Find all Python files
    python_files = []
    for pattern in ['*.py', '**/*.py']:
        python_files.extend(root.glob(pattern))
    
    # Filter out some paths
    python_files = [
        f for f in python_files 
        if not any(skip in str(f) for skip in ['__pycache__', '.pyc', 'venv', 'env'])
    ]
    
    print(f"Checking {len(python_files)} Python files...\n")
    
    total_errors = 0
    files_with_errors = 0
    
    for filepath in sorted(python_files):
        rel_path = filepath.relative_to(root)
        
        # Check syntax and common issues
        errors = check_python_file(filepath)
        import_errors = check_imports(filepath)
        errors.extend(import_errors)
        
        if errors:
            files_with_errors += 1
            total_errors += len(errors)
            print(f"❌ {rel_path}")
            for line, error in sorted(errors):
                print(f"   Line {line}: {error}")
            print()
        else:
            print(f"✅ {rel_path}")
    
    print(f"\nSummary:")
    print(f"  Files checked: {len(python_files)}")
    print(f"  Files with issues: {files_with_errors}")
    print(f"  Total issues: {total_errors}")
    
    return 1 if total_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())