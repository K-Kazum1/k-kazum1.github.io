#!/usr/bin/env python3
"""
Jekyll Blog Structure Exporter

This script walks through a Jekyll blog directory and creates a single Markdown file
that includes the content of all relevant files while ignoring unnecessary ones like
cache files, lock files, etc.

For binary files like images, it simply notes their existence and location.

Usage:
    python export_jekyll_structure.py /path/to/blog/directory [output_file.md]

If output_file is not specified, it defaults to jekyll_structure_export.md in the current directory.
"""

import os
import sys
import datetime
from pathlib import Path

# Files and directories to ignore
IGNORE_DIRS = [
    '_site', '.git', '.jekyll-cache', '.sass-cache', 'node_modules', 
    'vendor', '.bundle', '.github'
]

IGNORE_FILES = [
    'Gemfile', 'Gemfile.lock', '.gitignore', '.DS_Store', 'package.json', 
    'package-lock.json', 'yarn.lock', '.ruby-version'
]

# File extensions to treat as binary (just note existence, don't include content)
BINARY_EXTENSIONS = [
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.pdf', '.zip', 
    '.tar.gz', '.woff', '.woff2', '.ttf', '.eot'
]

def should_ignore(path):
    """Check if a path should be ignored."""
    name = os.path.basename(path)
    
    # Check if it's in the ignore files list
    if name in IGNORE_FILES:
        return True
    
    # Check if it's a hidden file (starts with .)
    if name.startswith('.') and name != '.htaccess':
        return True
    
    # Check if the directory contains any of the ignore directories
    parts = Path(path).parts
    for ignore_dir in IGNORE_DIRS:
        if ignore_dir in parts:
            return True
    
    return False

def is_binary(path):
    """Check if a file is binary based on extension."""
    for ext in BINARY_EXTENSIONS:
        if path.lower().endswith(ext):
            return True
    return False

def write_file_to_md(file_path, output):
    """Process a file and write its content or note its existence to the output."""
    rel_path = os.path.relpath(file_path, start_dir)
    
    if is_binary(file_path):
        output.write(f"## Binary File: {rel_path}\n\n")
        return
    
    # Get file extension to determine the code block language
    _, ext = os.path.splitext(file_path)
    language = ""
    
    # Map extensions to markdown code block language identifiers
    extension_map = {
        '.html': 'html',
        '.htm': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.sass': 'sass',
        '.js': 'javascript',
        '.jsx': 'jsx',
        '.py': 'python',
        '.rb': 'ruby',
        '.md': 'markdown',
        '.yml': 'yaml',
        '.yaml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.sh': 'bash',
        '.bash': 'bash',
        '.txt': '',
    }
    
    if ext in extension_map:
        language = extension_map[ext]
    
    # Write file path and content to the output
    output.write(f"## File: {rel_path}\n\n")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if content.strip():  # Only write code block if there's content
            output.write(f"```{language}\n")
            output.write(content)
            if not content.endswith('\n'):
                output.write('\n')
            output.write("```\n\n")
        else:
            output.write("*This file is empty*\n\n")
    except UnicodeDecodeError:
        # If we can't decode it as UTF-8, it might be binary
        output.write("*This file could not be read as text and might be binary*\n\n")
    except Exception as e:
        output.write(f"*Error reading this file: {str(e)}*\n\n")
    
    output.write("---\n\n")

def process_directory(directory, output):
    """Process a directory and all its files/subdirectories."""
    for item in sorted(os.listdir(directory)):
        path = os.path.join(directory, item)
        
        if should_ignore(path):
            continue
            
        if os.path.isdir(path):
            # Note the directory existence
            rel_path = os.path.relpath(path, start_dir)
            output.write(f"# Directory: {rel_path}/\n\n")
            output.write("---\n\n")
            
            # Process its contents
            process_directory(path, output)
        else:
            write_file_to_md(path, output)

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <jekyll_directory> [output_file]")
        sys.exit(1)
    
    start_dir = sys.argv[1]
    output_file = "jekyll_structure_export.md"
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    if not os.path.isdir(start_dir):
        print(f"Error: {start_dir} is not a directory")
        sys.exit(1)
    
    print(f"Exporting Jekyll blog structure from {start_dir} to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as output:
        # Write header information
        output.write("# Jekyll Blog Structure Export\n\n")
        output.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        output.write(f"Source directory: `{os.path.abspath(start_dir)}`\n\n")
        output.write("This file contains the structure and content of your Jekyll blog for reference.\n\n")
        output.write("---\n\n")
        
        # Process the entire directory
        process_directory(start_dir, output)
    
    print(f"Export complete! The result is saved to {output_file}")
