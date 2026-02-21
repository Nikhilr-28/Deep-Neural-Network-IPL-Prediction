"""
Directory Structure Explorer & Markdown Diagram Generator
==========================================================

Purpose: Generate a complete directory tree diagram for the CSCI_566 project
Output: Markdown file with visual directory structure and full paths
Root: A:\\DL\\Dataset_Nikhl\\CSCI_566\\Nikhil
"""

import os
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIR = Path(r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil")
OUTPUT_FILE = "PROJECT_DIRECTORY_STRUCTURE.md"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_directory_size(path):
    """Calculate total size of directory in MB"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except:
                    pass
    except:
        pass
    return total_size / (1024 * 1024)  # Convert to MB

def get_file_info(filepath):
    """Get file size and type"""
    try:
        size = os.path.getsize(filepath) / 1024  # KB
        ext = filepath.suffix.lower()
        return size, ext
    except:
        return 0, ""

def count_items(path):
    """Count files and folders in directory"""
    try:
        items = list(path.iterdir())
        files = sum(1 for item in items if item.is_file())
        folders = sum(1 for item in items if item.is_dir())
        return files, folders
    except:
        return 0, 0

def generate_tree(directory, prefix="", is_last=True, max_depth=10, current_depth=0):
    """Generate tree structure recursively"""
    
    if current_depth >= max_depth:
        return []
    
    lines = []
    
    try:
        # Get all items and sort (directories first, then files)
        items = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        
        for index, item in enumerate(items):
            is_last_item = (index == len(items) - 1)
            
            # Tree characters
            if is_last_item:
                current_prefix = "└── "
                extension_prefix = "    "
            else:
                current_prefix = "├── "
                extension_prefix = "│   "
            
            if item.is_dir():
                # Directory
                files, folders = count_items(item)
                dir_size = get_directory_size(item)
                
                line = f"{prefix}{current_prefix}📁 **{item.name}/** "
                line += f"`({folders} folders, {files} files, {dir_size:.1f} MB)`"
                lines.append(line)
                
                # Recurse into subdirectory
                sub_lines = generate_tree(
                    item, 
                    prefix + extension_prefix, 
                    is_last_item,
                    max_depth,
                    current_depth + 1
                )
                lines.extend(sub_lines)
                
            else:
                # File
                size, ext = get_file_info(item)
                
                # Choose emoji based on file type
                if ext in ['.py']:
                    emoji = "🐍"
                elif ext in ['.csv']:
                    emoji = "📊"
                elif ext in ['.md', '.txt']:
                    emoji = "📄"
                elif ext in ['.json']:
                    emoji = "📋"
                elif ext in ['.xlsx', '.xls']:
                    emoji = "📈"
                elif ext in ['.zip']:
                    emoji = "📦"
                elif ext in ['.pdf']:
                    emoji = "📕"
                else:
                    emoji = "📄"
                
                if size < 1024:
                    size_str = f"{size:.1f} KB"
                else:
                    size_str = f"{size/1024:.1f} MB"
                
                line = f"{prefix}{current_prefix}{emoji} {item.name} `({size_str})`"
                lines.append(line)
                
    except PermissionError:
        # Define current_prefix for error cases
        current_prefix = "├── " if not is_last else "└── "
        lines.append(f"{prefix}{current_prefix}⚠️ [Permission Denied]")
    except Exception as e:
        # Define current_prefix for error cases
        current_prefix = "├── " if not is_last else "└── "
        lines.append(f"{prefix}{current_prefix}❌ [Error: {str(e)}]")
    
    return lines

def generate_path_list(directory, prefix=""):
    """Generate flat list of all paths"""
    paths = []
    
    try:
        items = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        
        for item in items:
            relative_path = item.relative_to(ROOT_DIR)
            full_path = str(item)
            
            if item.is_dir():
                paths.append(("📁 FOLDER", str(relative_path), full_path))
                # Recurse
                paths.extend(generate_path_list(item, prefix + "  "))
            else:
                size, ext = get_file_info(item)
                if size < 1024:
                    size_str = f"{size:.1f} KB"
                else:
                    size_str = f"{size/1024:.1f} MB"
                paths.append((f"📄 FILE ({size_str})", str(relative_path), full_path))
    except:
        pass
    
    return paths

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_markdown():
    """Generate complete markdown documentation"""
    
    if not ROOT_DIR.exists():
        print(f"❌ ERROR: Directory does not exist: {ROOT_DIR}")
        return
    
    print("=" * 80)
    print("GENERATING PROJECT DIRECTORY STRUCTURE")
    print("=" * 80)
    print(f"\nRoot: {ROOT_DIR}")
    print(f"Output: {OUTPUT_FILE}")
    
    # Generate tree
    print("\n📊 Analyzing directory structure...")
    tree_lines = generate_tree(ROOT_DIR)
    
    # Generate path list
    print("📋 Generating path list...")
    path_list = generate_path_list(ROOT_DIR)
    
    # Calculate statistics
    print("📈 Calculating statistics...")
    total_files = sum(1 for p in path_list if p[0].startswith("📄"))
    total_folders = sum(1 for p in path_list if p[0].startswith("📁"))
    total_size = get_directory_size(ROOT_DIR)
    
    # Count file types
    file_types = {}
    for item_type, rel_path, full_path in path_list:
        if item_type.startswith("📄"):
            ext = Path(full_path).suffix.lower()
            if ext:
                file_types[ext] = file_types.get(ext, 0) + 1
    
    # Build markdown content
    print("📝 Building markdown document...")
    
    md_content = []
    
    # Header
    md_content.append("# 📂 CSCI 566 - El Dorado Project Directory Structure")
    md_content.append("")
    md_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_content.append(f"**Root Path:** `{ROOT_DIR}`")
    md_content.append("")
    md_content.append("---")
    md_content.append("")
    
    # Statistics
    md_content.append("## 📊 Project Statistics")
    md_content.append("")
    md_content.append(f"- **Total Folders:** {total_folders}")
    md_content.append(f"- **Total Files:** {total_files}")
    md_content.append(f"- **Total Size:** {total_size:.2f} MB")
    md_content.append("")
    
    if file_types:
        md_content.append("### File Types Breakdown:")
        md_content.append("")
        md_content.append("| Extension | Count |")
        md_content.append("|-----------|-------|")
        for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
            md_content.append(f"| `{ext}` | {count} |")
        md_content.append("")
    
    md_content.append("---")
    md_content.append("")
    
    # Tree diagram
    md_content.append("## 🌳 Directory Tree")
    md_content.append("")
    md_content.append("```")
    md_content.append(f"📁 {ROOT_DIR.name}/")
    for line in tree_lines:
        md_content.append(line)
    md_content.append("```")
    md_content.append("")
    md_content.append("---")
    md_content.append("")
    
    # Complete path list
    md_content.append("## 📋 Complete File & Folder Paths")
    md_content.append("")
    md_content.append("### Folders:")
    md_content.append("")
    
    for item_type, rel_path, full_path in path_list:
        if item_type.startswith("📁"):
            md_content.append(f"**{rel_path}**")
            md_content.append(f"```")
            md_content.append(f"{full_path}")
            md_content.append(f"```")
            md_content.append("")
    
    md_content.append("### Files:")
    md_content.append("")
    
    current_folder = None
    for item_type, rel_path, full_path in path_list:
        if item_type.startswith("📄"):
            folder = str(Path(rel_path).parent)
            
            if folder != current_folder:
                current_folder = folder
                md_content.append(f"#### 📁 {folder if folder != '.' else 'Root'}:")
                md_content.append("")
            
            md_content.append(f"- **{Path(rel_path).name}** {item_type.replace('📄 FILE ', '')}")
            md_content.append(f"  ```")
            md_content.append(f"  {full_path}")
            md_content.append(f"  ```")
            md_content.append("")
    
    md_content.append("---")
    md_content.append("")
    md_content.append("## 🎯 Quick Reference Paths")
    md_content.append("")
    md_content.append("For easy copy-paste in your scripts:")
    md_content.append("")
    md_content.append("```python")
    md_content.append(f'ROOT_DIR = Path(r"{ROOT_DIR}")')
    md_content.append("")
    
    # Add key folder paths
    key_folders = []
    for item_type, rel_path, full_path in path_list:
        if item_type.startswith("📁"):
            folder_name = Path(rel_path).name
            var_name = folder_name.upper().replace(" ", "_").replace("-", "_")
            key_folders.append((var_name, full_path))
    
    for var_name, full_path in key_folders[:20]:  # Show first 20
        md_content.append(f'{var_name}_DIR = ROOT_DIR / r"{Path(full_path).relative_to(ROOT_DIR)}"')
    
    md_content.append("```")
    md_content.append("")
    
    md_content.append("---")
    md_content.append("")
    md_content.append("*Generated by Directory Structure Explorer*")
    
    # Write to file
    print(f"\n💾 Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))
    
    print(f"\n✅ SUCCESS!")
    print(f"\n📄 File created: {os.path.abspath(OUTPUT_FILE)}")
    print(f"   Size: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")
    print(f"   Lines: {len(md_content)}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Folders analyzed: {total_folders}")
    print(f"  Files analyzed: {total_files}")
    print(f"  Total size: {total_size:.2f} MB")
    print(f"  Output file: {OUTPUT_FILE}")
    print("\n✅ Directory structure documentation complete!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    generate_markdown()