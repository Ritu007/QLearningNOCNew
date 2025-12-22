import ast
from pathlib import Path
from typing import Dict, List


def extract_values_from_dict_file(file_path: Path) -> List:
    """
    Reads the file at file_path, expects it to contain a Python dict literal,
    e.g.:
        {(0, 0): 0, (0, 1): 2, (0, 2): 2}
    It will parse the content and return list(data.values()).
    If parsing fails or the top-level is not a dict, returns an empty list.
    """
    try:
        text = file_path.read_text(encoding='utf-8').strip()
    except Exception as e:
        print(f"Warning: could not read {file_path}: {e}")
        return []

    # Try to locate the dict literal:
    # If the file has exactly the dict literal and nothing else, ast.literal_eval(text) works directly.
    # If there may be extra whitespace or trailing newline, .strip() handles that.
    # If there is additional text before/after, we could try to extract the substring
    # between the first '{' and the last '}', but that depends on file format.
    # Here we assume the file content is exactly (or at least starting with) a dict literal.
    try:
        data = ast.literal_eval(text)
    except Exception as e:
        # Fallback: try to heuristically extract between first '{' and last '}'
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            substring = text[start:end + 1]
            try:
                data = ast.literal_eval(substring)
            except Exception as e2:
                print(f"Warning: parsing failed for {file_path}: {e2}")
                return []
        else:
            print(f"Warning: no valid dict literal found in {file_path}")
            return []

    if not isinstance(data, dict):
        print(f"Warning: content in {file_path} is not a dict after parsing.")
        return []

    # Extract values:
    values = list(data.values())
    return values


def find_and_extract(base_dir: Path) -> Dict[Path, List]:
    """
    Walks base_dir recursively, finds all files with names ending in '-index.txt',
    parses each as a dict and extracts values list.
    Returns a dict mapping each file Path to its list of values.
    """
    result: Dict[Path, List] = {}
    # Use rglob to find recursively. Pattern "*-index.txt" matches any filename ending with "-index.txt".
    for file_path in base_dir.rglob("*-index.txt"):
        if file_path.is_file():
            vals = extract_values_from_dict_file(file_path)
            result[file_path] = vals
    return result



# Adjust this to wherever your folder structure root is.
base_folder = Path("Q_Tables")  # or Path("/path/to/Q_Tables") if not in CWD
if not base_folder.exists():
    print(f"Error: base folder {base_folder} does not exist.")
else:
    extraction = find_and_extract(base_folder)
    # Example: print summary
    for fp, vals in extraction.items():
        print(f"File: {fp} -> extracted {len(vals)} values: {vals!r}")

    # If you need a single flattened list of all values across all files:
    all_values = []
    for vals in extraction.values():
        all_values.extend(vals)
    print(f"\nTotal files processed: {len(extraction)}")
    print(f"Total values collected across all files: {len(all_values)}")
    # Optionally: do something with all_values, e.g., save to a file, further processing, etc.
