import zipfile
from pathlib import Path

'''
Module for managing input data by extracting it from a ZIP file if necessary.
'''

def save_data():
    """
    Ensures the input data is available locally by extracting it from a ZIP file
    if necessary. Returns the path to the local data directory.
    """

    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR / "my_local_data_GridCreator"
    ZIP_FILE = SCRIPT_DIR / "input.zip"

    # Ensure the target folder exists
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    # Extract data if not already present
    if not any(DATA_DIR.iterdir()):  # Directory is empty
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print(f"Data extracted to {DATA_DIR}")
    else:
        print(f"Data already present in {DATA_DIR}")
    
    return DATA_DIR