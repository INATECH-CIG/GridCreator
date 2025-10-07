import zipfile
from pathlib import Path

def daten_speichern():
    
    DATA_DIR = Path.home() / "my_local_data_GridCreator"  # z. B. C:/Users/.../my_local_data
    ZIP_FILE = Path(__file__).parent / "input.zip"

    # Sicherstellen, dass Zielordner existiert
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    # Daten entpacken, falls noch nicht vorhanden
    if not any(DATA_DIR.iterdir()):  # Verzeichnis ist leer
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print(f"Daten entpackt nach {DATA_DIR}")
    else:
        print(f"Daten schon vorhanden in {DATA_DIR}")
    
    return DATA_DIR