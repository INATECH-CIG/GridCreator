import os
import pandas as pd
from tqdm import tqdm
import pypsa


def create_grid(bbox: list[float], grids_dir: str) -> pypsa.Network:
    """
    Erstellt ein PyPSA-Netzwerk basierend auf den Bus-Koordinaten innerhalb eines gegebenen Rechtecks (bbox).
    
    Args:
        bbox (list[float]): Liste mit den Koordinaten [xmin, ymin, xmax, ymax].
        grids_dir (str): Verzeichnis, das die Unterordner mit den Netzwerken enthält.
        
    Returns:
        pypsa.Network: Das gefilterte PyPSA-Netzwerk.
    """


    xmin, ymin, xmax, ymax = bbox  # aus bbox auspacken
    # Ergebnisliste
    found_buses = []
    folders = [f for f in os.listdir(grids_dir) if os.path.isdir(os.path.join(grids_dir, f))]

    # Durchsuche alle Ordner im grids_dir nach Bussen
    for folder in tqdm(folders, desc="Durchsuche Netzwerke"):
        folder_path = os.path.join(grids_dir, folder)
        bus_file = os.path.join(folder_path, "topology", "buses.csv")
        
        # Prüfe, ob die bus.csv Datei existiert
        if os.path.exists(bus_file):
            try:
                buses = pd.read_csv(bus_file, index_col=0)

                # Abgleichen der Koordinaten
                if 'x' in buses.columns and 'y' in buses.columns:
                    mask = (
                        (buses['x'] >= xmin) & (buses['x'] <= xmax) &
                        (buses['y'] >= ymin) & (buses['y'] <= ymax)
                    )
                    
                    selected_buses = buses[mask]
                    
                    if not selected_buses.empty:
                        for bus_name, row in selected_buses.iterrows():
                            found_buses.append({
                                "network_folder": folder,
                                "bus_name": bus_name,
                                "bus_x": row['x'],
                                "bus_y": row['y']
                            })
            except Exception as e:
                print(f"Fehler beim Laden von {bus_file}: {e}")

        

    if not found_buses:
        print("Keine Busse im Quadrat gefunden, Skript beendet.")
    else:
        # Wir nehmen nur den ersten gefundenen Bus (und damit sein Netzwerk)
        target_network_folder = found_buses[0]['network_folder']
        target_buses = [b['bus_name'] for b in found_buses if b['network_folder'] == target_network_folder]

        #print(f"\nGefundene Busse im Quadrat im Netzwerk-Ordner: {target_network_folder}")
        #print(f"Busse: {target_buses}")

    # Lade das Netzwerk des ersten gefundenen Ordners
    network_path = os.path.join(grids_dir, target_network_folder, "topology")
    net = pypsa.Network(network_path)

    # Entferne alle Busse, die nicht im Quadrat liegen
    buses_to_keep = set(target_buses)
    buses_to_remove = set(net.buses.index) - buses_to_keep
    print(f"Lösche {len(buses_to_remove)} Busse außerhalb des Quadrats.")
    net.remove("Bus", list(buses_to_remove))

    remaining_buses = set(net.buses.index)
    # analog für Generators:
    net.generators = net.generators[
    net.generators['bus'].isin(remaining_buses)
    ]


    # Rückgabe des Netzwerks
    return net
