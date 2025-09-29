import pypsa
import networkx as nx
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def extract_lv_subnetwork_to_nearest_transformers(start_buses: pd.DataFrame, path: str) -> pypsa.Network:
    """
    Extrahiert das Niederspannungs-Teilnetz, das die angegebenen Start-Busse
    mit den nächstgelegenen Niederspannungs-Transformatoren verbindet.
    Berücksichtigt dabei nur Leitungen, die ausschließlich Niederspannungs-Busse verbinden.
    
    Args:
        start_buses (pd.DataFrame): DataFrame mit den Start-Bussen (muss eine 'name'-Spalte enthalten).
        grids_dir (str): Verzeichnis, in dem die Netzdaten gespeichert sind.
        
    Returns:
        pypsa.Network: Das extrahierte Niederspannungs-Teilnetz.
    """


    def build_lv_only_graph(net, voltage_threshold=0.4):
        """Erzeugt einen NetworkX-Graph mit nur LV-Leitungen."""
        G = nx.Graph()
        buses_vn = net.buses["v_nom"]

        for _, row in net.lines.iterrows():
            b0 = row["bus0"]
            b1 = row["bus1"]
            if buses_vn[b0] <= voltage_threshold and buses_vn[b1] <= voltage_threshold:
                G.add_edge(b0, b1)
        return G
    
    def shortest_path_to_lv_transformer(G, start_bus, lv_buses):
        shortest_path = None
        shortest_length = float('inf')

        for lv_bus in lv_buses:
            try:
                path = nx.shortest_path(G, source=start_bus, target=lv_bus)
                if len(path) < shortest_length:
                    shortest_length = len(path)
                    shortest_path = path
            except nx.NetworkXNoPath:
                continue
        return shortest_path

    # Netzwerk laden
    network_path = path
    net = pypsa.Network(network_path)

    # LV-only Graph aufbauen (nur für Pfadsuche!)
    G_lv = build_lv_only_graph(net)
    buses_vn = net.buses["v_nom"].to_dict()

    # LV-Seite aller Transformatoren (immer bus1)
    lv_transformer_buses_all = net.transformers["bus1"].unique()

    # Optional: Filter auf LV-Trafos innerhalb der BBOX (wenn start_buses aus bbox kommt)
    bbox_bus_names = set(start_buses["name"])
    lv_transformer_buses_in_bbox = [bus for bus in lv_transformer_buses_all if bus in bbox_bus_names]

    if lv_transformer_buses_in_bbox:
        lv_targets = lv_transformer_buses_in_bbox
    else:
        lv_targets = [bus for bus in lv_transformer_buses_all if bus in G_lv.nodes]

    # Relevante Busse entlang der Pfade sammeln
    relevant_buses = set()
    for bus in tqdm(start_buses["name"], desc="Durchsuche Start-Busse"):
        # Nur LV-Busse (v_nom <= 0.4) benutzen, die auch im LV-Graph sind
        if buses_vn.get(bus, 1.0) <= 0.4 and bus in G_lv:
            pfad = shortest_path_to_lv_transformer(G_lv, bus, lv_targets)
            if pfad:
                relevant_buses.update(pfad)

    # Erweitere relevant_buses um Transformatoren, die an relevanten Bussen hängen
    for _, trafo in net.transformers.iterrows():
        if trafo["bus0"] in relevant_buses or trafo["bus1"] in relevant_buses:
            relevant_buses.add(trafo["bus0"])
            relevant_buses.add(trafo["bus1"])


    # Neues Netz vorbereiten
    new_net = pypsa.Network()
    new_net.set_snapshots(net.snapshots)

    components_with_bus = [
        "Bus", "Generator", "Load", "Shunt", "Store",
        "StorageUnit", "Line", "Link", "Transformer", "Switch"
    ]

    for comp in components_with_bus:
        df = getattr(net, comp.lower() + "s", None)
        if df is None or df.empty:
            continue

        if comp in ["Line", "Link", "Transformer", "Switch"]:
            mask = df["bus0"].isin(relevant_buses) & df["bus1"].isin(relevant_buses)
        else:
            mask = df["bus"].isin(relevant_buses)

        setattr(new_net, comp.lower() + "s", df[mask].copy())

    # Buses setzen
    new_net.buses = net.buses.loc[list(relevant_buses)].copy()

    # Optional weitere Tabellen übernehmen
    for name in ["carriers", "global_constraints"]:
        if hasattr(net, name):
            setattr(new_net, name, getattr(net, name))

    return new_net




def load_buses_in_bbox(bbox: tuple, base_path: str) -> pd.DataFrame:
    """
    Lädt alle Busse aus den Netzen im angegebenen Verzeichnis, die innerhalb der
    gegebenen BBOX liegen.
    
    Args:
        bbox (tuple): BBOX im Format (min_x, min_y, max_x, max_y).
        base_path (str): Verzeichnis mit den Netz-Unterordnern. 
        
    Returns:
        pd.DataFrame: DataFrame mit allen Bussen innerhalb der BBOX.
    """

    min_x, min_y, max_x, max_y = bbox
    all_buses = []  # Liste aller gefilterten Bus-DataFrames

    for subfolder in tqdm(os.listdir(base_path), desc="Lade Busse aus Netzen"):
        subfolder_path = os.path.join(base_path, subfolder)
        topology_path = os.path.join(subfolder_path, "topology")
        buses_file = os.path.join(topology_path, "buses.csv")

        if os.path.isdir(topology_path) and os.path.isfile(buses_file):
            try:
                buses_df = pd.read_csv(buses_file)

                if 'x' in buses_df.columns and 'y' in buses_df.columns:
                    # BBOX-Filter anwenden
                    buses_filtered = buses_df[
                        (buses_df['x'] >= min_x) & (buses_df['x'] <= max_x) &
                        (buses_df['y'] >= min_y) & (buses_df['y'] <= max_y)
                    ]

                    if not buses_filtered.empty:
                        all_buses.append(buses_filtered)
                        path = topology_path

            except Exception as e:
                print(f"Fehler beim Einlesen von {buses_file}: {e}")

    if all_buses:
        return pd.concat(all_buses, ignore_index=True), path
    else:
        print("Keine Busse innerhalb der BBOX gefunden.")
        return pd.DataFrame(), None  # Leerer DataFrame und None, wenn nichts gefunden



def load_grid(bbox: tuple, grids_dir: str) -> pypsa.Network:
    """
    Lädt das Ding0-Netz und extrahiert das Niederspannungs-Teilnetz,
    das die Busse innerhalb der gegebenen BBOX mit den nächstgelegenen
    Niederspannungs-Transformatoren verbindet.
    
    Args:
        bbox (tuple): BBOX im Format (min_x, min_y, max_x, max_y).
        grids_dir (str): Verzeichnis mit den Netz-Unterordnern.
        
    Returns:
        pypsa.Network: Das extrahierte Niederspannungs-Teilnetz.
    """
    
    filtered_buses, path = load_buses_in_bbox(bbox, grids_dir)
    if not filtered_buses.empty:
        grid = extract_lv_subnetwork_to_nearest_transformers(filtered_buses, path)
    else:
        print("Keine Busse innerhalb der BBOX gefunden.")
        grid = pypsa.Network()  # Leeres Netz zurückgeben, wenn keine Busse gefunden wurden
    return grid