import pypsa
import networkx as nx
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def extract_lv_subnetwork_to_nearest_transformers(start_buses, grids_dir):
    def shortest_path_to_lv_transformer(net, start_bus, lv_buses):
        G = net.graph()
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
    grid_id = str(start_buses["mv_grid_id"].iloc[0])
    network_path = os.path.join(grids_dir, grid_id, "topology")
    net = pypsa.Network(network_path)

    # LV-Seite aller Transformatoren (immer bus1)
    lv_transformer_buses_all = net.transformers["bus1"].unique()

    # Optional: Filter auf LV-Trafos innerhalb der BBOX (wenn start_buses aus bbox kommt)
    bbox_bus_names = set(start_buses["name"])
    lv_transformer_buses_in_bbox = [bus for bus in lv_transformer_buses_all if bus in bbox_bus_names]

    if lv_transformer_buses_in_bbox:
        #print(f"⚡ Verwende {len(lv_transformer_buses_in_bbox)} LV-Trafos aus der BBOX.")
        lv_targets = lv_transformer_buses_in_bbox
    else:
        #print("⚠️ Keine LV-Trafos in der BBOX gefunden – verwende alle LV-Trafos.")
        lv_targets = lv_transformer_buses_all

    # Relevante Busse entlang der Pfade sammeln
    relevant_buses = set()
    for bus in start_buses["name"]:
        pfad = shortest_path_to_lv_transformer(net, bus, lv_targets)
        if pfad:
            relevant_buses.update(pfad)

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




def load_buses_in_bbox(bbox, base_path):
    min_x, min_y, max_x, max_y = bbox
    all_buses = []  # Liste aller gefilterten Bus-DataFrames

    for subfolder in os.listdir(base_path):
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

            except Exception as e:
                print(f"Fehler beim Einlesen von {buses_file}: {e}")

    if all_buses:
        return pd.concat(all_buses, ignore_index=True)
    else:
        return pd.DataFrame()  # Leerer DataFrame, wenn nichts gefunden



def load_grid(bbox, grids_dir):
    filtered_buses = load_buses_in_bbox(bbox, grids_dir)
    grid = extract_lv_subnetwork_to_nearest_transformers(filtered_buses, grids_dir)
    return grid