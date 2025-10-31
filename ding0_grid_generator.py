import pypsa
import networkx as nx
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

'''
Module for extracting low-voltage subnetworks from a PyPSA network based on proximity to transformers.
'''

def extract_lv_subnetwork_to_nearest_transformers(start_buses: pd.DataFrame, path: str) -> pypsa.Network:
    """
    Extracts the low-voltage (LV) subnetwork that connects the given start buses
    to their nearest low-voltage transformers. Only considers lines that connect
    exclusively LV buses.

    Args:
        start_buses (pd.DataFrame): DataFrame containing the starting buses 
            (must include a 'name' column).
        path (str): Path to the directory containing the network data.

    Returns:
        pypsa.Network: The extracted LV subnetwork as a PyPSA network object.
    """


    def build_lv_only_graph(net: pypsa.Network, voltage_threshold=0.4) -> nx.Graph:
        """
        Creates a NetworkX graph containing only low-voltage (LV) lines.

        Args:
            net (pypsa.Network): The full PyPSA network.
            voltage_threshold (float): Maximum voltage (in kV) to consider a bus as LV.

        Returns:
            nx.Graph: Graph with only LV buses and lines connecting them.
        """

        G = nx.Graph()
        buses_vn = net.buses["v_nom"]

        for _, row in net.lines.iterrows():
            b0 = row["bus0"]
            b1 = row["bus1"]
            if buses_vn[b0] <= voltage_threshold and buses_vn[b1] <= voltage_threshold:
                G.add_edge(b0, b1)
        return G


    def shortest_path_to_lv_transformer(G: nx.Graph, start_bus: str, lv_buses: list[str]) -> list[str] | None:
        """
        Finds the shortest path from a given bus to the nearest LV transformer bus.

        Args:
            G (nx.Graph): NetworkX graph of LV buses.
            start_bus (str): The bus from which to start the search.
            lv_buses (list[str]): List of LV transformer bus names.

        Returns:
            list or None: List of bus names forming the shortest path, or None if no path exists.
        """

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


    # Load the network
    grid = pypsa.Network(path)

    # Build LV-only graph (for pathfinding only)
    G_lv = build_lv_only_graph(grid)
    buses_vn = grid.buses["v_nom"].to_dict()

    # LV side of all transformers (always bus1)
    lv_transformer_buses_all = grid.transformers["bus1"].unique()

    # Optional: Filter for LV transformers within the BBOX (if start_buses comes from bbox)
    bbox_bus_names = set(start_buses["name"])
    lv_transformer_buses_in_bbox = [bus for bus in lv_transformer_buses_all if bus in bbox_bus_names]

    if lv_transformer_buses_in_bbox:
        lv_targets = lv_transformer_buses_in_bbox
    else:
        lv_targets = [bus for bus in lv_transformer_buses_all if bus in G_lv.nodes]

    # Collect relevant buses along the paths
    relevant_buses = set()
    for bus in tqdm(start_buses["name"], desc="Searching start buses"):
        # Only use LV buses (v_nom <= 0.4) that are also in the LV graph
        if buses_vn.get(bus, 1.0) <= 0.4 and bus in G_lv:
            path_to_lv = shortest_path_to_lv_transformer(G_lv, bus, lv_targets)
            if path_to_lv:
                relevant_buses.update(path_to_lv)

    # Extend relevant_buses with transformers connected to relevant buses
    for _, trafo in grid.transformers.iterrows():
        if trafo["bus0"] in relevant_buses or trafo["bus1"] in relevant_buses:
            relevant_buses.add(trafo["bus0"])
            relevant_buses.add(trafo["bus1"])


    # Prepare a new network
    new_grid = pypsa.Network()
    new_grid.set_snapshots(grid.snapshots)

    # Components that are associated with buses
    components_with_bus = [
        "Bus", "Generator", "Load", "Shunt", "Store",
        "StorageUnit", "Line", "Link", "Transformer", "Switch"
    ]

    for comp in components_with_bus:
        df = getattr(grid, comp.lower() + "s", None)
        if df is None or df.empty:
            continue
        
        # Keep only elements fully connected to relevant LV buses
        if comp in ["Line", "Link", "Transformer", "Switch"]:
            mask = df["bus0"].isin(relevant_buses) & df["bus1"].isin(relevant_buses)
        else:
            mask = df["bus"].isin(relevant_buses)

        setattr(new_grid, comp.lower() + "s", df[mask].copy())

    # Set buses explicitly
    new_grid.buses = grid.buses.loc[list(relevant_buses)].copy()

    # Optionally copy other global tables
    for name in ["carriers", "global_constraints"]:
        if hasattr(grid, name):
            setattr(new_grid, name, getattr(grid, name))


    # drop lines with s_nom > 0.5
    if not new_grid.lines.empty:
        new_grid.lines = new_grid.lines[new_grid.lines["s_nom"] <= 0.5]

        
    return new_grid




def load_buses_in_bbox(bbox: tuple, base_path: str) -> pd.DataFrame:
    """
    Loads all buses from networks in the specified directory that fall within the given BBOX.
    
    Args:
        bbox (tuple): Bounding box in the format (min_x, min_y, max_x, max_y).
        base_path (str): Directory containing network subfolders. 
        
    Returns:
        pd.DataFrame: DataFrame with all buses located within the BBOX.
        str or None: Path to the last processed topology folder containing buses, or None if no buses found.
    """

    min_x, min_y, max_x, max_y = bbox
    all_buses = []  # List of all filtered bus DataFrames

    for subfolder in tqdm(os.listdir(base_path), desc="Loading buses from networks"):
        subfolder_path = os.path.join(base_path, subfolder)
        topology_path = os.path.join(subfolder_path, "topology")
        buses_file = os.path.join(topology_path, "buses.csv")

        if os.path.isdir(topology_path) and os.path.isfile(buses_file):
            try:
                buses_df = pd.read_csv(buses_file)

                if 'x' in buses_df.columns and 'y' in buses_df.columns:
                    # Apply BBOX filter
                    buses_filtered = buses_df[
                        (buses_df['x'] >= min_x) & (buses_df['x'] <= max_x) &
                        (buses_df['y'] >= min_y) & (buses_df['y'] <= max_y)
                    ]

                    if not buses_filtered.empty:
                        all_buses.append(buses_filtered)
                        path = topology_path # Store the path of the last valid topology

            except Exception as e:
                print(f"Fehler beim Einlesen von {buses_file}: {e}")

    if all_buses:
        return pd.concat(all_buses, ignore_index=True), path
    else:
        print("No buses found within the BBOX.")
        return pd.DataFrame(), None  # Return empty DataFrame and None if nothing found



def load_grid(bbox: tuple, grids_dir: str) -> pypsa.Network:
    """
    Loads the ding0 network and extracts the low-voltage subnetwork
    that connects buses within the given BBOX to the nearest low-voltage transformers.
    
    Args:
        bbox (tuple): Bounding box in the format (min_x, min_y, max_x, max_y).
        grids_dir (str): Directory containing the network subfolders.
        
    Returns:
        pypsa.Network: The extracted low-voltage subnetwork.
    """
    
    filtered_buses, path = load_buses_in_bbox(bbox, grids_dir)
    if not filtered_buses.empty:
        grid = extract_lv_subnetwork_to_nearest_transformers(filtered_buses, path)
    else:
        print("No buses found within the BBOX.")
        grid = pypsa.Network()  # Return empty network if no buses found
    return grid

def save_output_data(grid,
                     buses_df,
                     area,
                     features,
                     scenario: str,
                     steps: list,
                     path='output',
                     ):

    output_dir = os.path.join(path, scenario, f'step_{steps[-1]}')
    os.makedirs(output_dir, exist_ok=True)

    # Save grid data
    grid.export_to_netcdf(os.path.join(output_dir, "grid.nc"))
    
    # Save buses data
    if not buses_df.empty:
        buses_df.to_csv(os.path.join(output_dir, "buses.csv"))
    
    # Save area data
    if not area.empty:
        area.to_file(os.path.join(output_dir, "area.gpkg"), driver="GPKG")
    
    # Save features data
    if not features.empty:
        features.to_file(os.path.join(output_dir, "features.gpkg"), driver="GPKG")

    print(f"Output data saved to {output_dir}")
