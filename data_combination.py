import numpy as np
import scipy.spatial as spatial
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx
from dicts import generator_agg_dict

# import for type hints
import pypsa

'''
Module for combining grid data with OSM data based on spatial proximity.
'''

def generator_duplikate_zusammenfassen(grid: pypsa.Network) -> pypsa.Network:
    """
    Consolidates generators at the same bus with the same type (carrier).
    Sums p_nom and replaces multiple entries with a single one.

    Args:
        grid (pypsa.Network): The power system network containing generators.

    Returns:
        pypsa.Network: The updated network with consolidated generators.
    """
    gens = grid.generators

    if 'carrier' in gens.columns and 'type' in gens.columns:
        # Group by bus, type AND carrier
        new_generators = gens.groupby(['bus', 'type', 'carrier']).generators_agg_dict()
    elif 'carrier' not in gens.columns and 'type' in gens.columns:
        # Group by bus AND type
        new_generators = gens.groupby(['bus', 'type']).generators_agg_dict()
    elif 'type' not in gens.columns and 'carrier' in gens.columns:
        # Group by bus AND carrier
        new_generators = gens.groupby(['bus', 'carrier']).generators_agg_dict()
    else:
        # raise warning and return original grid
        new_generators = pd.DataFrame()
        print("Warning: Neither 'type' nor 'carrier' columns found in generators. No consolidation performed.")

    if not new_generators.empty:
        # Delete old generators
        grid.generators.drop(index=grid.generators.index, inplace=True)

        # Add new generators
        for gen in new_generators:
            grid.add("Generator", **gen)

    return grid


def data_combination(grid: pypsa.Network, osm: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Combines grid data from a PyPSA network with OSM data based on spatial proximity.
    Matches end buses in the grid to nearest OSM nodes within a specified distance.
    
    Args:
        ding0 (pypsa.Network): The power system network containing buses and generators.
        osm (gpd.GeoDataFrame): GeoDataFrame containing OSM data with geometry.
        
    Returns:
        pd.DataFrame: DataFrame with combined information from ding0 and matched OSM nodes.
    """

    # Consolidate duplicate generators at the same bus
    grid = generator_duplikate_zusammenfassen(grid)

    # Flatten generators per bus
    grouped = grid.generators.groupby("bus")
    rows = {}
    for name, group in grouped:
        group = group.drop(columns="bus").reset_index(drop=True)
        flat_row = {}
        for i, row in group.iterrows():
            for col in group.columns:
                flat_row[f"{col}_{i+1}"] = row[col]
        rows[name] = flat_row
    generators_flat = pd.DataFrame.from_dict(rows, orient="index")

    # Join bus data with flattened generators
    grid.buses.index = grid.buses.index.astype(str)
    buses_df = grid.buses.join(generators_flat)

    # Remove transformer buses from buses_df
    trafos = grid.transformers.copy()
    buses_df.drop(index=trafos['bus0'], inplace=True)
    buses_df.drop(index=trafos['bus1'], inplace=True)


    # Identify end buses (degree=1) for matching
    G = nx.Graph()
    for _, row in grid.lines.iterrows():
        G.add_edge(row["bus0"], row["bus1"])
    end_buses = [bus for bus in G.nodes if G.degree(bus) == 1]
    matching_buses = buses_df.loc[buses_df.index.isin(end_buses)].copy()

    # Prepare OSM node coordinates
    nodes_gdf = osm[osm['element'] == 'node'].copy()
    nodes_gdf["lon"] = nodes_gdf.geometry.x
    nodes_gdf["lat"] = nodes_gdf.geometry.y

    # KD-Tree for nearest-neighbor matching
    bus_coords = np.array(list(zip(matching_buses.x, matching_buses.y)))
    node_coords = np.array(list(zip(nodes_gdf.lon, nodes_gdf.lat)))
    tree = spatial.cKDTree(node_coords)
    max_dist = 0.0005

    # Collect all potential matches within max distance
    matches = []
    for bus_idx, coord in enumerate(bus_coords):
        node_idxs = tree.query_ball_point(coord, max_dist)
        for node_idx in node_idxs:
            dist = np.linalg.norm(coord - node_coords[node_idx])
            matches.append((bus_idx, node_idx, dist))
    matches_df = pd.DataFrame(matches, columns=["bus_idx", "node_idx", "dist"]).sort_values("dist")

    # Greedy assignment: one bus -> one node
    used_nodes = set()
    used_buses = set()
    final_matches = []

    for _, row in matches_df.iterrows():
        b, n = int(row["bus_idx"]), int(row["node_idx"])
        if b not in used_buses and n not in used_nodes:
            final_matches.append((b, n, row["dist"]))
            used_buses.add(b)
            used_nodes.add(n)
    final_df = pd.DataFrame(final_matches, columns=["bus_idx", "node_idx", "dist"])

    # Add matched info to end_buses_df
    end_buses_df = matching_buses.copy()
    end_buses_df["matched_node_idx"] = None
    end_buses_df["matched_dist"] = None
    match_bus_ids = matching_buses.index.to_list()
    for _, row in final_df.iterrows():
        bus_idx = int(row["bus_idx"])
        node_idx = row["node_idx"]
        dist = row["dist"]
        bus_id = match_bus_ids[bus_idx]
        end_buses_df.at[bus_id, "matched_node_idx"] = node_idx
        end_buses_df.at[bus_id, "matched_dist"] = dist

    # Add OSM node attributes to buses
    new_columns = {}
    for col in nodes_gdf.columns:
        values = []
        for idx in end_buses_df["matched_node_idx"]:
            if pd.notna(idx) and int(idx) < len(nodes_gdf):
                values.append(nodes_gdf.iloc[int(idx)][col])
            else:
                values.append(None)
        new_columns[f"osm_{col}"] = values

    # Append new columns to end_buses_df
    new_data = pd.DataFrame(new_columns, index=end_buses_df.index)
    end_buses_df = pd.concat([end_buses_df, new_data], axis=1)

    return end_buses_df
