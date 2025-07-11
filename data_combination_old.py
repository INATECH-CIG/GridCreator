import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point



def data_combination(ding0, osm):
    
    # Laden der Nodes aus osm Daten
    nodes_gdf = osm[osm['element'] == 'node'].copy()
    nodes_gdf["lon"] = nodes_gdf.geometry.x
    nodes_gdf["lat"] = nodes_gdf.geometry.y

    # Buses und Nodes zusammenbringen

    # Bus-Koordinaten
    bus_coords = np.array(list(zip(ding0.x, ding0.y)))


    # Node-Koordinaten
    node_coords = np.array(list(zip(nodes_gdf.lon, nodes_gdf.lat)))

    # KD-Tree
    tree = cKDTree(node_coords)


    # für jeden Bus nächsten Node finden
    distances, indices = tree.query(bus_coords, distance_upper_bound=0.00005)  # ~5m

    # Ergebnis als DataFrame
    bus_to_node_idx = pd.Series(indices, index=ding0.index)


    for col in nodes_gdf.columns:
        if col not in ["geometry", "lon", "lat"]:
            values = []
            for idx in bus_to_node_idx:
                if np.isfinite(idx) and idx < len(nodes_gdf):
                    values.append(nodes_gdf.iloc[int(idx)][col])
                else:
                    values.append(None)
            ding0[f"osm_{col}"] = values

    return ding0

