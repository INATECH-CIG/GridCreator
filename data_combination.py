import numpy as np
import scipy.spatial as spatial
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx


def data_combination(ding0, osm):


    # Alle Zeilen von Generator nach "bus" gruppieren
    grouped = ding0.generators.groupby("bus")

    # Jede Gruppe zu einer einzelnen Zeile "entfalten"
    # mit Spaltennamen
    rows = {}
    for name, group in grouped:
        group = group.drop(columns="bus").reset_index(drop=True)
        flat_row = {}
        for i, row in group.iterrows():
            for col in group.columns:
                flat_row[f"{col}_{i+1}"] = row[col]
        rows[name] = flat_row

    # In DataFrame umwandeln
    generators_flat = pd.DataFrame.from_dict(rows, orient="index")


    ding0.buses.index = ding0.buses.index.astype(str)
    # Mit df_A verbinden
    ding0.buses = ding0.buses.join(generators_flat)


    # Nur end-buses für matching verwenden
    # Graph aufbauen
    G = nx.Graph()
    for _, row in ding0.lines.iterrows():
        G.add_edge(row["bus0"], row["bus1"])

    # End-Busse: nur 1 Nachbar
    end_buses = [bus for bus in G.nodes if G.degree(bus) == 1]

    # Nur diese Buses fürs Matching verwenden
    matching_buses = ding0.buses.loc[ding0.buses.index.isin(end_buses)].copy()



    # osm Daten und Grid Daten kombinieren
    # Laden der Nodes aus osm Daten
    nodes_gdf = osm[osm['element'] == 'node'].copy()
    nodes_gdf["lon"] = nodes_gdf.geometry.x
    nodes_gdf["lat"] = nodes_gdf.geometry.y

    # Buses und Nodes zusammenbringen

    # Bus-Koordinaten
    bus_coords = np.array(list(zip(matching_buses.x, matching_buses.y)))

    # Node-Koordinaten
    node_coords = np.array(list(zip(nodes_gdf.lon, nodes_gdf.lat)))

    # KD-Tree
    tree = spatial.cKDTree(node_coords)


    # für jeden Bus nächsten Node finden

    # Maximaldistanz (ca. 50 m)
    max_dist = 0.0005

    # Alle Zuordnungen innerhalb des Radius sammeln
    matches = []
    for bus_idx, coord in enumerate(bus_coords):
        node_idxs = tree.query_ball_point(coord, max_dist)
        for node_idx in node_idxs:
            dist = np.linalg.norm(coord - node_coords[node_idx])
            matches.append((bus_idx, node_idx, dist))

    # DataFrame aller potenziellen Zuordnungen
    matches_df = pd.DataFrame(matches, columns=["bus_idx", "node_idx", "dist"])

    # Nach Entfernung sortieren (nächstgelegene zuerst)
    matches_df = matches_df.sort_values("dist")

    # Greedy-Zuordnung: jeder Node und jeder Bus nur einmal
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


    # Ergebnis initialisieren
    ding0.buses["matched_node_idx"] = None
    ding0.buses["matched_dist"] = None

    # Index-Zuordnung (von gefiltertem Matching-Set auf Gesamtset)
    match_bus_ids = matching_buses.index.to_list()

    for _, row in final_df.iterrows():
        bus_idx = int(row["bus_idx"])
        node_idx = row["node_idx"]
        dist = row["dist"]
        bus_id = match_bus_ids[bus_idx]

        # Werte eintragen
        ding0.buses.at[bus_id, "matched_node_idx"] = node_idx
        ding0.buses.at[bus_id, "matched_dist"] = dist

    # Neue Spalten aus OSM-Daten übertragen
    new_columns = {}
    for col in nodes_gdf.columns:
        values = []
        for idx in ding0.buses["matched_node_idx"]:
            if pd.notna(idx) and int(idx) < len(nodes_gdf):
                values.append(nodes_gdf.iloc[int(idx)][col])
            else:
                values.append(None)
        new_columns[f"osm_{col}"] = values

    """
    Bei Bedarf behilfs Spalte zur Zuordnung löschen
    """
    #ding0.buses = ding0.buses.drop(columns=["matched_node_idx", "matched_dist"])

    # Spalten hinzufügen
    new_data = pd.DataFrame(new_columns, index=ding0.buses.index)
    ding0.buses = pd.concat([ding0.buses, new_data], axis=1)
    ding0.buses = ding0.buses.copy()

    return ding0

