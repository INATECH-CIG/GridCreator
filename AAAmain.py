import main_function as mf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import osmnx as ox
import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd
import daten_speichern as ds

#%%

def main(top: float, bottom: float, left: float, right: float, steps=5) -> tuple:
    """
    Hauptprogramm zur Erstellung und Simulation eines Stromnetzes mit PyPSA.
    Das Programm umfasst die folgenden Schritte:
    1. Definition eines geografischen Bereichs (bbox) und Erstellung eines Stromnetzes
    2. Laden und Zuordnen von Geodaten (OSM, Bundesländer, Zensus)
    3. Zuordnung von Technologien (Solar, E-Auto, Wärmepumpe) zu den Netzpunkten
    4. Hinzufügen von Lastzeitreihen basierend auf den zugeordneten Technologien
    5. Vorbereitung des Netzes für die Optimierung
    6. Durchführung der Netzoptimierung mit PyPSA
    7. Visualisierung der Ergebnisse
    """
    # STEP 0
    pfad = ds.daten_speichern()

    # STEP 1
    # Speichern vom Grid
    output_file_grid = f"{pfad}/dist_grid.nc"
    grids_dir = f"{pfad}/input/grids"
    #Grid creation
    bbox=[left, bottom, right, top]
    grid_1, bbox_1 = mf.ding0_grid(bbox, grids_dir, output_file_grid)
    # Erstellen für Return
    buses_df = pd.DataFrame()

    # STEP 2
    if steps >1:
        # Daten laden
        # OSM Daten
        buffer = 0.0002  # entspricht ungefähr 20 m
        buses_df, area, features = mf.osm_data(grid_1, bbox_1, buffer)
        # Bundesland-Daten
        gpd_bundesland = gpd.read_file(f"{pfad}/input/georef-germany-postleitzahl.geojson")
        ordner = f"{pfad}/input/zensus_daten"
        buses_df = mf.daten_zuordnung(buses_df, gpd_bundesland, ordner)
    # STEP 3
    if steps >2:
        # Technik Zuordnen
        # Technik definieren
        Technik = ['solar', 'E_car', 'HP']
        # Technik zuordnen
        file_Faktoren = f"{pfad}/input/Faktoren.csv"
        file_solar = f"{pfad}/input/Bev_data_solar.csv"
        file_ecar = f"{pfad}/input/Bev_data_ecar.csv"
        file_hp = f"{pfad}/input/Bev_data_hp.csv"
        buses_df, factor_bbox = mf.technik_zuordnen(buses_df, file_Faktoren, file_solar, file_ecar, file_hp, Technik, pfad)
        buses_df = mf.wohnungen_zuordnen(buses_df)
        # Technik in buses_df einfügen
        buses_df = mf.technik_fill(buses_df, Technik, factor_bbox, pfad)
    # STEP 4
    if steps >3:
        # Zeitreihen hinzufügen
        grid_1 = mf.loads_zuordnen(grid_1, buses_df, bbox_1, pfad)


    # STEP 5
    if steps >4:
        # Grid für pysa.optimze() vorbereiten
        grid_1 = mf.pypsa_vorbereiten(grid_1)




    return grid_1, buses_df, bbox_1

#%%

# top =  54.48594882134629 # # Upper latitude
# bottom = 54.47265521486088 # Lower latitude
# left =  11.044533685584453    # Right longitude
# right =  11.084893505520695  # Left longitude
'''
Waldmünchen
'''
# top =  49.374518600877046 # # Upper latitude
# bottom = 49.36971937206515 # Lower latitude
# left =  12.697361279468392   # Right longitude
# right =  12.708888681047798  # Left longitude
'''
2 Nodes
'''
top =  49.3727 # # Upper latitude
bottom = 49.372485 # Lower latitude
left =  12.703688   # Right longitude
right =  12.704 # Left longitude

grid, buses, bbox = main(top, bottom, left, right, steps=5)
# %%
# STEP 6
# .optimize()


# # Fix Capacity
# grid_1.optimize.fix_optimal_capacities()

# # Set snapshots für Optimierung
# start_time = pd.Timestamp("2023-01-01 00:00:00")
# end_time = pd.Timestamp("2023-01-07 23:00:00")
# snapshots = pd.date_range(start=start_time, end=end_time, freq='h')
# grid_1.set_snapshots(snapshots)

#
# grid_1.optimize()