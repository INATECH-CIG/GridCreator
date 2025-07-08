#%% Import section


import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import openpyxl
import functions as func
import ding0_grid_transformers as ding0
import pypsa
from scipy.spatial import cKDTree
import numpy as np
import data_combination as dc
import cartopy.crs as ccrs

#%% GPS Data eingeben
top =  54.48594882134629 # # Upper latitude
bottom = 54.47265521486088 # Lower latitude
left =  11.044533685584453    # Right longitude
right =  11.084893505520695  # Left longitude#



bbox = [left, bottom, right, top]


#%% Grid creation


''''
Aufrufen der Funktion, um aus vorgefertigten Grids, den gesuchten Bereich zu Filtern
'''

# Speicher Ort
grids_dir = "grids" 

# Netz extrahieren
grid = ding0.load_grid(bbox, grids_dir)

# # Netz plotten (ohne Generator-Farben)
# grid.plot(bus_sizes=1 / 2e9)
#     # Generator-Positionen extrahieren
# tra_buses = grid.transformers['bus1']
# tra_coords = grid.buses.loc[tra_buses][['x', 'y']]
#     # Generatoren rot darüberplotten
# plt.scatter(tra_coords['x'], tra_coords['y'], color='red', label='Transformers', zorder=5)
# plt.legend()
# plt.show()


#%% neue bbox laden
bbox_neu = func.compute_bbox_from_buses(grid)


#%% osm Data abrufen
Area, Area_features = func.get_osm_data(bbox_neu)

# #%% Plotten der Daten
# fig, ax = plt.subplots(figsize=(8, 8))
# ox.plot_graph(Area, ax=ax, show=False, close=False)
# Area_features.plot(ax=ax, facecolor="khaki", edgecolor="black", alpha=0.7)
# plt.show()

#%% Speichern der OSM Daten
Area_features.to_file("Area_features.geojson", driver="GeoJSON")
# Laden der Daten
gpd_area_features = gpd.read_file('Area_features.geojson')
print(gpd_area_features.head())



#%% Grid creation für erweiterte bbox

# Netz extrahieren
grid = ding0.load_grid(bbox_neu, grids_dir)

# Netz überschreiben
output_file = "dist_grid.nc"
grid.export_to_netcdf(output_file)
print(f"Teilnetz gespeichert als: {output_file}")

#%% Netz laden
net = pypsa.Network("dist_grid.nc")

# # Netz plotten (ohne Generator-Farben)
# net.plot(bus_sizes=1 / 2e9)
# # Generator-Positionen extrahieren
# tra_buses = net.transformers['bus1']
# tra_coords = net.buses.loc[tra_buses][['x', 'y']]
# # Generatoren rot darüberplotten
# plt.scatter(tra_coords['x'], tra_coords['y'], color='red', label='Transformers', zorder=5)
# plt.legend()
# plt.show()


#%% Daten kombinieren

# csv Datei Speicherung zur Prüfung der Tabellen
# net.buses.to_csv("grid_alt",index=False, sep=";")
# gpd_area_features.to_csv("gpd", index=False, sep=";")
# print(net.buses.head())
net.buses = dc.data_combination(net.buses, gpd_area_features)
#print(net.buses.head())
net.buses.to_csv("grid_neu", index=False, sep=";")




#%% Plotten des Netzes mit den OSM Daten

# Plot-Vorbereitung
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
# Netzplot
net.plot(ax=ax, bus_sizes=1 / 2e9, margin=1000)
# OSM-Daten
ox.plot_graph(Area, ax=ax, show=False, close=False)
Area_features.plot(ax=ax, facecolor="khaki", edgecolor="black", alpha=0.7)
# Generatoren markieren
generator_buses = net.transformers['bus1'].unique()
generator_coords = net.buses.loc[generator_buses][['x', 'y']]
ax.scatter(
    generator_coords['x'],
    generator_coords['y'],
    color='red',
    s=10,         # Punktgröße
    label='Generatoren',
    zorder=5      # überlagert andere Layer
)
ax.legend(loc="upper right")
plt.show()

#%% Weitere Daten definieren

#%%
# #%% BEISPIEL weitere Daten definieren

# """
# Funktionen im anderen Skript definieren

# Hier nur als Beispiel für eine Funktion die die Fläche zurück gibt
# """

# def Fläche(line):
#     return 5




# #%% BEISPIEL weitere Daten berechnen

# """
# Fläche für jedes Nodes bestimmen

# Hier nur als Beispiel für eine Funktion die die Fläche zurück gibt
# """

# gpd_area_features["Fläche"] = [None]*len(gpd_area_features)
# for i in range(len(gpd_area_features)):
#     line = gpd_area_features.iloc[i]
#     gpd_area_features.at[i, "Fläche"] = Fläche(line)

#     """
#     weitere Bestimmungen in selber Schleife möglich
#     oder weitere Schleifen starten, falls vorherige Daten benötigt werden
#     """


# # %% Daten laden
# """
# Vllt auch ganz am Anfang mit allen Packages direkt auch alle Daten laden
# """

# pd_Zensus_Bevoelkerung_100m = pd.read_csv("Zensus_Bevoelkerung_100m-Gitter.csv", sep=";")
# gpd_bundesland = gpd.read_file("georef-germany-postleitzahl.geojson")

# print("fertig")

# # %% Generieren weiterer Daten und in Gesamttabelle hinzufügen




# """
# Kann für weitere ZENSUS Daten beliebig erweitert werden
# """

# gpd_area_features["Bewohnerzahl"] = [None]*len(gpd_area_features)
# gpd_area_features["Zensus ID Zuordnung"] = [None]*len(gpd_area_features)

# for i in range(2): #len(Area_features_df)):
#     """
#     Abrufen der Koordinaten für jede Node
#     Extrahieren der Spalte und dann mit Index
#     """
#     #coordinate_x, coordiate_y = Area_features_df["Gauss_Krüger"][i]
#     coordinate_x = 4368251
#     coordinate_y = 2718751

#     gpd_area_features.at[i, "Zensus ID Zuordnung"] = func.gitter_ID(pd_Zensus_Bevoelkerung_100m, coordinate_x, coordinate_y)[2]

#     gpd_area_features.at[i, "Bewohnerzahl"] = func.get_population_count(pd_Zensus_Bevoelkerung_100m, gpd_area_features.at[i, "Zensus ID Zuordnung"]) 

    
# """
# Bundesland zuordnung
# """
# gpd_area_features["Bundesland"] = [None]*len(gpd_area_features)
# for i in range(2): #len(Area_features_df)):
#     gpd_area_features.at[i, "Bundesland"] = func.Bundesland(gpd_bundesland, gpd_area_features.at[i, "addr:postcode"])




# # %% Daten kontrollieren
# """
# Kontrollieren der Daten
# """
# print(gpd_area_features.iloc[1])

# gpd_area_features.to_file("Area_features_df.geojson", driver="GeoJSON")
# # %% Speichern aller Daten
# """
# Daten als Excel speichern
# """

# df = pd.DataFrame(gpd_area_features)
# excel_file = 'geodaten.xlsx'  # Name der Excel-Datei
# df.to_excel(excel_file, index=False)

# %%
