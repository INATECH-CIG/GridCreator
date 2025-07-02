#%% Import section

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
# import openpyxl # für pycity erforderlich
import functions as func
import grid_creation as gc
import pypsa



#%% GPS Data eingeben


""""
Definieren des zu untersuchenden Bereichs
Eingabe von Rechteck mit Koordinaten
Koordinaten in WGS84 System
"""


# top =  48.001143  # Upper latitude
# bottom = 47.997224  # Lower latitude
# left =  7.837898   # Right longitude
# right =  7.838992  # Left longitude

top =  49.463 # # Upper latitude
bottom = 49.460  # Lower latitude
left =  11.397    # Right longitude
right =  11.402   # Left longitude
bbox = [left, bottom, right, top]

Area = ox.graph_from_bbox(top, bottom, right, left, network_type="all")



#%% osm Data abrufen


""""
Abrufen der OSM Daten für den definierten Bereich
Die Funktion get_osm_data() ist in functions_Linux.py definiert
Tags sind in der Funktion definiert
"""



Area_features = func.get_osm_data(top, bottom, right, left)



#%% Plotten der Daten

"""
Visualisierung der OSM Daten
"""

fig, ax = plt.subplots(figsize=(8, 8))
ox.plot_graph(Area, ax=ax, show=False, close=False)
Area_features.plot(ax=ax, facecolor="khaki", edgecolor="black", alpha=0.7)
plt.show()



#%% Data Management

"""
Speichern der Rohdaten
Laden der Daten in ein GeoDataFrame, für weitere Datensammlung
"""

Area_features.to_file("Area_features.geojson", driver="GeoJSON")


gpd_area_features = gpd.read_file('Area_features.geojson')

print(gpd_area_features.head())



# #%% Grid erstellen

"""
Erstellen von Grid mit Ding0

"""

''''
Aufrufen der Funktion, um aus vorgefertigten Grids, den gesuchten Bereich zu Filtern
'''

# Namen des Ordners mit den Grids
grids_dir = "grids" 
# Grid laden
grid = gc.create_grid(bbox, grids_dir)

# Grid speichern
output_file = "dist_grid.nc"
grid.export_to_netcdf(output_file)


#%% Netz laden und plotten
net = pypsa.Network("dist_grid.nc")
net.plot(
    bus_sizes=1 / 2e9,
)
plt.show()


"""

Anschließendes Kombinieren der Area_features_df mit dem Grid
Ding0 gibt Knoten als .csv Datei als "Buses" raus. Diese Datei kann eingelesen werden und mit GeoDataFrame kombiniert werden.
Beide Tabellen haben Koordinaten für jeweilige Zeilen.
Tabelle soll anschließend Pypsa kompatibel sein
"""



# """
# Extrahieren der Koordinaten von Area_features_df
# """
# gpd_area_features['x'] = gpd_area_features.geometry.x
# gpd_area_features['y'] = gpd_area_features.geometry.y

# """
# .nc Datei von ding0  ist eingelesen, einfach net.buses verwenden
# """

# """
# Beide DataFrames benötigen gleiches Format der Koordinaten
# Dann zusammenfügen beider DataFrames anhand der Koordinaten
# """


# net.buses = pd.merge(net.buses, gpd_area_features.drop(columns='geometry'), on=['x', 'y'], how='left')




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



# %% Daten laden
"""
Vllt auch ganz am Anfang mit allen Packages direkt auch alle Daten laden
"""

pd_Zensus_Bevoelkerung_100m = pd.read_csv("Zensus2022_Bevoelkerungszahl_100m-Gitter.csv", sep=";")


gpd_bundesland = gpd.read_file("georef-germany-postleitzahl.geojson")



#%% Generieren weiterer Daten und in Gesamttabelle hinzufügen

"""
Daten aus Area_features_df müssen zuerst mit Grid von Ding0 kombiniert werden
ACHTUNG: Spaltennamen können sich ändern
ACHTUNG: df Name kann sich ändern
"""




"""
Koordinaten umrechnen

ACHTUNG: in Area_features_df sind die Koordinaten in POINT() gespeichert

"""
print("fertig")
print(gpd_area_features)
gpd_area_features["Gauss_Krüger"] = [None]*len(gpd_area_features)

for i in range(2): #len(Area_features_df)):
    lon, lat = gpd_area_features["geometry"][i].x, gpd_area_features["geometry"][i].y
    #lon = Area_features_df["lon"][i]
    #lat = Area_features_df["lat"][i]

    rechtswert, hochwert = func.WG_zu_Gauss(lon, lat)

    gpd_area_features.at[i, "Gauss_Krüger"] = [rechtswert, hochwert]




"""
Kann für weitere ZENSUS Daten beliebig erweitert werden
"""

gpd_area_features["Bewohnerzahl"] = [None]*len(gpd_area_features)
gpd_area_features["Zensus ID Zuordnung"] = [None]*len(gpd_area_features)

for i in range(2): #len(Area_features_df)):
    """
    Abrufen der Koordinaten für jede Node
    Extrahieren der Spalte und dann mit Index
    """
    #coordinate_x, coordiate_y = Area_features_df["Gauss_Krüger"][i]
    coordinate_x = 4368251
    coordinate_y = 2718751

    gpd_area_features.at[i, "Zensus ID Zuordnung"] = func.gitter_ID(pd_Zensus_Bevoelkerung_100m, coordinate_x, coordinate_y)[2]

    gpd_area_features.at[i, "Bewohnerzahl"] = func.get_population_count(pd_Zensus_Bevoelkerung_100m, gpd_area_features.at[i, "Zensus ID Zuordnung"]) 



print(gpd_area_features.columns)
"""
Bundesland zuordnung
"""
gpd_area_features["Bundesland"] = [None]*len(gpd_area_features)
for i in range(2): #len(Area_features_df)):
    gpd_area_features.at[i, "Bundesland"] = func.Bundesland(gpd_bundesland, gpd_area_features.at[i, "addr:postcode"])


print("fertig")

# %% Daten kontrollieren
"""
Kontrollieren der Daten
"""
print(gpd_area_features.iloc[1])

# gpd_area_features.to_file("Area_features_df.geojson", driver="GeoJSON")
# # %% Speichern aller Daten
# """
# Daten als Excel speichern
# """

# df = pd.DataFrame(gpd_area_features)
# excel_file = 'geodaten.xlsx'  # Name der Excel-Datei
# df.to_excel(excel_file, index=False)
