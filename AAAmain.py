import main_function as mf
import matplotlib.pyplot as plt
import osmnx as ox
import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd
import data as data

#%% STEP 1

# Definieren von bbox

# top =  54.48594882134629 # # Upper latitude
# bottom = 54.47265521486088 # Lower latitude
# left =  11.044533685584453    # Right longitude
# right =  11.084893505520695  # Left longitude

top =  49.374518600877046 # # Upper latitude
bottom = 49.36971937206515 # Lower latitude
left =  12.697361279468392   # Right longitude
right =  12.708888681047798  # Left longitude

bbox = [left, bottom, right, top]

# Speichern vom Grid
output_file_grid = "dist_grid.nc"
grids_dir = "grids" 

#Grid creation
grid_1, bbox_1 = mf.ding0_grid(bbox, grids_dir, output_file_grid)


"""
Check durch PLOT
"""

# Netz plotten (ohne Generator-Farben)
grid_1.plot(bus_sizes=1 / 2e9)
# Generator-Positionen extrahieren
tra_buses = grid_1.transformers['bus1']
tra_coords = grid_1.buses.loc[tra_buses][['x', 'y']]
# Generatoren rot darüberplotten
plt.scatter(tra_coords['x'], tra_coords['y'], color='red', label='Transformers', zorder=5)
plt.legend()
plt.show()



#%% STEP 2

# Daten laden

# OSM Daten
buffer = 0.0002  # entspricht ungefähr 20 m
grid_2, area, features = mf.osm_data(grid_1, bbox_1, buffer)

# Bundesland-Daten
gpd_bundesland = gpd.read_file("georef-germany-postleitzahl.geojson")

# # buses als csv speichern
# grid_2.buses.to_csv("buses.csv")

# Zensus Daten
"""
Ordner zensus_daten enthält nur kleinen Tewil aller Zensus-Tabellen, um die Ladezeiten zu verkürzen.
Alle Tabellen sind im Ordner zensus_daten_all enthalten.
"""
ordner = "zensus_daten"
zensus = mf.daten_laden(ordner)

"""
Daten gespeichert, um sie ab jetzt nur noch direkt in einem DataFrame zu laden, spart Zeit!
"""
# zensus.to_pickle('Zensus_daten_als_DataFrame.pkl')
# zensus = pd.read_pickle('Zensus_daten_als_DataFrame.pkl')

grid_3 = mf.daten_zuordnung(grid_2, gpd_bundesland, zensus)


"""
Check durch PLOT
"""

#%% Plot-Vorbereitung
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})

# Netzplot
grid_2.plot(ax=ax, bus_sizes=1 / 2e9, margin=1000)

# OSM-Daten
ox.plot_graph(area, ax=ax, show=False, close=False)
features.plot(ax=ax, facecolor="khaki", edgecolor="black", alpha=0.7)
# Generatoren markieren
generator_buses = grid_2.transformers['bus1'].unique()
generator_coords = grid_2.buses.loc[generator_buses][['x', 'y']]
ax.scatter(
    generator_coords['x'],
    generator_coords['y'],
    color='red',
    s=10,         # Punktgröße
    label='Generatoren',
    zorder=5      # überlagert andere Layer
)
bus_coords = grid_2.buses[['x', 'y']]
ax.scatter(
    bus_coords['x'],
    bus_coords['y'],
    color='blue',
    s=5,            # kleinere Punktgröße
    label='Busse',
    zorder=4        # leicht unter Generatoren, über Netz
)

ax.legend(loc="upper right")
xmin, ymin, xmax, ymax = bbox_1
ax.set_xlim(xmin - buffer, xmax + buffer)
ax.set_ylim(ymin - buffer, ymax + buffer)
plt.show()
# %%


#%% STEP 3

# Technik Zuordnen

# Zensusdaten für Bundesland
Bev_data_Zensus = mf.bundesland_zensus(zensus, datei = "DE_VG5000.gpkg")

# Technik definieren
Technik = ['solar']

# Technik zuordnen
grid_4, factor_bbox = mf.technik_zuordnen(grid_3, data.faktoren_technik, data.kategorien_eigenschaften,  Bev_data_Zensus, data.Bev_data_Technik, Technik)
grid_5 = mf.technik_fill(grid_4, Technik, factor_bbox)