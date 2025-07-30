import main_function as mf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
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
grid_1_copy = grid_1.copy()
grid_2, area, features = mf.osm_data(grid_1_copy, bbox_1, buffer)

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

grid_2_copy = grid_2.copy()
grid_3 = mf.daten_zuordnung(grid_2_copy, gpd_bundesland, zensus)


# Plot-Vorbereitung
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

#%% STEP 3

# Technik Zuordnen

# Zensusdaten für Bundesland
Bev_data_Zensus = mf.bundesland_zensus(zensus, datei = "DE_VG5000.gpkg")

# Technik definieren
Technik = ['solar']

# Technik zuordnen
grid_3_copy = grid_3.copy()
grid_4, factor_bbox = mf.technik_zuordnen(grid_3_copy, data.faktoren_technik, data.kategorien_eigenschaften,  Bev_data_Zensus, data.Bev_data_Technik, Technik)
grid_4_copy = grid_4.copy()
grid_5 = mf.technik_fill(grid_4_copy, Technik, factor_bbox)



#%% STEP 4

# Zeitreihen hinzufügen

'''
Zeitreihen sind spezifischen buses zugeordnet!
Funktioniert also auch nur für ausgewählte Koordinaten!
'''

grid_5_copy = grid_5.copy()
grid_6 = mf.loads_zuordnen(grid_5_copy)


#%% STEP 5

# Grid für pysa.optimze() vorbereiten
grid_6_copy = grid_6.copy()
grid_7 = mf.pypsa_vorbereiten(grid_6_copy)


# .optimize()
grid_7.optimize()


#%% Ergebnisse plotten

# Color Map für Powerflow
cmap = plt.colormaps.get_cmap('coolwarm')  # Neue API für Colormap

# Zeitschritte einzelnd plotten
for i, snapshot in enumerate(grid_7.snapshots):
    pf = grid_7.lines_t.p0.loc[snapshot]

    # Normalisieren
    norm = colors.Normalize(vmin=pf.min(), vmax=pf.max())
    line_colors = [cmap(norm(val)) for val in pf]

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_dict = grid_7.plot.map(line_colors=line_colors, bus_sizes=1e-9, line_widths=2)
    plt.title(f"Power Flow - Stunde {i}")

    # Farbskala hinzufügen
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # notwendig für colorbar
    fig.colorbar(sm, ax=ax, label="Power Flow [MW]")

    plt.show()



# Ale Transformator-Generatoren plotten

fig, ax = plt.subplots()
ax.set_xlabel("Stunde")
ax.set_ylabel("Leistung [MW]")

# Nur Generatoren mit carrier "gas"
gas_generators = grid_7.generators[grid_7.generators.carrier == "gas"]

# Für jeden dieser Generatoren die Zeitreihe plotten
for name in gas_generators.index:
    ax.plot(grid_7.generators_t.p.index, grid_7.generators_t.p[name], label=name)

ax.axhline(0, color="gray", linewidth=0.5)
ax.legend(fontsize="small", loc="upper right", bbox_to_anchor=(1.3, 1.0))
plt.title("Generatorleistungen (nur an Transformatoren)")
plt.tight_layout()
plt.show()


# %%
