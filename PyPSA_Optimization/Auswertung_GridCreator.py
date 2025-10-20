# PyPSA importieren
import random
import numpy as np
import pandas as pd
import pypsa
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
import osmnx as ox
import random
import geopandas as gpd
import networkx as nx
import os
import pickle
from pathlib import Path


'''
Environment:

pypsa_gurobi

aktivieren!
'''

#%%

# Basis: das Verzeichnis zu GridCreator
this_dir = Path(__file__).parent

# Pfad zur .nc-Datei
file = this_dir / "output" / "grid_Schallstadt_GER_ecar.nc"

#%% Pypsa netzwerk einlesen aus .nc datei
network = pypsa.Network(file)

# #%% import area und features

# with open("../output/area_Schallstadt_GER.pkl", "rb") as f:
#     area = pickle.load(f)
# features = gpd.read_file("../output/features_Schallstadt_GER.gpkg")


# #%%
# # Nur Polygone (z. B. Gebäude, Flächen); Keine Nodes plotten!
# features_polygons = features[features.geometry.type.isin(["Polygon", "MultiPolygon"])]

# #%% Netzwerk plotten
# fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
# # Netzplot
# network.plot(ax=ax, bus_sizes=1 / 2e9, margin=1000)
# # OSM-Daten
# ox.plot_graph(area, ax=ax, show=False, close=False)
# # features_polygons.plot(ax=ax, facecolor="khaki", edgecolor="black", alpha=0.1)

# # Liste von Generator-Kategorien: (Filter-Funktion, Farbe, Label)
# # Busse nach Typ und Carrier
# gen_buses = {
#     'solar': set(network.generators[network.generators['carrier'] == 'solar']['bus']),
#     'HP': set(network.generators[network.generators['carrier'] == 'HP']['bus'])
#             }
# storage_buses = {
#     'E_Auto': set(network.storage_units[network.storage_units['carrier'] == 'E_Auto']['bus'])
#                }
# # Kategorien definieren: Busmengen, Farbe, Label
# gen_categories = [
#     (gen_buses['solar'], 'yellow', 'Solar Generatoren'),
#     (storage_buses['E_Auto'], 'green', 'E-Car Generatoren'),
#     (gen_buses['HP'], 'blue', 'HP Generatoren'),
#     (gen_buses['HP'] & gen_buses['solar'], 'purple', 'HP & Solar Generatoren'),
#     (gen_buses['HP'] & storage_buses['E_Auto'], 'pink', 'HP & E-Car Generatoren'),
#     (gen_buses['solar'] & storage_buses['E_Auto'], 'violet', 'Solar & E-Car Generatoren')
#                     ]
# for buses, color, label in gen_categories:
#     if buses:
#         coords = network.buses.loc[list(buses), ['x', 'y']]
#         ax.scatter(
#             coords['x'], coords['y'],
#             color=color,
#             s=20,
#             label=label,
#             zorder=5
#                     )
# # Trafo-Busse markieren
# tra_buses = network.transformers['bus1'].unique()
# tra_coords = network.buses.loc[tra_buses][['x', 'y']]
# ax.scatter(
#     tra_coords['x'],
#     tra_coords['y'],
#     color='red',
#     s=10,         # Punktgröße
#     label='Transformers',
#     zorder=5      # überlagert andere Layer
#             )
# ax.legend(loc='upper right')

# #%% Alle Gas generatoren auf p_nom_extendable = True setzen
# gas_gens = network.generators[network.generators['carrier'] == 'gas']
# for gen in gas_gens.index:
#     network.generators.at[gen, 'p_nom_extendable'] = True

#%% Random bus mit Generator ziehen
random_bus_with_gen = network.generators.sample()
print("Zufälliger Bus mit Generator:", random_bus_with_gen['bus'].values[0])
# Filtern der Generatoren am Bus
gens_at_bus = [g for g in network.generators.index if g.startswith(random_bus_with_gen['bus'].values[0])]
# Loads für diesen bus extrahieren und summieren
loads_at_bus = network.loads[network.loads['bus'] == random_bus_with_gen['bus'].values[0]]
# Random hp ziehen
random_hp = network.generators[network.generators['carrier'] == 'HP'].sample()
hp = random_hp.index[0]

#%% Plotten von Gesamtlast und Gesamterzeugung an dem bus
fig, ax = plt.subplots(figsize=(10, 5))
# Last-Summe
load_sum = network.loads_t.p_set[loads_at_bus.index].iloc[744*3:744*4].sum(axis=1)
ax.plot(load_sum, label='Summe aller Lasten', color='blue', linewidth=2)
# Generation-Summe
gen_sum = (network.generators_t.p_max_pu[gens_at_bus].iloc[744*3:744*4]*network.generators.p_nom[gens_at_bus]).sum(axis=1)
ax.plot(gen_sum, label='Summe aller Generatoren', color='orange', linewidth=2)
ax.plot(network.generators_t.p_max_pu[hp].iloc[744*3:744*4]*network.generators.p_nom[hp], label='HeatPump', color='green', linewidth=2)
ax.set_title(f'Last und Generation an Bus {random_bus_with_gen["bus"].values[0]}')
ax.set_xlabel('Zeit')
ax.set_ylabel('Leistung (MW)')
ax.legend()
plt.show()

#%% Hp übers jahr plotten
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(network.generators_t.p_max_pu[hp]*network.generators.p_nom[hp], label='HeatPump', color='green', linewidth=2)
ax.set_title(f'HeatPump Leistung über das Jahr an Bus {random_bus_with_gen["bus"].values[0]}')
ax.set_xlabel('Zeit')
ax.set_ylabel('Leistung (MW)')
ax.legend()
plt.show()

#%%
# random Store am bus ziehen
store = "BranchTee_mvgd_33775_lvgd_3165700001_building_779406E-Car_Storage"

#%%
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(network.stores_t.p[store], label='State of Charge', color='purple', linewidth=2)


ax.set_title(f'State of Charge der Stores an Bus {random_bus_with_gen["bus"].values[0]}')
ax.set_xlabel('Zeit')
ax.set_ylabel('State of Charge (MWh)')
ax.legend()
plt.show()

#%%

# snapshots auf eine Woche verkürzen für schnellere Optimierung
network.set_snapshots(network.snapshots[:150])

# #%% 
# # alle Leitungen auf erweiterbar setzen
# for line in network.lines.index:
#     network.lines.at[line, 's_nom_extendable'] = True

#%% 
# # carrier vn E-Autos von E_Auto auf battery setzen
# for su in network.storage_units.index:
#     if network.storage_units.at[su, 'carrier'] == 'E_Auto':
#         network.storage_units.at[su, 'carrier'] = 'battery'

# # %%
# # Alle storage units entfernen
# network.remove("Store", network.stores.index)


# #%%
# # für alle Store e_nom_extendable = True

# for su in network.stores.index:
#     network.stores.at[su, 'e_nom_extendable'] = True

# #%%
# # für alle Links e_nom_extendable = True

# for l in network.links.index:
#     network.links.at[l, 'p_nom_extendable'] = True



#%%
# für alle Links p_max_pu = 1

for l in network.links.index:
    network.links_t.p_max_pu[l] = 1

#%%
'''
Optimieren
'''
network.optimize(solver_name='gurobi', solver_options={'ResultFile':'model_all.ilp'}, snapshots=network.snapshots)
#%%
# Netzwerk speichern
network.export_to_netcdf("output/grid_Schallstadt_GER_optimize_ecar.nc")

#%% Plotten der Ergebnisse

#%% Energieflussbilanz
network.statistics.energy_balance.plot()
plt.show()

#%% Versorgungssicherheit
network.statistics.supply.plot()
plt.show()

#%% Versorgungssicherheit Jahresbilanz
network.statistics(aggregate_time=True).plot(kind="bar", stacked=True)
plt.show()


# %% Heatmap der Leitungsflüsse
import seaborn as sns
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(network.lines_t.p1.T, cmap="coolwarm", center=0, ax=ax)
ax.set_title("Leitungsflüsse über Zeit")
ax.set_xlabel("Zeit")
ax.set_ylabel("Leitungen")
plt.show()

#%% Auslastung der lines lines_t.p/n.lines_t.s_nom

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(network.lines_t.p1.T / network.lines.s_nom.T, cmap="coolwarm", center=0, ax=ax)
ax.set_title("Auslastung der Leitungen über Zeit")
ax.set_xlabel("Zeit")
ax.set_ylabel("Leitungen")
plt.show()

#%%

network.plot(bus_size = network.generators.groupby(['bus', 'carrier']).p_nom.sum())

#%%

network.statistics.curtailment(aggregate_time=False).droplevel(0).T.drop(['HP', 'E_Auto'], axis=1).plot()






# %%
'''
24 Plots für die ersten 24h
Netz dargestellt mit Leitungsflüssen in blau bis rot je nach Flussstärke
'''

from matplotlib import cm
cmap = cm.get_cmap("coolwarm")
norm = plt.Normalize(vmin=-1, vmax=1)  # Normalisierung für Farbskala
for hour in range(24):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    # Netzplot mit Flussfarben
    flows = network.lines_t.p0.iloc[hour]*1e3
    line_colors = [cmap(norm(flow)) for flow in flows]

    # Busfarben basierend auf Import oder Export
    bus_colors = {}
    for bus in network.buses.index:
        gens_at_bus = network.generators[network.generators['bus'] == bus].index
        if not gens_at_bus.empty:
            gen_sum = network.generators_t.p.loc[network.snapshots[hour], gens_at_bus].sum()
        else:
            gen_sum = 0
        load_sum = network.loads_t.p_set[network.loads[network.loads['bus'] == bus].index].sum(axis=1).iloc[hour] if not network.loads[network.loads['bus'] == bus].empty else 0
        net_flow = gen_sum - load_sum
        if net_flow > 0:
            bus_colors[bus] = 'green'   # Export
        elif net_flow < 0:
            bus_colors[bus] = 'red'    # Import
        else:
            bus_colors[bus] = 'gray'   # Neutral
    network.plot(ax=ax, bus_sizes=1 / 2e9, bus_colors=bus_colors, line_colors=line_colors, margin=1000)

    # Markierungen für Generatoren
    # Liste von Generator-Kategorien: (Filter-Funktion, Farbe, Label)
    gen_categories = [
        (lambda g: g['carrier'] == 'solar', 'yellow', 'Solar Generatoren'),
        (lambda g: g['carrier'] == 'E_car', 'green', 'E-Car Generatoren'),
        (lambda g: g['carrier'] == 'HP', 'blue', 'HP Generatoren'),
        (lambda g: (g['carrier'] == 'HP') & (g['carrier'] == 'solar'), 'purple', 'HP & Solar Generatoren'),
        (lambda g: (g['carrier'] == 'HP') & (g['carrier'] == 'E_car'), 'pink', 'HP & E-Car Generatoren'),
        (lambda g: (g['carrier'] == 'solar') & (g['carrier'] == 'E_car'), 'violet', 'Solar & E-Car Generatoren')
    ]

    for filt, color, label in gen_categories:
        gens = network.generators[filt(network.generators)]
        if not gens.empty:
            buses = gens['bus'].unique()
            coords = network.buses.loc[buses, ['x', 'y']]
            ax.scatter(
                coords['x'], coords['y'],
                color=color,
                s=20,
                label=label,
                zorder=5
            )

    # Trafo-Busse markieren
    tra_buses = network.transformers['bus1'].unique()
    tra_coords = network.buses.loc[tra_buses][['x', 'y']]
    for bus, (x, y) in tra_coords.iterrows():
        ax.text(
            x, y,
            'T',                     # Textlabel
            color='black',             # Farbe des Textes
            fontsize=12,             # Schriftgröße
            fontweight='bold',
            ha='center', va='center',# zentriert auf dem Bus
            zorder=10                # über anderen Layern
        )
    # ax.scatter(
    #     tra_coords['x'],
    #     tra_coords['y'],
    #     color='red',
    #     s=10,         # Punktgröße
    #     label='Transformers',
    #     zorder=5      # überlagert andere Layer
    # )

    # OSM-Daten
    ox.plot_graph(area, ax=ax, show=False, close=False)
    # features.plot(ax=ax, facecolor="khaki", edgecolor="black", alpha=0.1)
    ax.set_title(f'Leitungsflüsse – Stunde {hour + 1}')
    plt.show()
    plt.close(fig)

#%%

# Profil von einem Solargenerator, einem E-Auto und einer Wärmepumpe
solar_gen = network.generators[network.generators['carrier'] == 'solar'].sample().index[0]
e_car_gen = network.storage_units[network.storage_units['carrier'] == 'E_Auto'].sample().index[0]
hp_gen = network.generators[network.generators['carrier'] == 'HP'].sample().index[0]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(network.generators_t.p_max_pu[solar_gen], label=f'Solar Generator {solar_gen}', color='orange')
ax.plot(network.storage_units_t.state_of_charge_set[e_car_gen], label=f'E-Car Generator {e_car_gen}', color='green')
ax.plot(network.generators_t.p_max_pu[hp_gen], label=f'HP Generator {hp_gen}', color='blue')
ax.set_title('Leistung von Beispiel-Generatoren über 24 Stunden')
ax.set_xlabel('Zeit')
ax.set_ylabel('Leistung (kW)')
ax.legend()
plt.show()

#%%



#%%
'''
Zuzsammen als Raster
'''

fig, axs = plt.subplots(6, 4, figsize=(24, 18), subplot_kw={'projection': ccrs.PlateCarree()})

cmap = cm.get_cmap("coolwarm")
norm = plt.Normalize(vmin=-1, vmax=1)  # Normalisierung für Farbskala

for hour in range(24):
    ax = axs[hour // 4, hour % 4]  # 6 Reihen, 4 Spalten

    # Linienfarben
    flows = network.lines_t.p0.iloc[hour] * 1e3
    line_colors = [cmap(norm(flow)) for flow in flows]

    # Busfarben
    bus_colors = {}
    for bus in network.buses.index:
        gens_at_bus = network.generators[network.generators['bus'] == bus].index
        gen_sum = network.generators_t.p.loc[network.snapshots[hour], gens_at_bus].sum() if not gens_at_bus.empty else 0
        loads_at_bus = network.loads[network.loads['bus'] == bus].index
        load_sum = network.loads_t.p_set.loc[network.snapshots[hour], loads_at_bus].sum() if not loads_at_bus.empty else 0
        net_flow = gen_sum - load_sum
        if net_flow > 0:
            bus_colors[bus] = 'green'
        elif net_flow < 0:
            bus_colors[bus] = 'red'
        else:
            bus_colors[bus] = 'gray'

    # Netzplot in das Subplot
    network.plot(ax=ax, bus_sizes=1 / 2e9, bus_colors=bus_colors, line_colors=line_colors, margin=1000)
        # Markierungen für Generatoren
    # Liste von Generator-Kategorien: (Filter-Funktion, Farbe, Label)
    gen_categories = [
        (lambda g: g['carrier'] == 'solar', 'yellow', 'Solar Generatoren'),
        (lambda g: g['carrier'] == 'E_car', 'green', 'E-Car Generatoren'),
        (lambda g: g['carrier'] == 'HP', 'blue', 'HP Generatoren'),
        (lambda g: (g['carrier'] == 'HP') & (g['carrier'] == 'solar'), 'purple', 'HP & Solar Generatoren'),
        (lambda g: (g['carrier'] == 'HP') & (g['carrier'] == 'E_car'), 'pink', 'HP & E-Car Generatoren'),
        (lambda g: (g['carrier'] == 'solar') & (g['carrier'] == 'E_car'), 'violet', 'Solar & E-Car Generatoren')
    ]

    for filt, color, label in gen_categories:
        gens = network.generators[filt(network.generators)]
        if not gens.empty:
            buses = gens['bus'].unique()
            coords = network.buses.loc[buses, ['x', 'y']]
            ax.scatter(
                coords['x'], coords['y'],
                color=color,
                s=20,
                label=label,
                zorder=5
            )

    # Trafo-Busse markieren
    tra_buses = network.transformers['bus1'].unique()
    tra_coords = network.buses.loc[tra_buses][['x', 'y']]
    for bus, (x, y) in tra_coords.iterrows():
        ax.text(
            x, y,
            'T',                     # Textlabel
            color='black',             # Farbe des Textes
            fontsize=12,             # Schriftgröße
            fontweight='bold',
            ha='center', va='center',# zentriert auf dem Bus
            zorder=10                # über anderen Layern
        )

    # OSM-Daten
    ox.plot_graph(area, ax=ax, show=False, close=False)
    features.plot(ax=ax, facecolor="khaki", edgecolor="black", alpha=0.1)
    ax.set_title(f'Leitungsflüsse – Stunde {hour + 1}')


    ax.set_title(f'Stunde {hour + 1}')

plt.tight_layout()
plt.show()

# %%
'''
Flussgröße für jede Stunde für jede Linie
'''

for hour in range(24):
    fig, ax = plt.subplots(figsize=(8, 5))
    flows = network.lines_t.p0.iloc[hour]
    # Balkenplot
    ax.bar(np.arange(len(flows)), flows, color="steelblue")
    # X-Achse: nur Zahlen statt voller Leitungnamen
    ax.set_xticks(np.arange(len(flows)))
    ax.set_xticklabels(np.arange(1, len(flows)+1))
    ax.set_title(f"Leitungsflüsse – Stunde {hour + 1}")
    ax.set_xlabel("Leitung")
    ax.set_ylabel("Leistungsfluss (MW)")
    plt.tight_layout()
    plt.show()
    plt.close(fig)

# %%
'''
als Raster
'''

fig, axs = plt.subplots(6, 4, figsize=(24, 18))  # 6 Reihen, 4 Spalten

for hour in range(24):
    ax = axs[hour // 4, hour % 4]
    flows = network.lines_t.p0.iloc[hour]    
    # Balkenplot
    ax.bar(np.arange(len(flows)), flows, color="steelblue")
    # X-Achse: nur Zahlen statt voller Leitungnamen
    ax.set_xticks(np.arange(len(flows)))
    ax.set_xticklabels(np.arange(1, len(flows)+1))
    ax.set_title(f"Stunde {hour + 1}")
    ax.set_xlabel("Leitung")
    ax.set_ylabel("Leistungsfluss (MW)")

plt.tight_layout()
plt.show()
# %%
