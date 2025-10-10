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


'''
Environment:

pypsa_gurobi

aktivieren!
'''

#%% Pypsa netzwerk einlesen aus .nc datei
network = pypsa.Network("input/dist_grid_for_optimize.nc")

#%% import area und features

with open("input/area_for_optimize.pkl", "rb") as f:
    area = pickle.load(f)
features = gpd.read_file("input/features_for_optimize.gpkg")

#%%
#%% Pypsa netzwerk einlesen aus .nc datei
network_opf = pypsa.Network("input/grid_1_STEP_5_Opfingen.nc")

#%% import area und features

with open("input/area_Opfingen.pkl", "rb") as f:
    area_opf = pickle.load(f)
features_opf = gpd.read_file("input/features_Opfingen.gpkg")




#%%


# Fix Capacity
grid_1.optimize.fix_optimal_capacities()

# Set snapshots für Optimierung
start_time = pd.Timestamp("2023-01-01 00:00:00")
end_time = pd.Timestamp("2023-01-07 23:00:00")
snapshots = pd.date_range(start=start_time, end=end_time, freq='h')
grid_1.set_snapshots(snapshots)

#%%

grid_1.export_to_netcdf("input/dist_grid_for_optimize.nc")

#%%
grid_1.optimize()




#%%
# Loads von Buses aus Network_opf an jeweiligen bus von network anhängen
# und p_set übertragen
for bus in network.buses.index:
    if bus in network_opf.buses.index:
        loads_at_bus = network_opf.loads[network_opf.loads['bus'] == bus]
        for load in loads_at_bus.index:
            new_load_name = f"{load}_from_network"
            network.add("Load",
                            name=new_load_name,
                            bus=bus,
                            p_set=network_opf.loads_t.p_set[load],
                            controllable=network_opf.loads.at[load, 'controllable'] if 'controllable' in network_opf.loads.columns else False,
                            type=network_opf.loads.at[load, 'type'] if 'type' in network_opf.loads.columns else None,
                            )


#%%










#%% Netzwerk plotten


fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
# Netzplot
network.plot(ax=ax, bus_sizes=1 / 2e9, margin=1000)
# OSM-Daten
ox.plot_graph(area_opf, ax=ax, show=False, close=False)
features.plot(ax=ax, facecolor="khaki", edgecolor="black", alpha=0.1)
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
ax.scatter(
    tra_coords['x'],
    tra_coords['y'],
    color='red',
    s=10,         # Punktgröße
    label='Transformers',
    zorder=5      # überlagert andere Layer
)


ax.legend(loc='upper right')

#%%
# Alle Gas generatoren auf p_nom_extendable = True setzen
gas_gens = network.generators[network.generators['carrier'] == 'gas']
for gen in gas_gens.index:
    network.generators.at[gen, 'p_nom_extendable'] = True


# #%%
# Im Netz von Freiburg entstehen doppelte Generatoren?
# network.generators.drop('Generator_am_Transformer_lv_grid_8507500106_reinforced_2', inplace=True)

# network.generators.drop('Storage_am_Transformer_lv_grid_8507500106_reinforced_2', inplace=True)

# #%%
# Für alle solar generatoren p_min_pu = p_max_pu setzen
# for gen in network.generators.index:
#     if network.generators.at[gen, 'carrier'] == 'solar':
#         network.generators.at[gen, 'p_min_pu'] = network.generators.at[gen, 'p_max_pu']


# #%%
# # Gruppieren der Lasten nach Bus
# loads_grouped = network.loads.groupby(network.loads['bus'])

# # Random bus mit Generator ziehen
# random_bus_with_gen = network.generators.sample()
# print("Zufälliger Bus mit Generator:", random_bus_with_gen['bus'].values[0])

# # Filtern der Lasten am Bus
# filtered_group = loads_grouped.get_group(random_bus_with_gen['bus'].values[0])

# # Filtern der Generatoren am Bus
# gens_at_bus = [g for g in network.generators.index if g.startswith(random_bus_with_gen['bus'].values[0])]

#%%
# einen random bus mit generator filtern

random_bus_with_gen = network.generators.sample()

# alle loads_t.p_set die an diesem bus sind filtern und aufsummieren
loads_at_bus = network.loads[network.loads['bus'] == random_bus_with_gen['bus'].values[0]]
filtered_group = loads_at_bus
# alle generatoren an diesem bus filtern
gens_at_bus = network.generators[network.generators['bus'] == random_bus_with_gen['bus'].values[0]].index
print(f"Zufälliger Bus mit Generator: {random_bus_with_gen['bus'].values[0]}")
print(f"Anzahl der Lasten an diesem Bus: {len(filtered_group)}")
print(f"Anzahl der Generatoren an diesem Bus: {len(gens_at_bus)}")
print(f"Lasten an diesem Bus:\n{filtered_group}")
print(f"Generatoren an diesem Bus:\n{gens_at_bus}")




#%%
# Plotten von Gesamtlast und Gesamterzeugung an dem bus
fig, ax = plt.subplots(figsize=(10, 5))
# Last-Summe
load_sum = network.loads_t.p_set[filtered_group.index].sum(axis=1)
ax.plot(load_sum, label='Summe aller Lasten', color='blue', linewidth=2)
# Generation-Summe
gen_sum = network.generators_t.p_max_pu[gens_at_bus].sum(axis=1)*1e-3
ax.plot(gen_sum, label='Summe aller Generatoren', color='orange', linewidth=2)
ax.set_title(f'Last und Generation an Bus {random_bus_with_gen["bus"].values[0]}')
ax.set_xlabel('Zeit')
ax.set_ylabel('Leistung (MW)')
ax.legend()
plt.show()


#%%
'''
Optimieren
'''

network.optimize(solver_name='gurobi', solver_options={'ResultFile':'model_all.ilp'}, snapshots=network.snapshots[0])


# %%
'''
Plot der Flüsse der ersten 24h
'''
fig, ax = plt.subplots(figsize=(10, 5))
for line in network.lines.index:
    ax.plot(network.lines_t.p0[line][:24], label=line)
ax.set_title('Leitungsflüsse der ersten 24 Stunden')
ax.set_xlabel('Zeit (Stunden)')
ax.set_ylabel('Leistungsfluss (MW)')
ax.legend().remove()  # Legende entfernen, wenn zu viele Linien vorhanden sind
plt.show()
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
    features.plot(ax=ax, facecolor="khaki", edgecolor="black", alpha=0.1)
    ax.set_title(f'Leitungsflüsse – Stunde {hour + 1}')
    plt.show()
    plt.close(fig)

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
