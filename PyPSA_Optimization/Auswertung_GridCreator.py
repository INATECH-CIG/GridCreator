# PyPSA importieren
import random
import numpy as np
import pandas as pd
import pypsa
import cartopy.crs as ccrs
import matplotlib.cm as cm

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
this_dir = Path(__file__).parent.parent

# Pfad zur .nc-Datei
file = this_dir / "input" / "grid_Schallstadt_fast_optimized.nc"

#%% Pypsa netzwerk einlesen aus .nc datei
network = pypsa.Network(file)

#%% import area und features

# Pfad zur area-Pickle-Datei
area_file = this_dir / "input" / "area_Schallstadt_fast.pkl"

# Laden
with open(area_file, "rb") as f:
    area = pickle.load(f)
# features = gpd.read_file("../output/features_Schallstadt_GER.gpkg")

#%%
# bus_size_set definieren
# alle buses mit carrier external_supercharger = 0 setzen
bus_size_set = pd.Series(index=network.buses.index, dtype=object)

# Alle Bussen mit Carrier "external_supercharger" auf 0 setzen
for bus in network.buses.index:
    if "external_supercharger" in bus:
        bus_size_set.at[bus] = 0
    else:
        bus_size_set.at[bus] = 1 / 2e9  # Standardgröße
#%%
# Nur Polygone (z. B. Gebäude, Flächen); Keine Nodes plotten!
# features_polygons = features[features.geometry.type.isin(["Polygon", "MultiPolygon"])]

#%% Netzwerk plotten
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
# Netzplot
network.plot(ax=ax, bus_sizes=bus_size_set, margin=1000)


# OSM-Daten
ox.plot_graph(area, ax=ax, show=False, close=False)
# features_polygons.plot(ax=ax, facecolor="khaki", edgecolor="black", alpha=0.1)

# Liste von Generator-Kategorien: (Filter-Funktion, Farbe, Label)
# Busse nach Typ und Carrier
gen_buses = {
    'solar': set(network.generators[network.generators.index.to_series().str.contains('_solar')]['bus']),
    'HP': set(network.generators[network.generators.index.to_series().str.contains('_HP')]['bus'])
            }
storage_buses = {
    'E_Auto': set(network.buses[network.buses.index.isin(network.links['bus0'])].index)
               }
# Kategorien definieren: Busmengen, Farbe, Label
gen_categories = [
    (gen_buses['solar'], 'yellow', 'Solar Generatoren'),
    (storage_buses['E_Auto'], 'green', 'E-Car Generatoren'),
    (gen_buses['HP'], 'blue', 'HP Generatoren'),
    (gen_buses['HP'] & gen_buses['solar'], 'purple', 'HP & Solar Generatoren'),
    (gen_buses['HP'] & storage_buses['E_Auto'], 'pink', 'HP & E-Car Generatoren'),
    (gen_buses['solar'] & storage_buses['E_Auto'], 'violet', 'Solar & E-Car Generatoren')
                    ]
for buses, color, label in gen_categories:
    if buses:
        coords = network.buses.loc[list(buses), ['x', 'y']]
        ax.scatter(
            coords['x'], coords['y'],
            color=color,
            s=20,
            label=label,
            zorder=5
                    )
        
# lines mit v_nom > 1 in grün plotten
high_voltage_lines = network.lines[network.lines['v_nom'] > 1]
# alle line colors in braun und die highj voltage lines in  grün
for line in network.lines.index:
    line_coords = network.buses.loc[[network.lines.at[line, 'bus0'], network.lines.at[line, 'bus1']], ['x', 'y']]
    if line in high_voltage_lines.index:
        ax.plot(
            line_coords['x'],
            line_coords['y'],
            color='green',
            linewidth=1,
            zorder=3
                )
    else:
        ax.plot(
            line_coords['x'],
            line_coords['y'],
            color='brown',
            linewidth=0.5,
            zorder=2
                )

# Generator bus mit conventional im namen
# gen_buses['conventional'] = set(network.generators[network.generators.index.to_series().str.contains('_conventional')]['bus'])
# # Plotten in rot
# if gen_buses['conventional']:
#     coords = network.buses.loc[list(gen_buses['conventional']), ['x', 'y']]
#     ax.scatter(
#         coords['x'], coords['y'],
#         color='brown',
#         s=20,
#         label='Conventional Generatoren',
#         zorder=5
#                 )
# # Trafo-Busse markieren
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
# bus random auswählen
random_load = random.choice(network.loads.index.tolist())
random_bus = network.loads.at[random_load, 'bus']
random_gen = network.generators[network.generators['bus'] == random_bus].index.tolist()
random_st = network.stores[network.stores['bus'] == random_bus].index.tolist()

# plot von Load
fig, ax = plt.subplots()
network.loads_t.p[f'{random_bus}_load_1'].plot(ax=ax, label='Load', color='blue')
if random_gen:
    for gen in random_gen:
        network.generators_t.p[gen].plot(ax=ax, label=f'Generator: {gen}', linestyle='--')
if random_st:
    for st in random_st:
        network.stores_t.e[st].plot(ax=ax, label=f'Storage: {st}', linestyle='--')
plt.title(f'Load Profile at Bus: {random_bus}')
plt.xlabel('Time')
plt.ylabel('Power (kW)')
plt.legend()
plt.show()
plt.close()


#%%
# Beispielplot für ein E-Auto mit Speicher, Lade- und Entladeleistung, Backup-Generator
fig, ax = plt.subplots()
bus = 'BranchTee_mvgd_31910_lvgd_3165600002_building_778656'
network.links_t.p0[f'{bus}_E_Car_Connector_charge'].plot(ax=ax, label='E-Car Connector Charge', color='orange')
network.links_t.p0[f'{bus}_E_Car_Connector_discharge'].plot(ax=ax, label='E-Car Connector Discharge', color='blue')
network.stores_t.e[f'{bus}_E_Car_Storage'].plot(ax=ax, label='E-Car Storage', color='green')
network.loads_t.p[f'{bus}_E_Car_Load'].plot(ax=ax, label='E-Car Load', color='red')
network.generators_t.p[f'external_supercharger_{bus}_E_Car_Storage'].plot(ax=ax, label='external_supercharger Generator', color='black')
ax2 = ax.twinx()
network.links_t.p_max_pu[f'{bus}_E_Car_Connector_charge'].plot(ax=ax2, label='E-Car Connector Charge Max PU', color='orange', linestyle='--')
ax.legend()
#%%
# selber plot nur für eine Woche
fig, ax = plt.subplots()
bus = 'BranchTee_mvgd_31910_lvgd_3165600002_building_778656'
network.links_t.p0[f'{bus}_E_Car_Connector_charge'].loc[network.snapshots[:168]].plot(ax=ax, label='E-Car Connector Charge', color='orange')
network.links_t.p0[f'{bus}_E_Car_Connector_discharge'].loc[network.snapshots[:168]].plot(ax=ax, label='E-Car Connector Discharge', color='blue')
network.stores_t.e[f'{bus}_E_Car_Storage'].loc[network.snapshots[:168]].plot(ax=ax, label='E-Car Storage', color='green')
network.loads_t.p[f'{bus}_E_Car_Load'].loc[network.snapshots[:168]].plot(ax=ax, label='E-Car Load', color='red')
network.generators_t.p[f'external_supercharger_{bus}_E_Car_Storage'].loc[network.snapshots[:168]].plot(ax=ax, label='external_supercharger Generator', color='black')
ax2 = ax.twinx()
network.links_t.p_max_pu[f'{bus}_E_Car_Connector_charge'].loc[network.snapshots[:168]].plot(ax=ax2, label='E-Car Connector Charge Max PU', color='orange', linestyle='--')
ax.legend()

#%%
# Beispiel Plot für einen Solar Generator mit der LOad an dem Bus
fig, ax = plt.subplots()
bus = 'BranchTee_mvgd_31910_lvgd_3165600006_building_778342'
network.generators_t.p[f'{bus}_solar'].plot(ax=ax, label='Solar Generator', color='orange')
network.loads_t.p[f'{bus}_load_1'].plot(ax=ax, label='Load', color='blue')
ax.legend()

#%%
# selber Plot nur für eine Woche
fig, ax = plt.subplots()
bus = 'BranchTee_mvgd_31910_lvgd_3165600006_building_778342'
network.generators_t.p[f'{bus}_solar'].loc[network.snapshots[:168]].plot(ax=ax, label='Solar Generator', color='orange')
network.loads_t.p[f'{bus}_load_1'].loc[network.snapshots[:168]].plot(ax=ax, label='Load', color='blue')
ax.legend()

#%% Statistics ausgeben

# all
# Installierte Kapazität
installed_capacity = network.statistics.installed_capacity()
# Operational Expenditure (Betriebsausgaben), also laufende Kosten für den täglichen Geschäftsbetrieb
opex = network.statistics.opex()
# CAPEX steht für Capital Expenditures (Investitionsausgaben) und bezeichnet die Ausgaben
capex = network.statistics.capex()
# Berechnet die Investitionsausgaben für erweiterte Kapazitäten
expanded_capex = network.statistics.expanded_capex()

# grouped
installed_capacity_carrier = network.statistics.installed_capacity(groupby='carrier')
opex_carrier = network.statistics.opex(groupby='carrier')
capex_carrier = network.statistics.capex(groupby='carrier')
expanded_capex_carrier = network.statistics.expanded_capex(groupby='carrier')

#%%

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.EqualEarth()}, figsize=(10, 10))
network.plot()
#%% Plot der Statistcs

installed_capacity.plot(kind='bar', figsize=(10, 6))
plt.title('Installed Capacity by Component Type')
plt.xlabel('Component Type')
plt.ylabel('Installed Capacity (MW)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
plt.close()
#%%
opex.droplevel(0).plot(kind='bar', figsize=(10, 6))
plt.title('opex by Component Type')
plt.xlabel('Component Type')
plt.ylabel('opex (MW)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
plt.close()
#%%
capex.droplevel(0).plot(kind='bar', figsize=(10, 6))
plt.title('capex by Component Type')
plt.xlabel('Component Type')
plt.ylabel('capex (MW)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
plt.close()
#%%
expanded_capex.droplevel(0).plot(kind='bar', figsize=(10, 6))
plt.title('expanded_capex by Component Type')
plt.xlabel('Component Type')
plt.ylabel('expanded_capex (MW)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
plt.close()
#%%
# grouped by carrier
installed_capacity_carrier.droplevel(0).plot(kind='bar', figsize=(10, 6))
plt.title('Installed Capacity_carrier by Component Type')
plt.xlabel('Component Type')
plt.ylabel('Installed Capacity_carrier (MW)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
plt.close()
#%%
opex_carrier.droplevel(0).plot(kind='bar', figsize=(10, 6))
plt.title('opex_carrier by Component Type')
plt.xlabel('Component Type')
plt.ylabel('opex_carrier (MW)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
plt.close()
#%%
capex_carrier.droplevel(0).plot(kind='bar', figsize=(10, 6))
plt.title('capex_carrier by Component Type')
plt.xlabel('Component Type')
plt.ylabel('capex_carrier (MW)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
plt.close()

#%%
expanded_capex_carrier.droplevel(0).plot(kind='bar', figsize=(10, 6))
plt.title('expanded_capex_carrier by Component Type')
plt.xlabel('Component Type')
plt.ylabel('expanded_capex_carrier (MW)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
plt.close()

#%% Energieflussbilanz
network.statistics.energy_balance(aggregate_time=False).loc[:, network.snapshots[:168]].T.plot()
plt.show()

#%% Versorgungssicherheit
network.statistics.supply.plot()
plt.show()

#%% Versorgungssicherheit Jahresbilanz
network.statistics(aggregate_time=True).plot(kind="bar", stacked=True)
plt.show()


# # %% Heatmap der Leitungsflüsse
# import seaborn as sns
# fig, ax = plt.subplots(figsize=(12, 8))
# sns.heatmap(network.lines_t.p1.T, cmap="coolwarm", center=0, ax=ax)
# ax.set_title("Leitungsflüsse über Zeit")
# ax.set_xlabel("Zeit")
# ax.set_ylabel("Leitungen")
# plt.show()

# #%% Auslastung der lines lines_t.p/n.lines_t.s_nom

# fig, ax = plt.subplots(figsize=(12, 8))
# sns.heatmap(network.lines_t.p1.T / network.lines.s_nom.T, cmap="coolwarm", center=0, ax=ax)
# ax.set_title("Auslastung der Leitungen über Zeit")
# ax.set_xlabel("Zeit")
# ax.set_ylabel("Leitungen")
# plt.show()


#%%

# calculate line loading in snapshot s
line_loading = abs(network.lines_t.p0.div(network.lines['s_max_pu'] * network.lines['s_nom']))

# line loading plot
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.AlbersEqualArea()})
s = network.snapshots[1]
links_cmap='Reds'
bus_cmap = 'Blues'
dem_idx = network.generators.filter(like='demand', axis=0).index
dem_per_bus = (network.generators_t.p.loc[s, dem_idx]
                .groupby(network.generators
                        .filter(like='demand', axis=0).bus)
                .sum())

p_by_carrier_bus = (network.generators_t.p.loc[s]
                .T
                .groupby([network.generators.bus, network.generators.carrier])
                .sum())

link_v_min = 0
link_v_max = line_loading.loc[s].max()
 
norm = plt.Normalize(vmin=link_v_min, vmax=link_v_max)


# bus_size_set definieren
# alle buses mit carrier external_supercharger = 0 setzen
bus_size_set = pd.Series(index=network.buses.index, dtype=object)

# Alle Bussen mit Carrier "external_supercharger" auf 0 setzen
for bus in network.buses.index:
    if "external_supercharger" in bus:
        bus_size_set.at[bus] = 0
    else:
        # pick the scalar value for this specific bus
        if bus in p_by_carrier_bus.index:
            bus_size_set.at[bus] = -1 * p_by_carrier_bus[bus][0] / 1e5
        else:
            bus_size_set.at[bus] = 0  # fallback if missing

network.plot(ax=ax,
        bus_sizes = bus_size_set,
        link_cmap=links_cmap,
        line_norm=norm,
        line_colors=line_loading.loc[s],
        title='line loading plot at single snapshot',
)
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=links_cmap), ax=ax, label='line loading (per unit)', orientation='vertical',
             pad=0.01, fraction=0.03, location='right')


#%%
fig, ax = plt.subplots()
for l in line_loading.columns:
    line_loading.loc[:, l].sort_values(ascending=False).reset_index(drop=True).plot(ax=ax, alpha=0.1, color='gray')
ax.set_xlabel('sorted snapshots')
ax.set_ylabel('line loading (per unit)')
ax.grid()
#%%
'''

Falsche Carrier!!!!

'''




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
    # features.plot(ax=ax, facecolor="khaki", edgecolor="black", alpha=0.1)
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
