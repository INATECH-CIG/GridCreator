import numpy as np


import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString


import geopandas as gpd
from shapely.geometry import Point, LineString
import cartopy.crs as ccrs
import osmnx as ox

def zuordnung (ding0):
    ding0.buses["dist_osm_ding0_meter"] = np.sqrt(
        (ding0.buses["x"] - ding0.buses["osm_lon"])**2 +
        (ding0.buses["y"] - ding0.buses["osm_lat"])**2
    ) * 111_000  # grobe Umrechnung in Meter (1° ≈ 111 km)

    ding0.buses["dist_osm_ding0_meter"].describe()
    return ding0



def plot_zuordnung(ding0):
    # GeoDataFrame der Busse
    bus_gdf = gpd.GeoDataFrame(
        ding0.buses,
        geometry=gpd.points_from_xy(ding0.buses["x"], ding0.buses["y"]),
        crs="EPSG:4326"  # falls das deine Koordinatenprojektion ist
    )

    # GeoDataFrame der zugeordneten OSM-Nodes
    osm_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(ding0.buses["osm_lon"], ding0.buses["osm_lat"]),
        crs="EPSG:4326"
    )

    # Linien von Bus zu Node zur Prüfung
    lines = [
        LineString([bus, osm]) for bus, osm in zip(bus_gdf.geometry, osm_points.geometry)
    ]
    lines_gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    bus_gdf.plot(ax=ax, color="blue", label="Bus")
    osm_points.plot(ax=ax, color="red", label="OSM Node", alpha=0.7)
    lines_gdf.plot(ax=ax, color="gray", linewidth=0.5, alpha=0.5)

    plt.legend()
    plt.title("Zuordnung: Bus → nächster OSM-Node")
    plt.show()

    return





def plot_zuordnung_karte(grid_2, area, features):
    # GeoDataFrame für Busse (wenn nicht schon vorhanden)
    bus_gdf = gpd.GeoDataFrame(
        grid_2.buses,
        geometry=gpd.points_from_xy(grid_2.buses['x'], grid_2.buses['y']),
        crs="EPSG:4326"
    )

    # GeoDataFrame für zugeordnete OSM-Nodes
    osm_nodes = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_2.buses['osm_lon'], grid_2.buses['osm_lat']),
        crs="EPSG:4326"
    )

    # Verbindungslinien: Bus → OSM-Node
    lines = [
        LineString([bus, node]) if node.is_valid else None
        for bus, node in zip(bus_gdf.geometry, osm_nodes.geometry)
    ]
    line_gdf = gpd.GeoDataFrame(geometry=[line for line in lines if line], crs="EPSG:4326")

    #%% Plot-Vorbereitung mit Basiskarte
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    # Hintergrund (OSM + Features)
    ox.plot_graph(area, ax=ax, show=False, close=False)
    features.plot(ax=ax, facecolor="khaki", edgecolor="black", alpha=0.7)

    # Netz anzeigen
    grid_2.plot(ax=ax, bus_sizes=1 / 2e9, line_widths=0.5)

    # Busse (rot)
    bus_gdf.plot(ax=ax, color='red', markersize=5, label='Busse', zorder=3)

    # Generator-Busse (grün)
    generator_buses = grid_2.transformers['bus1'].unique()
    generator_coords = grid_2.buses.loc[generator_buses][['x', 'y']]
    ax.scatter(
        generator_coords['x'],
        generator_coords['y'],
        color='green',
        s=10,
        label='Generatoren',
        zorder=4
    )

    # Zugeordnete OSM-Nodes (blau)
    osm_nodes.plot(ax=ax, color='blue', markersize=5, label='OSM-Nodes', zorder=2)

    # Linien zwischen Bus und OSM-Node (grau)
    line_gdf.plot(ax=ax, color='black', linewidth=0.5, alpha=0.7, zorder=1)

    ax.legend(loc="upper right")
    plt.title("Visuelle Prüfung: Bus → OSM-Zuordnung mit Karte")
    plt.show()

    return