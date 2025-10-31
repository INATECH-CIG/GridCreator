'''
basic plotting
'''
import matplotlib.pyplot as plt
import pypsa
import cartopy.crs as ccrs
import osmnx as ox

def plot_grid(grid,
              area,
              features):
    """
    plots the pypsa grid along with OSM data and features.
    Args:
        grid (pypsa.Network): The power system network containing buses and generators.
        area (gpd.GeoDataFrame): GeoDataFrame containing OSM area data with geometry.
        features (gpd.GeoDataFrame): GeoDataFrame containing OSM features with geometry.
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    # pypsa grid
    grid.plot(ax=ax,
              bus_sizes=1 / 2e9,
              margin=1)
    # OSM data
    # ox.plot_graph(area, ax=ax, show=False, close=False)
    area.plot(ax=ax, facecolor="lightgrey", edgecolor="black", alpha=0.5)
    features.plot(ax=ax, facecolor="khaki", edgecolor="black", alpha=0.1)
    plt.legend()
    plt.title('Grid Overview')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
   
