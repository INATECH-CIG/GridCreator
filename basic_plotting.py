'''
basic plotting
'''
import matplotlib.pyplot as plt
import pypsa
import cartopy.crs as ccrs
import osmnx as ox
import pandas as pd
import polars as pl
import geopandas as gpd
import random
from shapely.geometry import Point, Polygon

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


def plot_step1(grid,
              area,
              features):
    """
    plots the pypsa grid along with OSM data and features.
    Args:
        grid (pypsa.Network): The power system network containing buses and generators.
        area (gpd.GeoDataFrame): GeoDataFrame containing OSM area data with geometry.
        features (gpd.GeoDataFrame): GeoDataFrame containing OSM features with geometry.
    """
    fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'projection': ccrs.PlateCarree()})
    # pypsa grid
    #grid.plot(bus_sizes=1 / 2e9)
    grid.plot(ax=ax, bus_sizes=1 / 2e9, margin=0)
    # OSM data
    # ox.plot_graph(area, ax=ax, show=False, close=False)
    #area.plot(ax=ax, facecolor="lightgrey", edgecolor="black", alpha=0.5)
    #features.plot(ax=ax, facecolor="khaki", edgecolor="black", alpha=0.1)
    plt.legend()
    plt.title('Grid Overview')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


def plot_step2(grid,
              area,
              features):
    """
    plots the pypsa grid along with OSM data and features.
    Args:
        grid (pypsa.Network): The power system network containing buses and generators.
        area (gpd.GeoDataFrame): GeoDataFrame containing OSM area data with geometry.
        features (gpd.GeoDataFrame): GeoDataFrame containing OSM features with geometry.
    """
    fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'projection': ccrs.PlateCarree()})
    # pypsa grid
    grid.plot(ax=ax, bus_sizes=1 / 2e9, margin=0)
    
    
    # OSM data
    features_polygons = features[features.geometry.type.isin(["Polygon", "MultiPolygon"])]
    features_polygons.plot(ax=ax, facecolor="khaki", edgecolor="black", alpha=0.4, transform=ccrs.PlateCarree())
    plt.legend()
    plt.title('Grid Overview')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()




def plot_step3(grid: pypsa.Network, area: gpd.GeoDataFrame, features: gpd.GeoDataFrame, buses: pd.DataFrame, zensus: str)->None:
    '''
    plots the pypsa grid along with OSM data and Census features.
    Args:
        grid (pypsa.Network): The power system grid containing buses and generators.
        area (gpd.GeoDataFrame): GeoDataFrame containing OSM area data with geometry.
        features (gpd.GeoDataFrame): GeoDataFrame containing OSM features with geometry.
        buses (pd.DataFrame): DataFrame containing bus information with 'GITTER_ID_100m'.
        zensus (str): Path to the census data CSV file.
    '''

    # Loading Census Data and preparing Polygons
    columns = buses['GITTER_ID_100m'].copy()
    zensus = (pl.scan_csv(zensus, separator=";")
                .filter(pl.col("GITTER_ID_100m").is_in(columns))
                .select("GITTER_ID_100m",
                        "x_mp_100m",
                        "y_mp_100m",
                        "Gas",
                        ).collect()
            )
    zensus = zensus.to_pandas()

    # Manche CSVs enthalten Unicode-Striche (– statt -)
    zensus["Gas"] = (zensus["Gas"].astype(str).str.replace(r"[^\d\.-]", "",regex=True).replace("", "0").astype(float))

    # Build Polygons from Census Paoints
    def build_cell(row, half=50):
        x = row["x_mp_100m"]
        y = row["y_mp_100m"]
        return Polygon([
            (x - half, y - half),
            (x - half, y + half),
            (x + half, y + half),
            (x + half, y - half)
        ])

    # Creating GeoDataFrame for Census Data
    zensus["geometry"] = zensus.apply(build_cell, axis=1)
    zensus_gdf = gpd.GeoDataFrame(zensus, geometry="geometry", crs="EPSG:3035")
    # to EPSG:4326
    zensus_gdf = zensus_gdf.to_crs("EPSG:4326")
    zensus_voronoi_gdf = zensus_gdf[["geometry", "Gas"]].copy()

    # # Creating GeoDataFrame for Census Data using Points

    # def build_cell(geometries):
    #     polygons = []
    #     # in degrees, 100m ~ 0.0009°
    #     half = 0.00045
    #     for point in geometries:
    #         x = point.x
    #         y = point.y
    #         polygon = Polygon([
    #             (x - half, y - half),
    #             (x - half, y + half),
    #             (x + half, y + half),
    #             (x + half, y - half)
    #         ])
    #         polygons.append(polygon)
    #     return polygons

    # zensus["geometry"] = zensus.apply(lambda row: Point(row["x_mp_100m"], row["y_mp_100m"]), axis=1)
    # zensus_gdf = gpd.GeoDataFrame(zensus, geometry="geometry", crs="EPSG:3035")
    # # to EPSG:4326
    # zensus_gdf = zensus_gdf.to_crs("EPSG:4326")
    # zensus_voronoi_gdf = zensus_gdf[["geometry", "Gas"]].copy()
    # # Building Polygons around Points
    # zensus_voronoi_gdf["geometry"] = build_cell(zensus_voronoi_gdf["geometry"])
    print(zensus_voronoi_gdf['geometry'].head())

    # Filtering generators and storages
    gen_buses = {
        'solar': set(grid.generators[grid.generators.index.to_series().str.contains('_solar')]['bus']),
        'HP': set(grid.generators[grid.generators.index.to_series().str.contains('_HP')]['bus'])
                }
    storage_buses = {
        'E_Auto': set(grid.buses[grid.buses.index.isin(grid.links['bus0'])].index)
                }
    # Defining categories: bus sets, color, label
    gen_categories = [
        (gen_buses['solar'], 'yellow', 'Solar Generatoren'),
        (storage_buses['E_Auto'], 'green', 'E-Car Generatoren'),
        (gen_buses['HP'], 'blue', 'HP Generatoren'),
        (gen_buses['HP'] & gen_buses['solar'], 'purple', 'HP & Solar Generatoren'),
        (gen_buses['HP'] & storage_buses['E_Auto'], 'pink', 'HP & E-Car Generatoren'),
        (gen_buses['solar'] & storage_buses['E_Auto'], 'violet', 'Solar & E-Car Generatoren')
                        ]
    
    # Conventional Generators
    gen_buses['conventional'] = set(grid.generators[grid.generators.index.to_series().str.contains('_conventional')]['bus'])

    # Defining bus sizes
    bus_size_set = pd.Series(index=grid.buses.index, dtype=object)

    # Setting all buses with carrier "external_supercharger" to 0
    for bus in grid.buses.index:
        if "external_supercharger" in bus:
            bus_size_set.at[bus] = 0
        else:
            bus_size_set.at[bus] = 1 / 2e9  # Standardgröße



    # Plotting the network
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # pypsa plot
    #grid = grid.to_crs("EPSG:25832")
    grid.plot(ax=ax, bus_sizes=bus_size_set, margin=0)

    # Plot zensus['geometry']
    zensus_gdf["geometry"].plot(ax=ax, facecolor="none", edgecolor="black")

    # Census as heatmap overlay
    zensus_voronoi_gdf.plot(
                                    column="Gas",
                                    cmap="viridis",
                                    legend=False,
                                    ax=ax,
                                    alpha=0.7,
                                    edgecolor="black",
                                    linewidth=0.2,
                                    transform=ccrs.PlateCarree()
                                )


    # OSM Data
    #features_polygons = features_polygons.to_crs("EPSG:25832")
    features_polygons = features[features.geometry.type.isin(["Polygon", "MultiPolygon"])]
    features_polygons.plot(ax=ax, facecolor="khaki", edgecolor="black", alpha=0.1,
                           transform=ccrs.PlateCarree())
    
    # Plotting different generator types
    for buses, color, label in gen_categories:
        if buses:
            coords = grid.buses.loc[list(buses), ['x', 'y']]
            ax.scatter(
                coords['x'], coords['y'],
                color=color,
                s=20,
                label=label,
                zorder=5,
                transform=ccrs.PlateCarree()
            )


    # Plotting conventional generators in brown
    if gen_buses['conventional']:
        coords = grid.buses.loc[list(gen_buses['conventional']), ['x', 'y']]
        ax.scatter(
            coords['x'], coords['y'],
            color='brown',
            s=20,
            label='Conventional Generatoren',
            zorder=5,
            transform=ccrs.PlateCarree()
                    )
    # Marking transformer buses
    tra_buses = grid.transformers['bus1'].unique()
    tra_coords = grid.buses.loc[tra_buses][['x', 'y']]
    ax.scatter(
        tra_coords['x'],
        tra_coords['y'],
        color='red',
        s=10,         # Punktgröße
        label='Transformers',
        zorder=5,     # überlagert andere Layer
        transform=ccrs.PlateCarree()
    )

    # Setting the legend
    # Colorbar erzeugen
    sm = plt.cm.ScalarMappable(cmap="viridis",
                            norm=plt.Normalize(
                                vmin=zensus_voronoi_gdf["Gas"].min(),
                                vmax=zensus_voronoi_gdf["Gas"].max()
                            ))

    sm._A = []  # Trick, damit matplotlib nicht meckert
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.7)
    cbar.set_label("Gas")          
    ax.legend(loc='lower right')



'''
Der Plot geht nur nach der Optiomierung oder? Da sonst die Stores nicht genutzt sind!
'''
def plot_step4(grid,
              area,
              features):
    """
    plots the pypsa grid along with OSM data and features.
    Args:
        grid (pypsa.Network): The power system grid containing buses and generators.
        area (gpd.GeoDataFrame): GeoDataFrame containing OSM area data with geometry.
        features (gpd.GeoDataFrame): GeoDataFrame containing OSM features with geometry.
    """
    # bus random auswählen
    random_load = random.choice(grid.loads.index.tolist())
    random_bus = grid.loads.at[random_load, 'bus']
    random_gen = grid.generators[grid.generators['bus'] == random_bus].index.tolist()
    random_st = grid.stores[grid.stores['bus'] == random_bus].index.tolist()

    # plot von Load
    fig, ax = plt.subplots()
    grid.loads_t.p_set[f'{random_bus}_load_1'].plot(ax=ax, label='Load', color='blue')
    if random_gen:
        for gen in random_gen:
            (grid.generators_t.p_max_pu[gen]*grid.generators.at[gen, 'p_nom']).plot(ax=ax, label=f'Generator: {gen}', linestyle='--')
    if random_st:
        for st in random_st:
            grid.stores_t.e[st].plot(ax=ax, label=f'Storage: {st}', linestyle='--')
    plt.title(f'Load Profile at Bus: {random_bus}')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.show()
    plt.close()