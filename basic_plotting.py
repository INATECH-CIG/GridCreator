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
              bus_sizes=1 / 2e9,
              figsize=(10,10),
              plot_trafos=True,
              bool_legend=True,
              legend_loc=None,
              bool_gridlines=True,
              bool_gridlinelabels=False,
              title:str=None,
              legend_fontsize: int = 12,
              legend_markerscale: float = 1.5,
              gridlabel_size: int = 12):
    """
    plots the pypsa grid along with OSM data and features.
    Args:
        grid (pypsa.Network): The power system network containing buses and generators.
        area (gpd.GeoDataFrame): GeoDataFrame containing OSM area data with geometry.
        features (gpd.GeoDataFrame): GeoDataFrame containing OSM features with geometry.
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    # pypsa grid
    #grid.plot(bus_sizes=1 / 2e9)
    grid.plot(ax=ax, bus_sizes=bus_sizes, margin=0)
    
    if plot_trafos:
        # Marking transformer buses
        tra_buses = grid.transformers['bus1'].unique()
        tra_coords = grid.buses.loc[tra_buses][['x', 'y']]
        ax.scatter(
            tra_coords['x'],
            tra_coords['y'],
            color='red',
            s=10,    
            label='Transformers',
            zorder=5,     
            transform=ccrs.PlateCarree()
        )
    if bool_legend == True:
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cadetblue', markersize=10),
            plt.Line2D([0, 1], [0, 0], color='rosybrown', lw=2)
        ]
        labels = ['Nodes', 'Lines']
        if plot_trafos:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10))
            labels.append('Transformers')
        if legend_loc is None:
            legend_loc = 'upper right'
        plt.legend(handles=handles, labels=labels, loc=legend_loc)
        ax.legend(
            handles,
            labels,
            loc=legend_loc,
            fontsize=legend_fontsize,
            markerscale=legend_markerscale,
            title="Legend",
            title_fontsize=legend_fontsize + 1,
        )
    if title is not None:
        plt.title(title)

    # Tick‑Beschriftungen größer
    if not bool_gridlinelabels:
        ax.tick_params(bottom=False, left=False)

    if bool_gridlines:
        # Add cartopy gridlines that draw latitude/longitude tick labels and set axis labels with padding
        gl = ax.gridlines(draw_labels=bool_gridlinelabels, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        # disable labels on top/right to avoid duplication
        try:
            gl.top_labels = False
            gl.right_labels = False
        except Exception:
            # some cartopy versions use different attributes; ignore if not available
            pass

        if bool_gridlinelabels:
            gl.xlabel_style = {"size": gridlabel_size}
            gl.ylabel_style = {"size": gridlabel_size}

    plt.tight_layout()
    return fig, ax


def plot_step2_only_osm(grid,
                        features,
                        bus_sizes=1 / 2e9,
                        figsize=(10,10),
                        plot_trafos=True,
                        bool_legend=True,
                        legend_loc=None,
                        bool_gridlines=True,
                        bool_gridlinelabels=False,
                        title:str=None):
    """
    plots the pypsa grid along with OSM data and features.
    Args:
        grid (pypsa.Network): The power system network containing buses and generators.
        area (gpd.GeoDataFrame): GeoDataFrame containing OSM area data with geometry.
        features (gpd.GeoDataFrame): GeoDataFrame containing OSM features with geometry.
    """
    fig, ax = plot_step1(grid, bus_sizes, figsize, plot_trafos, bool_legend,
                         legend_loc, bool_gridlines, bool_gridlinelabels)    
    
    # OSM data
    features_polygons = features[features.geometry.type.isin(["Polygon", "MultiPolygon"])]
    features_polygons.plot(ax=ax, facecolor="khaki", edgecolor="black", alpha=0.2, transform=ccrs.PlateCarree())
    if title is not None:
        plt.title(title)
    return fig, ax


def plot_step2(grid: pypsa.Network,
               features: gpd.GeoDataFrame,
               buses: pd.DataFrame,
               zensus_path: str,
               zensus_feature: str,
               zensus_feature_nicename: str,
               title:str=None,
               bus_sizes=1 / 2e9,
               figsize=(10,10),
               plot_trafos=True,
               bool_legend=True,
               legend_loc='lower right',
               bool_gridlines=True,
               bool_gridlinelabels=False,
               cbar_labelsize: int = 20,
               cbar_ticksize: int = 15)->None:
    '''
    plots the pypsa grid along with OSM data and Census features.
    Args:
        grid (pypsa.Network): The power system grid containing buses and generators.
        area (gpd.GeoDataFrame): GeoDataFrame containing OSM area data with geometry.
        features (gpd.GeoDataFrame): GeoDataFrame containing OSM features with geometry.
        buses (pd.DataFrame): DataFrame containing bus information with 'GITTER_ID_100m'.
        zensus (str): Path to the census data CSV file.
    '''
    if zensus_feature is None:
        warn.warning("No zensus feature provided, defaulting to 'durchschnMieteQM'")
        zensus_feature = "durchschnMieteQM"
    # Loading Census Data and preparing Polygons
    ids = buses['GITTER_ID_100m'].copy()
    zensus = (pl.scan_csv(zensus_path, separator=";")
                .filter(pl.col("GITTER_ID_100m").is_in(ids))
                .select("GITTER_ID_100m",
                        "x_mp_100m",
                        "y_mp_100m",
                        zensus_feature,
                        ).collect()
            )
    zensus = zensus.to_pandas()

    zensus[zensus_feature] = (zensus[zensus_feature].astype(str).str.replace(",", ".", regex=False).replace(r"[^\d\.-]", "",regex=True).replace("", "0").astype(float))

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
    zensus_voronoi_gdf = zensus_gdf[["geometry", zensus_feature]].copy()

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
    # print(zensus_voronoi_gdf['geometry'].head())


    # Plotting the network
    fig, ax = plot_step2_only_osm(grid, features, bus_sizes, figsize, plot_trafos, bool_legend,
                         legend_loc, bool_gridlines, bool_gridlinelabels)
    
    # Plot zensus['geometry']
    zensus_gdf["geometry"].plot(ax=ax, facecolor="none", edgecolor="black")

    # Census as heatmap overlay
    zensus_voronoi_gdf.plot(
                                    column=zensus_feature,
                                    cmap="viridis",
                                    legend=False,
                                    ax=ax,
                                    alpha=0.7,
                                    edgecolor="black",
                                    linewidth=0.2,
                                    transform=ccrs.PlateCarree()
                                )
    
    # Setting the legend
    # Creating the colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis",
                            norm=plt.Normalize(
                                vmin=zensus_voronoi_gdf[zensus_feature].min(),
                                vmax=zensus_voronoi_gdf[zensus_feature].max()
                            ))

    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.7)
    cbar.set_label(zensus_feature_nicename, fontsize=cbar_labelsize)
    cbar.ax.tick_params(labelsize=cbar_ticksize)


    if title is not None:
        plt.title(title)       
        #ax.set_title('Low-voltage network of Opfingen with Generator Types and Census Feature')    
    plt.tight_layout()
    return fig, ax

        
def plot_step3(grid: pypsa.Network,
               features: gpd.GeoDataFrame,
               buses: pd.DataFrame,
               zensus_path: str,
               zensus_feature: str,
               zensus_feature_nicename: str,
               title:str=None,
               bus_sizes=1 / 2e9,
               figsize=(10,10),
               bool_legend=True,
               legend_loc='lower right',
               plot_trafos=True,
               bool_gridlines=True,
               bool_gridlinelabels=False,
               axis_labelsize: int = 14,
               tick_labelsize: int = 12,
               title_size: int = 16,
               legend_fontsize: int = 12,
               legend_markerscale: float = 1.5,
               cbar_labelsize: int = 20,
               cbar_ticksize: int = 15,
               xaxis_label_rotation: int = 0,
               xtick_rotation: int = 0)->None:
    '''
    plots the pypsa grid along with OSM data and Census features.
    Args:
        grid (pypsa.Network): The power system grid containing buses and generators.
        area (gpd.GeoDataFrame): GeoDataFrame containing OSM area data with geometry.
        features (gpd.GeoDataFrame): GeoDataFrame containing OSM features with geometry.
        buses (pd.DataFrame): DataFrame containing bus information with 'GITTER_ID_100m'.
        zensus (str): Path to the census data CSV file.
    '''

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
        (gen_buses['solar'], 'yellow', 'PV'),
        (storage_buses['E_Auto'], 'green', 'EV'),
        (gen_buses['HP'], 'blue', 'HP'),
        (gen_buses['HP'] & gen_buses['solar'], 'purple', 'HP & PV'),
        (gen_buses['HP'] & storage_buses['E_Auto'], 'pink', 'HP & EV'),
        (gen_buses['solar'] & storage_buses['E_Auto'], 'violet', 'PV & EV'),
        (gen_buses['HP'] & gen_buses['solar'] & storage_buses['E_Auto'], 'orange', 'HP, PV & EV')
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
    fig, ax = plot_step2(grid, features, buses, zensus_path, zensus_feature=zensus_feature,
                         zensus_feature_nicename=zensus_feature_nicename, bus_sizes=bus_size_set,
                         figsize=figsize, plot_trafos=plot_trafos, bool_legend=False,
                         bool_gridlines=bool_gridlines, bool_gridlinelabels=bool_gridlinelabels)
    
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
    if title is not None:
        plt.title(title)     
    if bool_legend:
        handles, labels = ax.get_legend_handles_labels()
        if legend_loc is None:
            legend_loc = 'upper right'
        plt.legend(handles=handles, labels=labels, loc=legend_loc, fontsize=legend_fontsize,)

        
    plt.tight_layout()
    return fig, ax



'''
Der Plot geht nur nach der Optiomierung oder? Da sonst die Stores nicht genutzt sind!
'''
def plot_step4(grid,
               bus = None,
               figsize=(10,10),
               begin:int=0,
               gridlines=True,
              title:str=None,
              ylabel='Power (kW)',
              xlabel='Time'):
    """
    plots the pypsa grid along with OSM data and features.
    Args:
        grid (pypsa.Network): The power system grid containing buses and generators.
        area (gpd.GeoDataFrame): GeoDataFrame containing OSM area data with geometry.
        features (gpd.GeoDataFrame): GeoDataFrame containing OSM features with geometry.
    """
    if bus is None:
        # grid.generators nach bus  gruppieren
        grupped = grid.generators.groupby('bus')
        # als dict speichern
        gen_dict = {bus: grupped.get_group(bus) for bus in grupped.groups}
        # keys mti mehr als 1 generator
        keys_multiple_gens = [key for key, value in gen_dict.items() if len(value) > 1]

        bus = keys_multiple_gens[0]

    # for one week
    end = begin + 24*7

    solar = f'{bus}_solar'
    hp = f'{bus}_HP'
    # plot von Load
    fig, ax = plt.subplots(figsize=figsize)
    (grid.loads_t.p_set[f'{bus}_load_1'].iloc[begin:end]*1e6).plot(ax=ax, label='Load', color='green')
    # Plot of only one week
    if solar in grid.generators.index:
        (grid.generators_t.p_max_pu[solar][begin:end]*grid.generators.at[solar, 'p_nom']*1e6).plot(ax=ax, label=f'Solar Generator', linestyle='-', color='orange')
    if hp in grid.generators.index:
        (grid.generators_t.p_max_pu[hp][begin:end]*grid.generators.at[hp, 'p_nom']*1e3).plot(ax=ax, label=f'Heat Pump', linestyle='-', color='blue')
    if gridlines:
        ax.grid()
    if title is not None:
        plt.title(title)
    
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    return fig, ax