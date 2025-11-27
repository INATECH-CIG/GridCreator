#%%
import main_functions as mf
import input_data as data
import ding0_grid_generator
import os

import geopandas as gpd
import pandas as pd

import pypsa

'''
Main module for the GridCreator tool.
Handles the overall workflow including data preparation, grid creation, technology assignment, load profile generation.
'''

#%%
def GridCreator(top: float,
                bottom: float,
                left: float,
                right: float,
                scenario: str,
                steps=[1,2,3,4,5],
                technologies=['solar', 'E_car', 'HP'],
                load_method: int = 0,
                buses_df = pd.DataFrame(),
                grid = pypsa.Network()) -> tuple:
    '''
    Main function for GridCreator  
    Steps:
    0. Save input data (input_data.py)
    1. Create the grid (ding0_grid_generator.py)
    2. Load OSM data and assign federal state information (osm_data.py, daten_zuordnung.py)
    3. Assign technologies (technik_zuordnen.py, technik_fill.py, wohnungen_zuordnen.py)
    4. Assign load profiles (loads_zuordnen.py)
    5. Prepare the grid for PyPSA (pypsa_vorbereiten.py)
    6. Optimize the grid (.optimize())
    
    Args:
        top: Upper latitude
        bottom: Lower latitude
        left: Left longitude
        right: Right longitude
        steps: List of steps to execute (1-5)
        technologies: List of technologies to consider (e.g., ['solar', 'E_car', 'HP'])
        load_method: Method for load profile generation (0: Creation of 10 individual profiles for each household type, random assignment of profiles to each household type
                                                         1: Creation of individual profiles for each household)
        buses_df: DataFrame containing buses and related data (optional, default is empty DataFrame)
                  Ability to provide pre-populated buses_df to skip step 2 or manipulated data after executing step 2
        
    Returns:
        grid_1: PyPSA grid
        buses_df: DataFrame containing buses and related data (created/modified in steps 2-4)
        bbox_1: Bounding box of the selected area
        area: GeoDataFrame containing OSM area data (created in step 2)
        features: GeoDataFrame containing OSM feature data (created in step 2)
    '''

    # STEP 0
    data.save_data()
    input_path = os.path.join(os.getcwd(), 'input')
    output_path = os.path.join(os.getcwd(), 'output', scenario)

    # Define bounding box
    bbox = [left, bottom, right, top]

    # Initialize empty variables
    area = gpd.GeoDataFrame()
    features = gpd.GeoDataFrame()

    # STEP 1
    if 1 in steps:        
        #Grid creation
        grid, bbox = mf.ding0_grid(bbox, input_path)
        grid.name = scenario
        # Check if pypsa network is empty
        if grid.buses.empty:
            print("Das erzeugte Netz ist leer. Bitte überprüfen Sie die Eingabekoordinaten.")
            return grid, buses_df, bbox, area, features
        
        # Return if only Step 1 is selected
        if steps[-1] == 1:
            return grid, buses_df, bbox, area, features

    # STEP 2
    if 2 in steps:
        # Load data
        # OSM data
        buffer = 0.0002  # corresponds to approximately 20 m
        if 1 not in steps:
            # step 1 might have been executed before, so we need to get the adjusted bbox
            # bbox = [left, bottom, right, top]
            bbox_str = list(pd.read_csv(os.path.join(output_path, 'step_1', 'bbox.csv'), header=1))
            bbox = [float(i) for i in bbox_str]
            # grid might also exist already
            grid = pypsa.Network(os.path.join(output_path, 'step_1', 'grid.nc'))

        buses_df, area, features = mf.osm_data(grid, bbox, buffer)
        # Federal state data
        buses_df = mf.data_assignment(buses_df, input_path)

        # Return if Step 2 is the last selected step
        if steps[-1] == 2:    
            return grid, buses_df, bbox, area, features

    # STEP 3
    if 3 in steps:
        # step 2 might have been executed before, so we need to get the existing buses_df
        if 2 not in steps:
            buses_df = pd.read_csv(os.path.join(output_path, 'step_5', 'buses.csv'), index_col=0)
        # Define technologies
        gcp = technologies # gcp = generation and consumption plants
        # Assign technologies
        buses_df, factor_bbox = mf.gcp_assignment(buses_df, gcp, input_path)
        buses_df = mf.appartments_assignment(buses_df)
        # Add technology data to buses_df
        buses_df = mf.gcp_fill(buses_df, gcp, factor_bbox, input_path)

        # Return if Step 3 is the last selected step
        if steps[-1] == 3:
            return grid, buses_df, bbox, area, features
        
    
    # STEP 4
    if 4 in steps:
        # grid might exist already, so we need to load it
        if 1 not in steps:
            try:
                grid = pypsa.Network(os.path.join(output_path, 'step_1', 'grid.nc'))
                bbox = [float(i) for i in list(pd.read_csv(os.path.join(output_path, 'step_1', 'bbox.csv'), header=1))]
            except:
                try:
                    grid = pypsa.Network(os.path.join(output_path, 'step_2', 'grid.nc'))
                    bbox = [float(i) for i in list(pd.read_csv(os.path.join(output_path, 'step_2', 'bbox.csv'), header=1))]
                except:
                    try:
                        grid = pypsa.Network(os.path.join(output_path, 'step_3', 'grid.nc'))
                        bbox = [float(i) for i in list(pd.read_csv(os.path.join(output_path, 'step_3', 'bbox.csv'), header=1))]
                    except:
                        raise FileNotFoundError('File "grid.nc" not found. Perform Step 1 first.')
        if not 3 in steps:
            buses_df = pd.read_csv(os.path.join(output_path, 'step_3', 'buses.csv'), index_col=0)
        
        # Add time series
        grid = mf.loads_assignment(grid, buses_df, bbox, input_path, load_method)

        # Return if Step 4 is the last selected step
        if steps[-1] == 4:
            return grid, buses_df, bbox, area, features

    # STEP 5
    if 5 in steps:
        # grid might exist already, so we need to load it
        if 4 not in steps:
            grid = pypsa.Network(os.path.join(output_path, 'step_4', 'grid.nc'))
            
        # Prepare grid for pysa.optimize()
        grid = mf.pypsa_preparation(grid)
    
    return grid, buses_df, bbox, area, features

# #%%
examples = {
    "Schallstadt_small": {"top":  47.96, # # Upper latitude
                            "bottom": 47.95, # Lower latitude
                            "left":  7.74,   # left longitude
                            "right":  7.75   # right longitude
    },
    "Schallstadt_large": {"top":  47.967835, # # Upper latitude
                            "bottom": 47.955593, # Lower latitude
                            "left":  7.735381,   # Right longitude
                            "right":  7.772647   # Left longitude
    },
    "Opfingen": {"top":  48.00798, # # Upper latitude
                            "bottom": 47.99434, # Lower latitude
                            "left":  7.70691,   # Right longitude
                            "right":  7.72483   # Left longitude
    },
    "Waldmünchen": {"top":  49.374518600877046, # # Upper latitude
                            "bottom": 49.36971937206515, # Lower latitude
                            "left":  12.697361279468392,   # Right longitude
                            "right":  12.708888681047798  # Left longitude
    },
    "2 Nodes": {"top":  49.3727, # # Upper latitude
                            "bottom": 49.372485, # Lower latitude
                            "left":  12.703688,   # Right longitude
                            "right":  12.704  # Left longitude
    }
}

# grid, buses, bbox, area, features = GridCreator(top, bottom, left, right, scenario=scenario)

# #%%
# # Anzahl an Solaranlagen im Grid:
# num_solar_gens = grid.generators[grid.generators['carrier'] == 'solar'].shape[0]
# print(f"Anzahl an Solaranlagen im Netz: {num_solar_gens}")

# # Anzahl an StorgaeUnits im Grid:
# num_storage_units = grid.storage_units.shape[0]
# print(f"Anzahl an Speichersystemen im Netz: {num_storage_units}")


# #%%
# grid.export_to_netcdf("output/grid_Schallstadt_test.nc")

# #%%
# # buses.to_csv("output/buses_Schallstadt_GER.csv")

# import networkx as nx
# nx.write_gpickle(area, "output/area_Schallstadt_test.pkl") 

# import geopandas as gpd
# features.to_file("output/features_Schallstadt_GER.gpkg", driver="GPKG")


# #%%
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import osmnx as ox

# network = grid

# fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
# # Netzplot
# network.plot(ax=ax, bus_sizes=1 / 2e9, margin=1000)
# # OSM-Daten
# ox.plot_graph(area, ax=ax, show=False, close=False)
# features.plot(ax=ax, facecolor="khaki", edgecolor="black", alpha=0.1)
# # Liste von Generator-Kategorien: (Filter-Funktion, Farbe, Label)
# gen_categories = [
#     (lambda g: g['carrier'] == 'solar', 'yellow', 'Solar Generatoren'),
#     (lambda g: g['carrier'] == 'E_car', 'green', 'E-Car Generatoren'),
#     (lambda g: g['carrier'] == 'HP', 'blue', 'HP Generatoren'),
#     (lambda g: (g['carrier'] == 'HP') & (g['carrier'] == 'solar'), 'purple', 'HP & Solar Generatoren'),
#     (lambda g: (g['carrier'] == 'HP') & (g['carrier'] == 'E_car'), 'pink', 'HP & E-Car Generatoren'),
#     (lambda g: (g['carrier'] == 'solar') & (g['carrier'] == 'E_car'), 'violet', 'Solar & E-Car Generatoren')
# ]

# for filt, color, label in gen_categories:
#     gens = network.generators[filt(network.generators)]
#     if not gens.empty:
#         buses = gens['bus'].unique()
#         coords = network.buses.loc[buses, ['x', 'y']]
#         ax.scatter(
#             coords['x'], coords['y'],
#             color=color,
#             s=20,
#             label=label,
#             zorder=5
#         )

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
# )


# ax.legend(loc='upper right')


# #%%
# grid.export_to_netcdf("input/dist_grid_for_optimize.nc")

# import networkx as nx
# nx.write_gpickle(area, "input/area_for_optimize.pkl") 

# import geopandas as gpd
# features.to_file("input/features_for_optimize.gpkg", driver="GPKG")

# # %%
# # STEP 6
# # .optimize()

# # # Fix Capacity
# # grid_1.optimize.fix_optimal_capacities()

# # # Set snapshots für Optimierung
# # start_time = pd.Timestamp("2023-01-01 00:00:00")
# # end_time = pd.Timestamp("2023-01-07 23:00:00")
# # snapshots = pd.date_range(start=start_time, end=end_time, freq='h')
# # grid_1.set_snapshots(snapshots)

# #
# # grid_1.optimize()
# %%
scenario = "Opfingen"
steps = [1,2,3,4,5]

if __name__ == "__main__":
    
    top = examples[scenario]["top"]
    bottom = examples[scenario]["bottom"]
    left = examples[scenario]["left"]
    right = examples[scenario]["right"]

    grid, buses, bbox, area, features = GridCreator(top,
                                                    bottom,
                                                    left,
                                                    right,
                                                    steps=steps,
                                                    scenario=scenario,
                                                    )

    ding0_grid_generator.save_output_data(grid,
                                           buses,
                                           bbox,
                                           area,
                                           features,
                                           scenario=scenario,
                                           steps=steps,
                                           path='output')
# %%
import importlib
import basic_plotting  
importlib.reload(basic_plotting)

save_plots = True
if __name__ == "__main__":
    grid = pypsa.Network(os.path.join('output', scenario, 'step_5', 'grid.nc'))
    area = gpd.read_file(os.path.join('output', scenario, 'step_5', 'area.gpkg'))
    features = gpd.read_file(os.path.join('output', scenario, 'step_5', 'features.gpkg'))
    buses = pd.read_csv(os.path.join('output', scenario, 'step_5', 'buses.csv'), index_col=0)
    zensus_path = os.path.join(os.getcwd(), 'input', 'zensus_daten', 'Zensus2022_Durchschn_Nettokaltmiete_Anzahl_der_Wohnungen_100m-Gitter.csv')

    fig, ax = basic_plotting.plot_step1(grid, figsize=(10,10), legend_loc='upper left', bool_gridlinelabels=True)
    if save_plots:
        fig.savefig(os.path.join('output', scenario, 'step1_grid.png'), dpi=300)
        fig.savefig(os.path.join('output', scenario, 'step1_grid.pdf'))

    zensus_feature = "durchschnMieteQM"
    zensus_feature_nicename = r"Average Rent in €/$m^2$"
    fig, ax = basic_plotting.plot_step2(grid, features, buses, zensus_path, zensus_feature=zensus_feature,
                                        zensus_feature_nicename=zensus_feature_nicename, figsize=(10,10), legend_loc='upper left', bool_gridlinelabels=False)
    if save_plots:
        fig.savefig(os.path.join('output', scenario, f'step2_zensus_{zensus_feature}.png'), dpi=300)
        fig.savefig(os.path.join('output', scenario, f'step2_zensus_{zensus_feature}.pdf'))

    fig, ax = basic_plotting.plot_step3(grid, features, buses, zensus_path, zensus_feature=zensus_feature,
                                        zensus_feature_nicename=zensus_feature_nicename, figsize=(10,10), bool_legend=True, legend_loc='upper left',
                                        plot_trafos=True)
    if save_plots:
        fig.savefig(os.path.join('output', scenario, f'step3_technologies_{zensus_feature}.png'), dpi=300)
        fig.savefig(os.path.join('output', scenario, f'step3_technologies_{zensus_feature}.pdf'))

    fig, ax = basic_plotting.plot_step4(grid)
    if save_plots:
        fig.savefig(os.path.join('output', scenario, f'step4_ts.png'), dpi=300)
        fig.savefig(os.path.join('output', scenario, f'step4_ts.pdf'))

# %%
