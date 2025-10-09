import main_functions as mf
import input_data as ds

import geopandas as gpd
import pandas as pd



#%%
def GridCreator(top: float, bottom: float, left: float, right: float, steps=5, technologies=['solar', 'E_car', 'HP']) -> tuple:
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
        steps: Number of steps to execute (1-6), default=5 (up to step 5)
        
    Returns:
        grid_1: PyPSA grid
        buses_df: DataFrame containing buses and related data
        bbox_1: Bounding box of the selected area
    '''

    # STEP 0
    path = ds.save_data()

    # STEP 1
    #Grid creation
    bbox = [left, bottom, right, top]
    grid, bbox = mf.ding0_grid(bbox, path)
    # Initialize DataFrame for return
    buses_df = pd.DataFrame()
    area = gpd.GeoDataFrame()
    features = gpd.GeoDataFrame()

    # STEP 2
    if steps >1:
        # Load data
        # OSM data
        buffer = 0.0002  # corresponds to approximately 20 m
        buses_df, area, features = mf.osm_data(grid, bbox, buffer)
        # Federal state data
        gpd_federal_state = gpd.read_file(f"{path}/input/georef-germany-postleitzahl.geojson")
        buses_df = mf.data_assignment(buses_df, gpd_federal_state, path)
    # STEP 3
    if steps >2:
        # Assign technologies
        # Define technologies
        gcp = technologies # gcp = generation and consumption plants
        # Assign technologies
        buses_df, factor_bbox = mf.gcp_assignment(buses_df, gcp, path)
        buses_df = mf.appartments_assignment(buses_df)
        # Add technology data to buses_df
        buses_df = mf.gcp_fill(buses_df, gcp, factor_bbox, path)
    # STEP 4
    if steps >3:
        # Add time series
        grid = mf.loads_assignment(grid, buses_df, bbox, path)


    # STEP 5
    if steps >4:
        # Prepare grid for pysa.optimize()
        grid = mf.pypsa_preparation(grid)

    return grid, buses_df, bbox, area, features

#%%

# top =  54.48594882134629 # # Upper latitude
# bottom = 54.47265521486088 # Lower latitude
# left =  11.044533685584453    # Right longitude
# right =  11.084893505520695  # Left longitude
'''
Waldmünchen
'''
# top =  49.374518600877046 # # Upper latitude
# bottom = 49.36971937206515 # Lower latitude
# left =  12.697361279468392   # Right longitude
# right =  12.708888681047798  # Left longitude
'''
2 Nodes
'''
top =  49.3727 # # Upper latitude
bottom = 49.372485 # Lower latitude
left =  12.703688   # Right longitude
right =  12.704 # Left longitude

grid, buses, bbox, area, features = GridCreator(top, bottom, left, right, steps=5)
# %%
# STEP 6
# .optimize()


# # Fix Capacity
# grid_1.optimize.fix_optimal_capacities()

# # Set snapshots für Optimierung
# start_time = pd.Timestamp("2023-01-01 00:00:00")
# end_time = pd.Timestamp("2023-01-07 23:00:00")
# snapshots = pd.date_range(start=start_time, end=end_time, freq='h')
# grid_1.set_snapshots(snapshots)

#
# grid_1.optimize()