
import functions as func
import ding0_grid_generator as ding0
import data_combination as dc
from dicts import agg_dict
import creating_demand_and_load as demand_load

import pandas as pd
import numpy as np

# import for type hints
from typing import List
import pypsa
import geopandas as gpd


#%%
def ding0_grid(bbox: list[float], path: str) -> tuple[pypsa.Network, list[float]]:
    """
    Creates a grid object based on the given bounding box coordinates and saves it as a NetCDF file.

    Args:
        bbox (list): A list of bounding box coordinates in the format [left, bottom, right, top].
        grids_dir (str): Path to the directory where the grid should be saved.
        output_file_grid (str): Name of the output file for the grid in NetCDF format.

    Returns:
        tuple: A tuple containing the grid object and the bounding box.
    """
    # Directory where the grids are stored
    grids_dir = f"{path}/input/grids"
    # Load the grid
    grid = ding0.load_grid(bbox, grids_dir)

    # Compute a new bounding box based on all contained buses
    bbox_new = func.compute_bbox_from_buses(grid)

    # Load grid for the expanded bounding box
    grid = ding0.load_grid(bbox_new, grids_dir)

    return grid, bbox_new


def osm_data(network: pypsa.Network, bbox_new: list[float], buffer: float) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Retrieves OSM data for the given bounding box and returns it along with the network buses.

    Args:
        network: The network object containing buses and lines.
        bbox_new: The expanded bounding box for the OSM query [left, bottom, right, top].
        buffer: The buffer distance to expand the bounding box.

    Returns:
        tuple: A tuple containing:
            - buses_df (pd.DataFrame): DataFrame with combined bus and OSM data.
            - area (gpd.GeoDataFrame): The OSM area geometry.
            - area_features (gpd.GeoDataFrame): Detailed OSM features.
    """

    left, bottom, right, top = bbox_new
    bbox_osm = (left - buffer, bottom - buffer, right + buffer, top + buffer)
    # osm Data abrufen
    Area, Area_features = func.get_osm_data(bbox_osm)
    # Reset index for feature data
    Area_features_df = Area_features.reset_index()

    # Combine network and OSM data
    buses_df = dc.data_combination(network, Area_features_df)

    return buses_df, Area, Area_features


def data_assignment(buses: pd.DataFrame, bundesland_data: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Assigns federal state (Bundesland) and census data to the buses in the network.

    Args:
        buses: DataFrame containing the network buses.
        bundesland_data: DataFrame with federal state (Bundesland) information.
        path: Base path to the input data directory containing census files.
        
    Returns:
        buses: Updated DataFrame with assigned federal state and census data.
    """
    zensus_dir = f"{path}/input/zensus_daten"
    # Assign federal state to each bus
    buses = func.federal_state(buses, bundesland_data)

    # Assign zensus IDs
    buses = func.zensus_ID(buses, zensus_dir)

    # Save old index
    buses = buses.reset_index() 

    buses = func.load_zensus(buses, zensus_dir)

    # Restore old index
    buses.set_index("Bus", inplace=True)

    return buses



def gcp_assignment(buses: pd.DataFrame, gcp_list: list[str], path: str) -> tuple[pd.DataFrame, list[float]]:
    """
    Assigns different gcp (generation and consumption plants) to buses in the network based on zensus and population data.

    Args:
        buses (pd.DataFrame): DataFrame containing bus information.
        gcp_list (list[str]): List of gcp to assign (e.g., ["solar", "E_car", "HP"]).
        path (str): Path to the input data folder containing CSV files for factors and population data.

    Returns:
        tuple: A tuple consisting of:
            - Updated buses DataFrame with assigned technology factors.
            - List of calculated factors for each technology.
    """

    # Load input CSVs
    file_Faktoren = f"{path}/input/Faktoren.csv"
    file_solar = f"{path}/input/Bev_data_solar.csv"
    file_ecar = f"{path}/input/Bev_data_ecar.csv"
    file_hp = f"{path}/input/Bev_data_hp.csv"
    
    gcp_factors = pd.read_csv(file_Faktoren)
    gcp_factors = gcp_factors.set_index("Technik")

    # Solar population data
    data_solar = pd.read_csv(file_solar, sep=",")
    data_solar["PLZ"] = data_solar["PLZ"].astype(int).astype(str).str.zfill(5)
    data_solar.set_index("PLZ", inplace=True)

    # E-car population data
    data_ecar = pd.read_csv(file_ecar, sep=",")
    data_ecar["Schluessel_Zulbz"] = data_ecar["Schluessel_Zulbz"].astype(int).astype(str).str.zfill(5)
    data_ecar.set_index("Schluessel_Zulbz", inplace=True)

    # Heat pump (HP) data
    data_hp = pd.read_csv(file_hp, sep=",")
    data_hp.set_index("GEN", inplace=True)

    # Prepare census columns
    buses_zensus = buses[[col for col in buses.columns if col.startswith("Zensus")]].copy()
    buses_zensus.drop(columns=["Zensus_Einwohner"], inplace=True)
    buses_zensus = (buses_zensus.astype(str).replace(",", ".", regex=True).apply(pd.to_numeric, errors="coerce").fillna(0.0))

    # Group by grid ID and take first row of zensus data for aggregation
    buses_zensus['GITTER_ID_100m'] = buses['GITTER_ID_100m']
    buses_zensus_grouped = buses_zensus.groupby(['GITTER_ID_100m'])

    # Compute bbox aggregation
    bbox_zensus_df = pd.DataFrame()
    for name, group in buses_zensus_grouped:
        df = group.iloc[[0]].copy()
        bbox_zensus_df = pd.concat([bbox_zensus_df, df])
    bbox_zensus = bbox_zensus_df.agg(agg_dict)

    # Remove temporary column
    buses_zensus.drop(columns=['GITTER_ID_100m'], inplace=True)

    # Initialize factor array
    factor_bbox = np.array([0.0] * len(gcp_list))
    
    # Solar assignment
    if 'solar' in gcp_list:
        buses_plz = buses['plz_code'].copy()
        technik = 'solar'
        i = gcp_list.index(technik)
        buses['Factor_' + technik], factor_bbox[i]  = func.faktoren(buses_zensus, gcp_factors, data_solar, bbox_zensus, 'solar', buses_plz)

    # E-car assignment
    if 'E_car' in gcp_list:
        buses_permission = func.permission(buses, path)
        technik = 'E_car'
        i = gcp_list.index(technik)
        buses['Factor_' + technik], factor_bbox[i]  = func.faktoren(buses_zensus, gcp_factors, data_ecar, bbox_zensus, 'E_car', buses_permission)

    # HP assignment
    if 'HP' in gcp_list:
        buses_district = buses['lan_name'].copy()
        technik = 'HP'
        i = gcp_list.index(technik)
        buses['Factor_' + technik], factor_bbox[i]  = func.faktoren(buses_zensus, gcp_factors, data_hp, bbox_zensus, 'HP', buses_district)

    return buses, factor_bbox



def appartments_assignment(buses: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns the number of apartments and residents to buses in the network based on census data.

    Args:
        buses (pd.DataFrame): DataFrame containing the network buses.

    Returns:
        pd.DataFrame: Updated DataFrame with assigned apartments and residents.
    """

    # Create new columns for housing types and resident counts
    buses['Bus_1_Wohnung'] = 0
    buses['Bus_2_Wohnungen'] = 0
    buses['Bus_3bis6_Wohnungen'] = 0
    buses['Bus_7bis12_Wohnungen'] = 0
    buses['Bus_13undmehr_Wohnungen'] = 0
    buses['Bus_1_Person'] = 0
    buses['Bus_2_Personen'] = 0
    buses['Bus_3_Personen'] = 0
    buses['Bus_4_Personen'] = 0
    buses['Bus_5_Personen'] = 0
    buses['Bus_6_Personen_und_mehr'] = 0
    buses['Haushalte'] = 0
    buses['Bewohnerinnen'] = 0

    # Fill missing values and replace special dash characters
    buses = buses.fillna(0)
    buses = buses.replace("–", 0)

    # Group by GITTER_ID_100m
    buses_grouped = buses.groupby('GITTER_ID_100m')

    # Compute apartment distribution
    for name, group in buses_grouped:

        num_nodes = len(group)
        total_apartments = sum([float(group.iloc[0]['Zensus_1_Wohnung']),
                        float(group.iloc[0]['Zensus_2_Wohnungen']),
                        float(group.iloc[0]['Zensus_3bis6_Wohnungen']),
                        float(group.iloc[0]['Zensus_7bis12_Wohnungen']),
                        float(group.iloc[0]['Zensus_13undmehr_Wohnungen'])])
        if total_apartments == 0:
            # If no data is available, distribute residents evenly among nodes
            residents_per_node = int(group.iloc[0]['Zensus_Einwohner']) / num_nodes
            for i in group.index:
                buses.at[i, 'Bewohnerinnen'] = residents_per_node
            continue
            
        else:

            """
            If apartment data exists, residents are distributed proportionally.
            Steps:
            1. Determine apartment type shares.
            2. Assign each bus an apartment type.
            3. Create a list of bus indices repeated according to apartment counts.
            4. Compute residents per household type.
            5. Distribute residents proportionally until the total population is reached.
            """

            
            group = group.replace("–", 0)
            typ1_share = float(group.iloc[0]['Zensus_1_Wohnung']) / total_apartments
            typ2_share = float(group.iloc[0]['Zensus_2_Wohnungen']) / total_apartments
            typ3_share = float(group.iloc[0]['Zensus_3bis6_Wohnungen']) / total_apartments
            typ4_share = float(group.iloc[0]['Zensus_7bis12_Wohnungen']) / total_apartments
            typ5_share = float(group.iloc[0]['Zensus_13undmehr_Wohnungen']) / total_apartments

            # Randomly assign an apartment type to each bus based on shares
            for i in group.index:
                print('i, ', i)
                random_value = np.random.rand()
                print('random_value, ', random_value)
                if random_value < typ1_share:
                    buses.at[i, 'Bus_1_Wohnung'] = 1
                elif random_value < typ1_share + typ2_share:
                    buses.at[i, 'Bus_2_Wohnungen'] = 1
                elif random_value < typ1_share + typ2_share + typ3_share:
                    buses.at[i, 'Bus_3bis6_Wohnungen'] = 1
                elif random_value < typ1_share + typ2_share + typ3_share + typ4_share:
                    buses.at[i, 'Bus_7bis12_Wohnungen'] = 1
                else:
                    buses.at[i, 'Bus_13undmehr_Wohnungen'] = 1

            # Create a list of bus indices, repeated according to number of apartments
            bus_list = []
            apartments_assigned = 0
            for bus_index in group.index:
                print('bus_index, ', bus_index)
                if buses.at[bus_index, 'Bus_1_Wohnung'] == 1:
                    print('1 Wohnung')
                    bus_list.extend([bus_index] * 1)
                    buses.at[bus_index, 'Haushalte'] = 1
                    buses.at[bus_index, 'Bewohnerinnen'] = 1
                    apartments_assigned += 1
                if buses.at[bus_index, 'Bus_2_Wohnungen'] == 1:
                    print('2 Wohnungen')
                    bus_list.extend([bus_index] * 2)
                    buses.at[bus_index, 'Haushalte'] = 2
                    buses.at[bus_index, 'Bewohnerinnen'] = 2
                    apartments_assigned += 2
                if buses.at[bus_index, 'Bus_3bis6_Wohnungen'] == 1:
                    print('3 bis 6 Wohnungen')
                    bus_list.extend([bus_index] * 3)
                    buses.at[bus_index, 'Haushalte'] = 3
                    buses.at[bus_index, 'Bewohnerinnen'] = 3
                    apartments_assigned += 3
                if buses.at[bus_index, 'Bus_7bis12_Wohnungen'] == 1:
                    print('7 bis 12 Wohnungen')
                    bus_list.extend([bus_index] * 7)
                    buses.at[bus_index, 'Haushalte'] = 7
                    buses.at[bus_index, 'Bewohnerinnen'] = 7
                    apartments_assigned += 7
                if buses.at[bus_index, 'Bus_13undmehr_Wohnungen'] == 1:
                    print('13 und mehr Wohnungen')
                    bus_list.extend([bus_index] * 13)
                    buses.at[bus_index, 'Haushalte'] = 13
                    buses.at[bus_index, 'Bewohnerinnen'] = 13
                    apartments_assigned += 13

            # Assign number of residents
            resident_count = group.iloc[0]['Zensus_Einwohner']

            person_count = sum([float(group.iloc[0]['Zensus_1_Person']),
                    float(group.iloc[0]['Zensus_2_Personen']),
                    float(group.iloc[0]['Zensus_3_Personen']),
                    float(group.iloc[0]['Zensus_4_Personen']),
                    float(group.iloc[0]['Zensus_5_Personen']),
                    float(group.iloc[0]['Zensus_6_Personen_und_mehr'])])

            if person_count == 0:
                '''
                If no data is available, residents are distributed evenly
                '''

                res_per_apartment = int(group.iloc[0]['Zensus_Einwohner']) / apartments_assigned
                for i in group.index:
                    buses.at[i, 'Bewohnerinnen'] = res_per_apartment * buses.at[i, 'Haushalte']
                continue


            # Share of residents per household type
            group = group.replace("–", 0)
            res_share_1 = float(group.iloc[0]['Zensus_1_Person'])/person_count
            res_share_2 = float(group.iloc[0]['Zensus_2_Personen'])/person_count
            res_share_3 = float(group.iloc[0]['Zensus_3_Personen'])/person_count
            res_share_4 = float(group.iloc[0]['Zensus_4_Personen'])/person_count
            res_share_5 = float(group.iloc[0]['Zensus_5_Personen'])/person_count
            res_share_6 = float(group.iloc[0]['Zensus_6_Personen_und_mehr'])/person_count

            # Distribute residents proportionally across apartments until total residents are assigned
            # Each bus gets at least one resident
            residents_distributed = apartments_assigned
            while residents_distributed < resident_count:
                if len(bus_list) == 0:
                    print("Anzahl an Einwohner, die nicht verteilt werden konnten: ", resident_count - residents_distributed)
                    break
                # Select a random index from the list
                random_index = np.random.choice(bus_list)
                # Remove random_index from the list to avoid repeated selection
                bus_list.remove(random_index)

                random_value = np.random.rand()
                if random_value < res_share_1:
                    buses.at[random_index, 'Bewohnerinnen'] += 0
                    buses.at[random_index, 'Bus_1_Person'] += 1
                    residents_distributed += 0


                elif random_value < res_share_1 + res_share_2:
                    buses.at[random_index, 'Bewohnerinnen'] += 1
                    buses.at[random_index, 'Bus_2_Personen'] += 1
                    residents_distributed += 1
                    
                elif random_value < res_share_1 + res_share_2 + res_share_3:
                    buses.at[random_index, 'Bewohnerinnen'] += 2
                    buses.at[random_index, 'Bus_3_Personen'] += 1
                    residents_distributed += 2
                    
                elif random_value < res_share_1 + res_share_2 + res_share_3 + res_share_4:
                    buses.at[random_index, 'Bewohnerinnen'] += 3
                    buses.at[random_index, 'Bus_4_Personen'] += 1
                    residents_distributed += 3
                    
                elif random_value < res_share_1 + res_share_2 + res_share_3 + res_share_4 + res_share_5:
                    buses.at[random_index, 'Bewohnerinnen'] += 4
                    buses.at[random_index, 'Bus_5_Personen'] += 1
                    residents_distributed += 4
                    
                else:
                    '''
                    Even for 6-person households, only assign 5 residents
                    '''
                    buses.at[random_index, 'Bewohnerinnen'] += 4
                    buses.at[random_index, 'Bus_6_Personen_und_mehr'] += 1
                    residents_distributed += 5

            # Adjust 1-person households to match total residents
            for bus in group.index:
                dif = int(buses.at[bus, 'Bewohnerinnen']) - (buses.at[bus, 'Bus_1_Person'] + buses.at[bus, 'Bus_2_Personen']*2 + buses.at[bus, 'Bus_3_Personen']*3 + buses.at[bus, 'Bus_4_Personen']*4 + buses.at[bus, 'Bus_5_Personen']*5 + buses.at[bus, 'Bus_6_Personen_und_mehr']*6)
                if dif > 0:
                    buses.at[bus, 'Bus_1_Person'] += dif

    # Remove columns no longer needed
    buses.drop(columns=['Bus_1_Wohnung', 'Bus_2_Wohnungen', 'Bus_3bis6_Wohnungen', 'Bus_7bis12_Wohnungen', 'Bus_13undmehr_Wohnungen'], inplace=True)
    return buses


def gcp_fill(buses: pd.DataFrame, gcp: List[str], p_total: List[float], pfad: str) -> pd.DataFrame:
    """
    Fills the grid object with gcp based on the given proportions.

    Args:
        buses (pd.DataFrame): DataFrame containing the bus data (grid nodes).
        gcp (list): List of gcps to be assigned.
        p_total (list): List of proportions for each gcps.
        path (str): Path to the input data.
        
    Returns:
        pd.DataFrame: Updated DataFrame with the assigned gcp.
    """

    # Load PV and storage data per postal code
    pv_plz = pd.read_csv(f"{pfad}/input/mastr_values_per_plz.csv", sep=",").set_index("PLZ")
    plz = int(buses['plz_code'].values[0])
    solar_power = pv_plz.loc[plz, 'Mean_Solar_Installed_Capacity_[MW]']

    # Change from MW to kW
    solar_power = solar_power * 1000

    for gcp_1, p in zip(gcp, p_total):
        buses = func.sort_gcp(buses, gcp_1, p, solar_power)

    # Determine solar orientation (e.g., south, east, west)
    buses = func.solar_orientation(buses, plz, pv_plz)

    # Assign storage according to PV ratio
    storage_pv = pv_plz.loc[plz, 'Storage_per_PV']
    buses = func.storage(buses, storage_pv)

    return buses




def loads_assignment(grid: pypsa.Network, buses: pd.DataFrame, bbox: List[float], pfad: str, env=None):
    """
    Assigns loads to the grid based on the bus data and environmental conditions.
    
    Args:
        grid (pypsa.Network): The grid object to which loads will be assigned.
        buses (pd.DataFrame): DataFrame containing the bus data.
        bbox (list): Bounding box coordinates for environmental data.
        pfad (str): Path to the input data directory.
        env: Optional environmental data object. If None, it will be created based on the bbox and path.

    Returns:
        grid (pypsa.Network): The updated grid object with assigned loads.
    """

    # Create or reuse environmental object
    if env is None:
        environment = func.env_weather(bbox, pfad)
    else:
        environment = env
    
    # Set time index for the grid
    start_time = pd.Timestamp("2023-01-01 00:00:00")
    snapshots = pd.date_range(start=start_time, periods=environment.timer.timesteps_total, freq=f"{int(environment.timer.time_discretization/60)}min")
    grid.set_snapshots(snapshots)


    # Remove all loads and storage units first
    grid.loads.drop(grid.loads.index, inplace=True)
    grid.storage_units.drop(grid.storage_units.index, inplace=True)

    # # Dictionary for adding loads
    # load_cols = {}
    # # Identify buses with electric vehicles
    # e_car_buses = buses.index[buses["Power_E_car"].notna()].tolist()
    # e_car_cols = {}

    # # Household sizes in a dictionary
    # household_types = {
    #     "Bus_1_Person": 1,
    #     "Bus_2_Personen": 2,
    #     "Bus_3_Personen": 3,
    #     "Bus_4_Personen": 4,
    #     "Bus_5_Personen": 5,
    #     "Bus_6_Personen_und_mehr": 5,
    #                     }

    # def house_and_car(household_type, persons, buses, bus, remaining_residents, load_cols, e_car_cols, e_car_buses, call_counter, grid, snapshots, environment):
    #     """
    #     Creates a household load (and possibly EV load) for a given bus.

    #     Args:
    #         household_type (str): Type of household (e.g., "Bus_2_Personen").
    #         persons (int): Number of people in the household.
    #         buses (pd.DataFrame): DataFrame containing bus and demographic data.
    #         bus (str): The current bus being processed.
    #         remaining_residents (int): Number of residents left to assign.
    #         load_cols (dict): Dictionary storing household power profiles.
    #         e_car_cols (dict): Dictionary storing electric vehicle power profiles.
    #         e_car_buses (list): List of buses that have electric vehicles.
    #         call_counter (int): Counter to ensure unique load names.
    #         grid (pypsa.Network): The network object.
    #         snapshots (pd.DatetimeIndex): Time index for power profiles.
    #         environment: Environmental data object.

    #     Returns:
    #         Tuple: Updated (buses, remaining_residents, load_cols, e_car_cols, e_car_buses, call_counter)
    #     """
    #     call_counter += 1
    #     power, occupants = demand_load.create_haus(people=persons, index=snapshots, env=environment)
    #     grid.add("Load", name=f"{bus}_load_{call_counter}", bus=bus, carrier="AC")
    #     load_cols[f"{bus}_load_{call_counter}"] = power
    #     remaining_residents -= persons

    #     # If the bus has electric vehicles, create an EV storage unit
    #     if bus in e_car_buses:
    #         e_auto_power = demand_load.create_e_car(occ = occupants, index=snapshots)
    #         grid.add("StorageUnit", name=bus + f"{bus}_E_Auto_{call_counter}", bus=bus, carrier="E_Auto")
    #         e_car_cols[f"{bus}_E_Auto_{call_counter}"] = e_auto_power
    #         if buses.loc[bus, 'Power_E_car'] == buses.loc[bus, 'Factor_E_car']:
    #             # Remove bus from EV list once fully assigned
    #             e_car_buses.remove(bus)
    #         else:
    #             buses.loc[bus, 'Power_E_car'] -= buses.loc[bus, 'Factor_E_car']
        
    #     return buses, remaining_residents, load_cols, e_car_cols, e_car_buses, call_counter

    # # Assign loads and EVs to each bus
    # for bus in buses.index:
    #     existing = grid.loads[(grid.loads['bus'] == bus)]
    #     if existing.empty:

    #         remaining_residents = int(round(buses.loc[bus, 'Bewohnerinnen']))
    #         call_counter = 0

    #         # Assign loads by household type
    #         for household_type, persons in household_types.items():
    #             n_households = int(buses.loc[bus, household_type])
    #             if n_households > 0:
    #                 for i in range(n_households):
    #                     buses, remaining_residents, load_cols, e_car_cols, e_car_buses, call_counter = house_and_car(household_type, persons, buses, bus, remaining_residents, load_cols, e_car_cols, e_car_buses, call_counter, grid, snapshots, environment)


    #         while remaining_residents > 0:
    #             persons = min(remaining_residents, 5)  # Nimm maximal 5 Personen für den Haushalt
    #             buses, remaining_residents, load_cols, e_car_cols, e_car_buses, call_counter = house_and_car(f"Rest_{persons}_Persons", persons, buses, bus, remaining_residents, load_cols, e_car_cols, e_car_buses, call_counter, grid, snapshots, environment)

    #     # else:
    #     #     print(f"Load für {bus} existiert bereits.")

    # # Combine all generated profiles into PyPSA objects
    # grid.loads_t.p_set = pd.concat([grid.loads_t.p_set, pd.DataFrame(load_cols)], axis=1)
    # grid.storage_units_t.p = pd.concat([grid.storage_units_t.p, pd.DataFrame(e_car_cols)], axis=1)

    """
    Remove all existing solar generators before adding new ones.
    """
    grid.generators.drop(grid.generators.index[grid.generators['type'] == 'solar'], inplace=True)

    # Select all buses that have solar capacity
    solar_buses = buses.index[buses["Power_solar"] != 0]

    # Ensure a minimum solar capacity of 0.25 kW
    for bus in solar_buses:
        print('Power_solar: ', buses.loc[bus, 'Power_solar'])
        if buses.loc[bus, 'Power_solar'] < 0.25:  # Minimum 0.25 kW
            print('Power_solar zu klein, setze auf Minimum 0.25 kW')
            buses.loc[bus, 'Power_solar'] = 0.25

    solar_cols = {}

    for bus in solar_buses:
        existing = grid.generators[(grid.generators['bus'] == bus)]

        beta = buses.loc[bus, 'HauptausrichtungNeigungswinkel_Anteil']
        gamma = buses.loc[bus, 'Hauptausrichtung_Anteil']


        # For east-west systems, split capacity into two PV systems
        if gamma == 'Ost-West':
            gamma_1 = 'Ost'
            gamma_2 = 'West'
            power_1 = demand_load.create_pv(peakpower=buses.loc[bus, 'Power_solar']*0.5, beta=beta, gamma=gamma_1, index=snapshots, env=environment)
            power_2 = demand_load.create_pv(peakpower=buses.loc[bus, 'Power_solar']*0.5, beta=beta, gamma=gamma_2, index=snapshots, env=environment)
            power = power_1 + power_2

        else:
            # Create standard PV generation profile
            power = demand_load.create_pv(peakpower=buses.loc[bus, 'Power_solar'], beta=beta, gamma=gamma, index=snapshots, env=environment)
        
        if existing.empty:
            # Add generator if it doesn’t exist yet

            # Maximum power
            power_max= max(power)
            
            grid.add("Generator", name=bus + "_solar", bus=bus, carrier="solar", type="solar", p_nom=power_max)
            # Normalise power profile to per unit values
            solar_cols[bus + "_solar"] = power.values/power_max

        else:
            # If already exists, just update its time series
            # Maximum power
            power_max= max(power)
            solar_cols[bus + "_solar"] = power.values/power_max

    # Append all generated PV profiles to PyPSA time series data
    if solar_cols:
        grid.generators_t.p_max_pu = pd.concat([grid.generators_t.p_max_pu, pd.DataFrame(solar_cols, index=snapshots)], axis=1)
        grid.generators_t.p_min_pu = pd.concat([grid.generators_t.p_min_pu, pd.DataFrame(solar_cols, index=snapshots)], axis=1)


    # Select all buses that have non-zero heat pump capacity
    HP_buses = buses.index[buses["Power_HP"] != 0]
    hp_cols = {}

    for bus in HP_buses:
        # Create heat pump power time series
        power = demand_load.create_hp(index=snapshots, env=environment)

        # Add heat pump generator to the grid
        grid.add("Generator", name=bus + "_HP", bus=bus, carrier="HP", type="HP")

        # Store the generated time series for later concatenation
        hp_cols[bus + "_HP"] = power.values

    # Append all generated heat pump profiles to PyPSA time series data
    if hp_cols:
        grid.generators_t.p_max_pu = pd.concat([grid.generators_t.p_max_pu, pd.DataFrame(hp_cols, index=snapshots)], axis=1)
        grid.generators_t.p_min_pu = pd.concat([grid.generators_t.p_min_pu, pd.DataFrame(hp_cols, index=snapshots)], axis=1)


    # Add commercial (Gewerbe) loads based on OSM data
    if "osm_building" in buses.columns:
        commercial_buses = buses.index[buses["osm_building"] != 0]
        commercial_cols = {}

        # Map OSM building types to internal load types
        commercial_dict = {
            'commercial': 'G0',
            'industrial': 'G0',
            'office': 'G1',
            'kiosk': 'G4',
            'retail': 'G4',
            'supermarket': 'G4',
            'warehouse': 'G0',
            'sports_hall': 'sports',
            'stadium': 'sports'
        }





        '''
        Wie viel Verbrauch?????
        '''







        for bus in commercial_buses:
            building_type = buses.loc[bus, 'osm_building']
            if building_type in commercial_dict:
                load_type = commercial_dict[building_type]
                power = demand_load.create_commercial(load_type, demand_per_year=1000, index=snapshots, env=environment)
                grid.add("Load", name=bus + "_commercial", bus=bus, carrier="commercial")
                commercial_cols[bus + "_commercial"] = power.values

        # Append commercial load profiles
        if commercial_cols:
            grid.loads_t.p_set = pd.concat([grid.loads_t.p_set, pd.DataFrame(commercial_cols, index=snapshots)], axis=1)

    # Add shop loads (OSM 'shop' attribute)
    if "osm_shop" in buses.columns:
        shops_buses = buses.index[buses["osm_shop"] != 0]
        shops_cols = {}

        for bus in shops_buses:
            # Bakeries (G5) get their own load type; others default to G4
            if buses.loc[bus, 'osm_shop'] == 'bakery':
                power = demand_load.create_commercial('G5', demand_per_year=1000, index=snapshots, env=environment)
            else:
                power = demand_load.create_commercial('G4', demand_per_year=1000, index=snapshots, env=environment)
            
            grid.add("Load", name=bus + "_Shop", bus=bus, carrier="Shop")
            shops_cols[bus + "_Shop"] = power.values
    
        # Append shop load profiles
        if shops_cols:
            grid.loads_t.p_set = pd.concat([grid.loads_t.p_set, pd.DataFrame(shops_cols, index=snapshots)], axis=1)

    return grid




def pypsa_preparation(grid: pypsa.Network) -> pypsa.Network:
    """
    Prepares a PyPSA network by setting default costs, efficiencies, and extendable options
    for generators, lines, and carriers. Additionally, adds backup generators and storage
    units at transformer buses.

    Args:
        grid (pypsa.Network): The PyPSA network to be prepared.

    Returns:
        pypsa.Network: The prepared and updated PyPSA network.
    """


    # Solar generator parameters
    grid.generators.loc[grid.generators['type'] == 'solar', 'marginal_cost'] = 100
    grid.generators.loc[grid.generators['type'] == 'solar', 'capital_cost'] = 1000
    grid.generators.loc[grid.generators['type'] == 'solar', 'efficiency'] = 0.9
    grid.generators.loc[grid.generators['type'] == 'solar', 'p_nom_extendable'] = False
    grid.generators.loc[grid.generators['type'] == 'solar', 'p_max_pu'] = 1

    # AC line parameters
    grid.lines.loc[grid.lines['carrier'] == 'AC', 'capital_cost'] = 50
    grid.lines.loc[grid.lines['carrier'] == 'AC', 's_nom_max'] = 1000
    grid.lines.loc[grid.lines['carrier'] == 'AC', 's_nom'] = 100
    grid.lines.loc[grid.lines['carrier'] == 'AC', 'r'] = 0.001
    grid.lines.loc[grid.lines['carrier'] == 'AC', 'x'] = 0.01
    grid.lines.loc[grid.lines['carrier'] == 'AC', 's_nom_extendable'] = True


    # Add common carriers
    grid.add("Carrier", "gas", co2_emissions=0, color="orange")
    grid.add("Carrier", "solar", co2_emissions=0, color="yellow")
    grid.add("Carrier", "wind", co2_emissions=0, color="cyan")
    grid.add("Carrier", "battery", co2_emissions=0, color="gray")
    grid.add("Carrier", "AC", co2_emissions=0, color="black")  # Für Busse, Lasten, Leitungen


    # Add gas generators and storage at transformers to simulate import and export from mv to lv
    for i, trafo in grid.transformers.iterrows():
        bus_mv = trafo['bus0']

        gen_name = f"Generator_am_{i}"
        grid.add("Generator",
                name=gen_name,
                bus=bus_mv,
                carrier="gas",
                p_nom=100,
                p_nom_extendable=True,
                capital_cost=500,
                marginal_cost=50,
                efficiency=0.4)

        storage_name = f"Storage_am_{i}"
        grid.add("Generator",
                name=storage_name,
                bus=bus_mv,
                carrier="battery",
                p_min = -1,
                p_max = 0,
                p_nom_extendable=True,
                capital_cost=500,
                marginal_cost=50,
                efficiency_store=0.9,
                efficiency_dispatch=0.9)  
        
    return grid

