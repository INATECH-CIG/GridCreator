from dicts import solar_dict

import numpy as np
import pandas as pd
from pycity_base.classes.demand.occupancy import Occupancy
from pycity_base.classes.demand.electrical_demand import ElectricalDemand
import pycity_base.classes.supply.photovoltaic as pv
import pycity_base.classes.supply.heat_pump as hp

from pycity_base.classes.environment import Environment
import os
import sys

'''
Module for creating demand and load profiles for different technologies.
'''

#%%
def create_appartment(env: Environment, people: int, index: pd.DatetimeIndex, light_config: int = 10, meth: int = 2, weather_file=None) -> (pd.Series, Occupancy):
    """
    Generates the electrical load profile for a household based on occupancy and stochastic appliance usage.

    Args:
        env: Environment object containing weather and simulation settings.
        people (int): Number of people in the household.
        index (pd.DatetimeIndex): Timestamps for the load profile.
        light_config (int, optional): Lighting configuration, defaults to 10.
        meth (int, optional): Method for ElectricalDemand calculation, defaults to 2.
        weather_file (optional): Placeholder for future weather-dependent demand (currently unused).

    Returns:
        power_series (pd.Series): Electrical load of the household in MW.
        occupancy (Occupancy): Occupancy object containing presence information.
    """


    # Occupancy profile
    occupancy = Occupancy(env, number_occupants=people)

    # Electrical demand (stochastic, includes appliances and lighting)
    el_demand = ElectricalDemand(env, method=meth, total_nb_occupants=people, randomize_appliances=True, light_configuration=light_config, occupancy=occupancy.occupancy)
    power = el_demand.get_power()

    # Convert from W to MW
    power = power * 1e-6

    # Create Pandas Series with the given index
    power_series = pd.Series(power, index=index, name='el_load_MW')

    return power_series, occupancy


def create_e_car(occ: Occupancy, index: pd.DatetimeIndex) -> pd.Series:
    """
    Generates an electric vehicle (EV) charging profile based on occupancy.

    Args:
        occ: Occupancy object containing presence information.
        index (pd.DatetimeIndex): Timestamps for the EV profile.

    Returns:
        soc_set (pd.Series): State of charge setpoints at departure times (1.0 = fully charged).
        spill (pd.Series): Energy loss during absence (10% of charging power).
        charging_power (float): Constant charging power in MW (7 kW = 0.007 MW).
    """

    # Determine occupancy at each timestep
    current_occupancy = np.rint(occ.get_occ_profile_in_curr_timestep()).astype(int)
    max_occ = np.max(current_occupancy)
    
    # EV charges only when all residents are home
    charging_array = np.where(current_occupancy == max_occ, 1, 0)

    # Assume 11 kW charging power, convert to MW (https://www.solarwatt.de/ratgeber/bidirektionales-laden)
    charging_power = 11e-3  # 11 kW → 0.011 MW

    # Convert charging array to Pandas Series to use .diff() and .loc
    charging = pd.Series(charging_array, index=index) 

    # Energy loss during absence (spill)
    spill = pd.Series(0.0, index=charging.index)
    spill.loc[charging == 0] = 0.1 * charging_power  # 10% of charging power when not at home

    return spill, charging_power, charging


def create_pv(env: Environment, peakpower: float, index: pd.DatetimeIndex, beta: float, gamma: str, area: float = 10.0, eta_noct: float = 0.15, meth: int = 1) -> pd.Series:
    """
    Generates a PV system power profile.

    Args:
        env: Environment object (with weather, timer, prices, etc.).
        peakpower (float): Peak power of the PV system (MW).
        index (pd.DatetimeIndex): Timestamps for the PV profile.
        beta (float): Tilt angle of the PV system (degrees).
        gamma (str): Orientation of the PV system ('South', 'East', etc.).
        area (float): Module area in m² (default: 10.0).
        eta_noct (float): Efficiency under NOCT (default: 0.15).
        meth (int): Method for PV calculation (1 = peakpower-based, 0 = area-based).

    Returns:
        pd.Series: PV power in MW at each timestep.
    """

    # Convert gamma string to internal PV dictionary format
    gamma = solar_dict[gamma]
    
    # Create PV system with peak power in kW
    pv_system = pv.PV(peak_power=peakpower, environment=env, area=area, eta_noct=eta_noct, beta=beta, gamma=gamma, method=meth)
    
    # Calculate PV power in W
    pv_power = pv_system.getPower()

    # Convert from W to MW
    pv_power = pv_power * 1e-6

    # Create Pandas Series with the given index
    power_series = pd.Series(pv_power, index=index, name='pv_power_MW')

    return power_series



def create_hp(index: pd.DatetimeIndex, env: Environment, flow_temp=None, schedule=None) -> pd.Series:
    """
    Generates a heat pump power profile.

    Args:
        index (pd.DatetimeIndex): Timestamps for the heat pump profile.
        env: Environment object (with weather, timer, etc.).
        hp_params (dict, optional): Heat pump parameters. Defaults provided if None.
        flow_temp (array-like, optional): Supply temperatures over time. If None, default 45°C is used.
        schedule (array-like or pd.Series, optional): Binary schedule (1 = ON, 0 = OFF). 
            If None, defaults to running when ambient temperature < 15°C.

    Returns:
        pd.Series: Heat pump electric power in MW for each timestep.
    """


    #     '''
    #     Werte aus
    #     file:///home/matthiasbehr/Downloads/tl_en_technicky-list_ea-622m.pdf
    #     RPS 50 Hz
    #     '''

    #     t_ambient = np.array([12, 7, 2, -7, -15])
    #     t_flow = np.array([35, 45, 55])

    #     # Heizleistung [kW]
    #     heat = np.array([
    #                     [13.50, 12.96, 12.41],   # 12 °C
    #                     [10.30, 10.33, 10.35],   # 7 °C
    #                     [8.27,  8.70,  9.12],    # 2 °C
    #                     [7.29,  7.11,  6.93],    # -7 °C
    #                     [5.77,  5.64,  5.51]     # -15 °C
    #                 ])

    #                 # Leistungsaufnahme [kW]
    #     power = np.array([
    #                     [2.49, 3.01, 3.52],      # 12 °C
    #                     [2.27, 2.80, 3.32],      # 7 °C
    #                     [2.19, 2.77, 3.35],      # 2 °C
    #                     [2.18, 2.64, 3.10],      # -7 °C
    #                     [2.07, 2.60, 3.12]       # -15 °C
    #                 ])

    #                 # COP
    #     cop = np.array([
    #                     [5.41, 4.31, 3.53],      # 12 °C
    #                     [4.53, 3.69, 3.12],      # 7 °C
    #                     [3.78, 3.14, 2.72],      # 2 °C
    #                     [3.34, 2.69, 2.24],      # -7 °C
    #                     [2.79, 2.17, 1.77]       # -15 °C
    #                 ])
    #     t_max = 55
    #     lower_activation_limit = 0.5


    '''
    Werte aus
    pyCity Beispiel
    '''
    heat = np.array([
                    [2960, 2260, 2030],
                    [3730, 3150, 2780],
                    [5500, 4560, 3980],
                    [7500, 6530, 5880],
                    [9200, 8210, 7100],
                    [10200, 8880, 7810],
                    [11450, 10290, 9110],
                    [12700, 11700, 10400]
                ])
    cop = np.array([

                    [1.86, 1.42, 1.28],
                    [2.16, 1.79, 1.51],
                    [2.8, 2.28, 1.87],
                    [3.7, 2.86, 2.45],
                    [4.2, 3.62, 2.7],
                    [4.5, 3.55, 2.95],
                    [4.77, 4.04, 3.31],
                    [5.29, 4.33, 3.48]
                ])
    power = heat / cop

    t_ambient = np.array([-20, -15, -7, 2, 7, 10, 12, 20])
    t_flow = np.array([35, 45, 55])
    t_max = 55
    lower_activation_limit = 0.5

    # Initialize heat pump
    heatpump = hp.Heatpump(env, t_ambient, t_flow, heat, power, cop, t_max, lower_activation_limit)

    # Default supply temperature over time
    if flow_temp is None:
        flow_temp = np.full(len(index), 45)

    # Nominal electric power values for given flow temps in kW
    nominals= heatpump.getNominalValues(flow_temp)

    # Default schedule based on ambient temperature if not provided
    if schedule is None:
        schedule = (env.weather.t_ambient < 15).astype(int)

    # Calculate electric power in MW
    power = nominals[0] * 1e-3 * schedule

    # Create Pandas Series with the given index
    power_series = pd.Series(power, index=index, name='hp_power_MW')

    return power_series


def create_commercial(type: str, demand_per_year, index, env):
    """
    Generates a commercial building electricity demand profile.

    Args:
        type (str): Type of commercial building (e.g., "office", "retail", "sports").
        demand_per_year (float): Annual electricity demand [kWh or W·h? consistent with ElectricalDemand].
        index (pd.DatetimeIndex): Timestamps for the profile.
        env: Environment object (for weather data if required).

    Returns:
        pd.Series: Electrical demand of the commercial building in MW.
    """

    if type == 'sports':
        meth = 3
        el_demand = ElectricalDemand(env, method=meth, method_3_type=type, annual_demand=demand_per_year)
        power = el_demand.get_power()

    else:
        # Strombedarf (stochastisch, mit Geräten & Licht)
        meth = 1
        # this is a quickfix to handle XLRDError ".xlsx format not supported" coming from the xlrd library used in pycity_base
        # try and if fails convert the files from xlsx to xls
        # TODO remove this when pyCity has fixed this issue
        # for this quickfix to work you must also change
        # 'pycity_base/classes/demand/electrical_demand.py' line 159 from
        # 'slp_electrical_2019.xlsx' to 'slp_electrical_2019.xls'
        try:
            el_demand = ElectricalDemand(env, method=meth, annual_demand=demand_per_year, profile_type=type, weather_file='data/weather/weather_dummy.xlsx')
            power = el_demand.get_power()
        except:
            print('Error: XLDR cannot open xlsx file.\nPlease convert the xlsx files in the pycity_base inputs/standard_load_profile folder to xls format and retry.')
            # print('Warning: XLDR cannot open xlsx file.\nAttempting to convert xlsx files to xls format...')
            # # convert xlsx files in folder to xls files
            # # get path of the active environment (Conda/Venv) or fall back to the current Python prefix
            # active_env_path = os.environ.get('CONDA_PREFIX') or os.environ.get('VIRTUAL_ENV') or sys.prefix
            # active_env_path = os.path.normpath(active_env_path)
            # print(f'Active environment path: {active_env_path}')
            # # join path with folder containing the xlsx files
            # folder = os.path.join(active_env_path, 'Lib', 'site-packages', 'pycity_base', 'inputs', 'standard_load_profile')
            # # transform all xlsx files in the folder to xls files
            # for file in os.listdir(folder):
            #     print(f'Checking file: {file}')
            #     if file.endswith('.xlsx'):
            #         xlsx_path = os.path.join(folder, file)
            #         xls_path = os.path.join(folder, file.replace('.xlsx', '.xls'))
            #         try:
            #             # read file with pandas and openpyxl engine, then save as xls file
            #             data_xls = pd.read_excel(xlsx_path, engine='openpyxl')
            #             # Let pandas infer the writer engine from the .xls extension (preferred)
            #             # TODO Ramiz fix this
            #             # data_xls.to_excel(xls_path, index=False, engine='openpyxl')
            #             # print(f'Converted {file} to {os.path.basename(xls_path)}')
            #             print(f'Not yet implemented: Please convert the file {file} manually and retry.')
            #         except Exception as e:
            #             print(f'Failed to convert {file}: {e}')
            
            # # retry creating the ElectricalDemand with the new xls file
            # el_demand = ElectricalDemand(env, method=meth, annual_demand=demand_per_year, profile_type=type)
            # power = el_demand.get_power()

    # Get electric power in W and convert to MW
    power = power * 1e-6

    # Create Pandas Series with the given index
    power_series = pd.Series(power, index=index, name='el_load_MW')

    return power_series
