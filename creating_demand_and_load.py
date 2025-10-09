from dicts import solar_dict

import numpy as np
import pandas as pd
from pycity_base.classes.demand.occupancy import Occupancy
from pycity_base.classes.demand.electrical_demand import ElectricalDemand
import pycity_base.classes.supply.photovoltaic as pv
import pycity_base.classes.supply.heat_pump as hp

from pycity_base.classes.environment import Environment


#%%
def create_haus(env: Environment, people: int, index: pd.DatetimeIndex, light_config: int = 10, meth: int = 2, weather_file=None) -> (pd.Series, Occupancy):
    """
    Generates the electrical load profile for a household based on occupancy and stochastic appliance usage.

    Args:
        env: Environment object containing weather and simulation settings.
        people (int): Number of people in the household.
        index (pd.DatetimeIndex): Timestamps for the load profile.
        light_config (int, optional): Lighting configuration, defaults to 10.
        method (int, optional): Method for ElectricalDemand calculation, defaults to 2.
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
    Generates the electric vehicle (EV) charging profile based on household occupancy.
    The EV charges only when all residents are at home.

    Args:
        occ (Occupancy): Occupancy object for the household.
        index (pd.DatetimeIndex): Timestamps for the load profile.

    Returns:
        pd.Series: EV charging power in MW at each timestep.
    """

    # Determine occupancy at each timestep
    max_occ = np.max(occ.get_occ_profile_in_curr_timestep())
    current_occupancy = occ.get_occ_profile_in_curr_timestep()

    # EV charges only when all residents are home
    charging = np.where(current_occupancy == max_occ, 1, 0)
    
    # Assume 7 kW charging power, convert to MW
    charging_power = charging * 7e-3  # 7 kW → 0.007 MW

    # Create Pandas Series with the given index
    charging_power = pd.Series(charging_power, index=index)

    return charging_power


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
    
    # Create PV system
    # Note: peakpower is already in MW, so no conversion needed
    pv_system = pv.PV(peak_power=peakpower, environment=env, area=area, eta_noct=eta_noct, beta=beta, gamma=gamma, method=meth)
    
    # Calculate PV power (already in MW)
    pv_power = pv_system.getPower()

    # Create Pandas Series with the given index
    power_series = pd.Series(pv_power, index=index, name='pv_power_MW')

    return power_series



def create_hp(index: pd.DatetimeIndex, env: Environment, hp_params=None, flow_temp=None, schedule=None) -> pd.Series:
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

    if hp_params is None:
        '''
        Werte aus
        file:///home/matthiasbehr/Downloads/tl_en_technicky-list_ea-622m.pdf
        RPS 50 Hz
        '''
        hp_params = {
                    "t_ambient": np.array([12, 7, 2, -7, -15]),     
                    "t_flow": np.array([35, 45, 55]),               

                    # Heizleistung [kW]
                    "heat": np.array([
                        [13.50, 12.96, 12.41],   # 12 °C
                        [10.30, 10.33, 10.35],   # 7 °C
                        [8.27,  8.70,  9.12],    # 2 °C
                        [7.29,  7.11,  6.93],    # -7 °C
                        [5.77,  5.64,  5.51]     # -15 °C
                    ]),

                    # Leistungsaufnahme [kW]
                    "power": np.array([
                        [2.49, 3.01, 3.52],      # 12 °C
                        [2.27, 2.80, 3.32],      # 7 °C
                        [2.19, 2.77, 3.35],      # 2 °C
                        [2.18, 2.64, 3.10],      # -7 °C
                        [2.07, 2.60, 3.12]       # -15 °C
                    ]),

                    # COP
                    "cop": np.array([
                        [5.41, 4.31, 3.53],      # 12 °C
                        [4.53, 3.69, 3.12],      # 7 °C
                        [3.78, 3.14, 2.72],      # 2 °C
                        [3.34, 2.69, 2.24],      # -7 °C
                        [2.79, 2.17, 1.77]       # -15 °C
                    ]),
                    "t_max": 55,
                    "lower_activation_limit": 0.5
                }


    # Initialize heat pump
    heatpump = hp.Heatpump(env, **hp_params)

    # Default supply temperature over time
    if flow_temp is None:
        flow_temp = np.full(len(index), 45)

    # Nominal electric power values for given flow temps
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
        el_demand = ElectricalDemand(env, method=meth, annual_demand=demand_per_year, profile_type=type)
        power = el_demand.get_power()

    # Get electric power in W and convert to MW
    power = power * 1e-6

    # Create Pandas Series with the given index
    power_series = pd.Series(power, index=index, name='el_load_MW')

    return power_series
