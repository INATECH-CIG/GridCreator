import pandas as pd
import numpy as np

import functions as func
import input_data as ds

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
    #current_occupancy = occ.get_occ_profile_in_curr_timestep()

    # EV charges only when all residents are home
    charging_array = np.where(current_occupancy == max_occ, 1, 0)

    # Assume 7 kW charging power, convert to MW
    charging_power = 7e-3  # 7 kW → 0.007 MW


    # Convert charging array to Pandas Series to use .diff() and .loc
    charging = pd.Series(charging_array, index=index) 

    # Setting state of charge = 1 at departure times (1 → 0)
    change = charging.diff()
    departures = (change == -1)

    soc_set = pd.Series(np.nan, index=charging.index)
    soc_set.loc[departures] = 1.0

    '''
    SOC wird gerade auf Abfahrtszeitpunkt gelegt, passt vllt auch?
    '''

    # # Setze SoC eine Stunde vor Abfahrt
    # soc_set.loc[departures.shift(-1, fill_value=False)] = 1.0

    # Energy loss during absence (spill)
    spill = pd.Series(0.0, index=charging.index)
    spill.loc[charging == 0] = 0.1 * charging_power  # 10% of charging power when not at home

    return soc_set, spill, charging_power, charging

#%%

top =  47.967835 # # Upper latitude
bottom = 47.955593 # Lower latitude
left =  7.735381   # Right longitude
right =  7.772647   # Left longitude

bbox = [left, bottom, right, top]
path = ds.save_data()
environment = func.env_weather(bbox, path)

start_time = pd.Timestamp("2023-01-01 00:00:00")
snapshots = pd.date_range(start=start_time, periods=environment.timer.timesteps_total, freq=f"{int(environment.timer.time_discretization/60)}min")

#%%
persons = 2
power, occupants = create_haus(people=persons, index=snapshots, env=environment)
soc_set, spill, charging_power, charging = create_e_car(occ = occupants, index=snapshots)


# %%
