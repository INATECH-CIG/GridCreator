import numpy as np
import pandas as pd


from data import solar_dict
#%% PyCity-Module
from pycity_base.classes.timer import Timer
from pycity_base.classes.weather import Weather
from pycity_base.classes.prices import Prices
from pycity_base.classes.environment import Environment

from pycity_base.classes.demand.occupancy import Occupancy
from pycity_base.classes.demand.electrical_demand import ElectricalDemand


import pycity_base.classes.supply.photovoltaic as pv

import pycity_base.classes.supply.heat_pump as hp


#%%
def create_haus(env, people, index, light_config = 10, meth = 2, weather_file=None):



    # Nutzerprofil (3 Personen im Haushalt)
    occupancy = Occupancy(env, number_occupants=people)

    # Strombedarf (stochastisch, mit Geräten & Licht)
    el_demand = ElectricalDemand(
        env,
        method=meth,
        total_nb_occupants=people,
        randomize_appliances=True,
        light_configuration=light_config,
        occupancy=occupancy.occupancy
    )
    power = el_demand.get_power()
    # Zeitstempel erzeugen
    """
    Sollte auch generell definiert und dann nur in Funktion  übergeben werden
    """
    #start = pd.Timestamp("2025-01-01 00:00:00")  # Simulationsstart definieren
    #index = pd.date_range(start=start, periods=len(power), freq=f'{int(timer.time_discretization/3600)}H')

    # Rückgabe als Pandas-Serie
    power_series = pd.Series(power, index=index, name='el_load_kw')

    return power_series, occupancy # Occupancy-Objekt zurückgeben,
                                # falls du z. B. in PyPSA auch Anwesenheiten
                                # (z. B. für Steuerstrategien oder EV-Laden)
                                # nutzen willst


def create_e_car(env, occ, index):
    """
    Auto wird geladen wenn alle Bewhoner da sind
    """
    max_occ = np.max(occ.get_occ_profile_in_curr_timestep())
    laden = occ.get_occ_profile_in_curr_timestep()
    for idx in range(len(laden)):
        if laden[idx] == max_occ:
            laden[idx] = 1
        else:
            laden[idx] = 0
    """
    Mit wie viel lädt ein Auto?
    """
    laden = laden * 7  # 7 kW Ladeleistung
    
    laden = pd.Series(laden, index=index)


    return laden


def create_pv(env, peakpower, index, beta, gamma, area=10.0, eta_noct=0.15, meth=1):
    # Umgebung (Timer, Wetter, Preise)



    # PV-Anlage erstellen
    # Nur peakpower nötig, wenn method = 1
    # wenn method = 0, dann area, eta_noct und beta nötig
    gamma = solar_dict[gamma]

    pv_system = pv.PV(peak_power=peakpower, environment=env, area=area, eta_noct=eta_noct, beta=beta, gamma=gamma, method=meth)
    
    # Berechnung der PV-Leistung
    pv_power = pv_system.getPower()

    # Rückgabe als Pandas-Serie
    power_series = pd.Series(pv_power, index=index, name='pv_power_kw')

    return power_series # Nur Rückgabe von Power




def create_hp(index, env, hp_params=None, flow_temp=None, schedule= None):
    """
    Erzeugt ein Wärmepumpen-Profil.

    Args:
        index: Zeitstempel für das Wärmepumpen-Profil
        hp_params: Parameter für die Wärmepumpe (optional)
        flow_temp: Vorlauftemperatur (optional, Standardwert wird verwendet)
        schedule: Betriebsprofil der Wärmepumpe (optional, Standardwert wird verwendet)
        env: Environment-Objekt (optional, falls benötigt)

    Returns:
        power_series: Pandas-Serie mit der elektrischen Leistung der Wärmepumpe in kW
    """

    if hp_params is None:
        '''
        Werte sind von ChatGPT
        '''
        hp_params = {
                    "t_ambient": np.array([-10, 0, 10, 20]),  # typische Außentemperaturen
                    "t_flow": np.array([35, 45, 55]),         # typische Vorlauftemperaturen
                    "heat": np.array([                        # abgegebene Heizleistung [kW]
                        [4.5, 4.5, 4.5],   # bei -10 °C
                        [5.0, 5.0, 5.0],   # bei 0 °C
                        [5.5, 5.5, 5.5],   # bei 10 °C
                        [6.0, 6.0, 6.0]    # bei 20 °C
                    ]),
                    "power": np.array([                        # elektrische Leistungsaufnahme [kW]
                        [2.2, 2.3, 2.4],
                        [1.9, 2.0, 2.1],
                        [1.7, 1.8, 1.9],
                        [1.6, 1.6, 1.7]
                    ]),
                    "cop": np.array([                          # COP = heat / power
                        [2.05, 1.95, 1.88],
                        [2.63, 2.50, 2.38],
                        [3.24, 3.06, 2.89],
                        [3.75, 3.56, 3.35]
                    ]),
                    "t_max": 55,
                    "lower_activation_limit": 0.5
                }


    # Wärmepumpe erstellen
    heatpump = hp.Heatpump(env, **hp_params)

    # Dummy-Vorlauftemperatur über Zeitschritte (z. B. 35 °C ± Zufall)
    """
    Standardwert oder variabel simulierbar?
        Bsp:
            if flow_temp is None:
                t_inside = 21
                a, b = 0.5, 35
                t_ambient_series = environment.weather.t_ambient
                flow_temp = a * (t_inside - t_ambient_series) + b  # einfache Heizkurve
    """
    # Dummy-Vorlauftemperatur über Zeitschritte (z. B. 35 °C ± Zufall)
    if flow_temp is None:
        flow_temp = np.full(len(index), 45)

    # Nominalwerte berechnen
    nominals= heatpump.getNominalValues(flow_temp)


    # Einfaches Betriebsprofil (z. B. Zufall ein/aus)
    if schedule is None:
        schedule = (env.weather.t_ambient < 15).astype(int)
    """
    Wie entscheide ich wann HP läuft?

        Ein realistischer Betrieb ergibt sich z.B. aus dem:
            Wärmebedarf (z.B. aus SpaceHeating oder HeatingDemand)
            Regelstrategie (z.B. einfache Zwei-Punkt-Regelung)
            Temperaturvergleich (Innensoll vs. Raumtemp)
            Bsp:
                schedule = (environment.weather.t_ambient < 15).astype(int)
    """

    # Verbrauch und Erzeugung berechnen
    power = nominals[0] * schedule

    # Rückgabe als Pandas-Serie
    power_series = pd.Series(power, index=index, name='hp_power_kw')

    return power_series # Rückgabe nur von elektrischer Leistung

