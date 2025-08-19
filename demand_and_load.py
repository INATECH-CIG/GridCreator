import numpy as np
import pandas as pd

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


def create_pv(env, peakpower, index, area=10.0, eta_noct=0.15, beta=30, meth = 1):
    # Umgebung (Timer, Wetter, Preise)



    # PV-Anlage erstellen
    # Nur peakpower nötig, wenn method = 1
    # wenn method = 0, dann area, eta_noct und beta nötig
    pv_system = pv.PV(peak_power=peakpower, environment=env, area=area, eta_noct=eta_noct, beta=beta, method=meth)
    
    # Berechnung der PV-Leistung
    pv_power = pv_system.getPower()

    # Rückgabe als Pandas-Serie
    power_series = pd.Series(pv_power, index=index, name='pv_power_kw')

    return power_series # Nur Rückgabe von Power



def create_hp(index, hp_params=None, flow_temp=None, schedule= None, env=None):

    if hp_params is None:
        hp_params = {
                        "t_ambient": np.array([0, 10, 20]),
                        "t_flow": np.array([35, 45, 55]),
                        "heat": np.array([[5, 5, 5], [6, 6, 6], [7, 7, 7]]),
                        "power": np.array([[1.6, 1.6, 1.6], [1.7, 1.7, 1.7], [1.8, 1.8, 1.8]]),
                        "cop": np.array([[3.1, 3.1, 3.1], [3.5, 3.5, 3.5], [3.9, 3.9, 3.9]]),
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
        schedule = np.random.randint(0, 2, size=len(index))
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

