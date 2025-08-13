
import functions as func
import ding0_grid_generators as ding0
import data_combination as dc
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import demand_and_load as demand_load
from pycity_base.classes.timer import Timer
from pycity_base.classes.weather import Weather
from pycity_base.classes.prices import Prices
from pycity_base.classes.environment import Environment


def daten_laden(ordner):
    """
    Lädt Zensusdaten aus CSV-Dateien in einem angegebenen Ordner und gibt sie als DataFrame zurück.
    
    Args:
        ordner (str): Der Pfad zum Ordner, der die Zensusdaten im CSV-Format enthält.
        
    Returns:
        pd.DataFrame: Ein DataFrame, das die kombinierten Zensusdaten enthält.
    """
    # Manuell laden
    pd_Zensus_Bevoelkerung_100m = pd.read_csv(ordner + "/Zensus2022_Bevoelkerungszahl_100m-Gitter.csv", sep=";")

    pd_Zensus_Durchschn_Nettokaltmiete_100m = pd.read_csv(ordner + "/Zensus2022_Durchschn_Nettokaltmiete_100m-Gitter.csv", sep=";")
    

    pd_Zensus_Eigentuemerquote_100m = pd.read_csv(ordner + "/Zensus2022_Eigentuemerquote_100m-Gitter.csv", sep=";")
    

    pd_Zensus_Heizungsart_100m = pd.read_csv(ordner + "/Zensus2022_Heizungsart_100m-Gitter.csv", sep=";", encoding="cp1252")
    

    # Zensus Daten in Liste umwandeln
    data = [pd_Zensus_Bevoelkerung_100m, pd_Zensus_Heizungsart_100m, pd_Zensus_Durchschn_Nettokaltmiete_100m, pd_Zensus_Eigentuemerquote_100m]

    for df in data:
        df.replace("–", 0.0, inplace=True)
        if "werterlaeuternde_Zeichen" in df.columns:
            df.drop(columns=["werterlaeuternde_Zeichen"], inplace=True)

    # Kombinieren der DataFrames
    daten = func.zensus_df_verbinden(data)

    return daten


def ding0_grid(bbox, grids_dir, output_file_grid):
    """
    Erstellt ein Grid-Objekt basierend auf den gegebenen Bounding Box-Koordinaten und speichert es in einer NetCDF-Datei.
    
    Args:
        bbox (list): Eine Liste mit den Koordinaten der Bounding Box in der Form [left, bottom, right, top].
        grids_dir (str): Der Pfad zum Verzeichnis, in dem das Grid gespeichert werden soll.
        output_file_grid (str): Der Name der Ausgabedatei für das Grid im NetCDF-Format.
        
    Returns:
        tuple: Ein Tupel bestehend aus dem Grid-Objekt und der Bounding Box.
    """
    
    # Netz extrahieren
    grid = ding0.load_grid(bbox, grids_dir)

    # neue bbox für alle enthaltenen buses laden
    bbox_neu = func.compute_bbox_from_buses(grid)

    # Grid creation für erweiterte bbox
    grid = ding0.load_grid(bbox_neu, grids_dir)

    # # Grid speichern
    # grid.export_to_netcdf(output_file_grid)

    return grid, bbox_neu


def osm_data(net, buses_df, bbox_neu, buffer):
    """
    Ruft OSM-Daten für die gegebene Bounding Box ab und gibt sie zurück.

    Args:
        net: Das Netzwerkobjekt, das die Buslinien enthält.
        bbox_neu: Die erweiterte Bounding Box für die OSM-Abfrage.
        buffer: Der Pufferbereich um die Bounding Box.

    Returns:
        tuple: Ein Tupel bestehend aus den OSM-Daten und den OSM-Feature-Daten.
    """

    left, bottom, right, top = bbox_neu
    bbox_osm = (left - buffer, bottom - buffer, right + buffer, top + buffer)
    # osm Data abrufen
    Area, Area_features = func.get_osm_data(bbox_osm)
    # Speichern der OSM Daten
    Area_features.to_file("Area_features.geojson", driver="GeoJSON")

    Area_features_df = Area_features.reset_index()

    # Daten kombinieren
    buses_df = dc.data_combination(net, buses_df, Area_features_df)

    return buses_df, Area, Area_features


def daten_zuordnung(buses, bundesland_data, zensus_data):
    """
    Weist den Bussen im Netzwerk Bundesland- und Zensusdaten zu.
    
    Args:
        net: Das Netzwerkobjekt, das die Buslinien enthält.
        bundesland_data: DataFrame mit den Bundesland-Daten.
        zensus_data: DataFrame mit den Zensus-Daten.
        
    Returns:
        net: Das aktualisierte Netzwerkobjekt mit zugeordneten Bundesland- und Zensusdaten.
    """

    # Bundesland
    buses = func.bundesland(buses, bundesland_data)

    # Zensus Daten
    buses = func.daten_zuordnen(buses, zensus_data)

    return buses



def bundesland_zensus(zensus, datei):
    """
    Lädt die Bundesland-Daten aus einer GeoJSON-Datei und ordnet sie den Zensusdaten zu.
    
    Args:
        zensus (pd.DataFrame): DataFrame mit den Zensusdaten.
        datei (str): Pfad zur GeoJSON-Datei mit den Bundesland-Daten.

    Returns:
        pd.DataFrame: DataFrame mit den zugeordneten Bundesland-Daten.
    """

    lan = func.load_shapes(datei)

    df = func.bundesland_zuordnung(zensus, lan)

    return df




def technik_zuordnen(buses, file_Faktoren, kategorien_eigenschaften, Bev_data_Zensus, file_Technik, technik_arr):
    """
    Ordnet den Bussen im Grid verschiedene Techniken zu und berechnet die entsprechenden Faktoren.
    Args:
        grid: Das Grid-Objekt, das die Buslinien enthält.
        factors: Ein Dictionary mit den Faktoren für jede Technik.
        kategorien_eigenschaften: Die Kategorien der Eigenschaften, die für die Zuordnung verwendet werden.
        Bev_data_Zensus: DataFrame mit den Zensusdaten.
        Bev_data_Technik: DataFrame mit den Technikinformationen.
        technik_arr: Liste der Techniken, die zugeordnet werden sollen.
        
    Returns:
        tuple: Ein Tupel bestehend aus dem aktualisierten Grid-Objekt und einem Array der Faktor-Bounding Box.
    """

    Technik_Faktoren = pd.read_csv(file_Faktoren, sep=";")
    Technik_Faktoren = Technik_Faktoren.set_index("Technik")
    Bev_data_Technik = pd.read_csv(file_Technik, sep=",")

    Bev_data = pd.merge(Bev_data_Zensus, Bev_data_Technik, on="GEN", how="left")

    df_land = buses['lan_name'].copy().to_frame()

    df_zensus = buses[[col for col in buses.columns if col.startswith("Zensus")]].copy()
    # Kommas durch Punkte ersetzen, damit pd.to_numeric klappt; In float umwandeln; Fehlende Werte auf 0 setzen
    df_zensus = (df_zensus.astype(str).replace(",", ".", regex=True).apply(pd.to_numeric, errors="coerce").fillna(0.0))

    df_eigenschaften = pd.concat([df_land, df_zensus], axis=1)

    factor_bbox = np.array([0.0] * len(technik_arr))
    for i, technik in enumerate(technik_arr):
        print(f"Berechne Faktoren für {technik}...")
        factors_technik = Technik_Faktoren.loc[technik]
        print('Berechne Faktoren für', factors_technik)
        buses['Factor_' + technik], factor_bbox[i] = func.calculate_factors(df_eigenschaften, factors_technik, kategorien_eigenschaften, Bev_data, technik)

    return buses, factor_bbox



def technik_fill(buses, Technik, p_total):
    """
    Füllt das Grid-Objekt mit den Techniken basierend auf den gegebenen Anteilen.
    
    Args:
        grid: Das Grid-Objekt, das die Buslinien enthält.
        Technik (list): Liste der Techniken, die zugeordnet werden sollen.
        p_total (list): Liste der Anteile für jede Technik.
        
    Returns:
        grid: Das aktualisierte Grid-Objekt mit den zugeordneten Techniken.
    """

    for tech, p in zip(Technik, p_total):
        buses = func.technik_sortieren(buses, tech, p)
        
    buses = func.storage(buses)

    return buses



def loads_zuordnen(grid, buses, bbox, env=None):
    if env is None:
        environment = func.env_wetter(bbox)
    else:
        environment = env
        
    
    # Powerflow zu bestehendem Netz
    grid.pf()

    # Zeit sollte auch schon integriert sein, wenn Lasten schon im Netz sind
    """
    Angepasst an env ?????
    """
    start_time = pd.Timestamp("2023-01-01 00:00:00")
    snapshots = pd.date_range(start=start_time, periods=environment.timer.timesteps_total, freq=f"{int(environment.timer.time_discretization/60)}min")

    grid.set_snapshots(snapshots)




    # Lasten sollten schon im Netz integriert sein, hier nur Beispielwerte
    """
    Alle Loads erstmal löschen? Weil von ding0 und dann alle einheitlich?
    """
    # for bus in buses.index:
    #     print(f"Prüfe, ob Load für {bus} existiert...")
    #     existing = grid.loads[(grid.loads['bus'] == bus)]
    #     if existing.empty:
    #         # Lasten hinzufügen
    #         """
    #         Max. allowed number of occupants per apartment is 5
    #         """
    #         # power = demand_load.create_haus(people=buses.loc[bus, 'Zensus_Einwohner_sum'], env=environment)
    #         power, occupants = demand_load.create_haus(people=3, index=snapshots, env=environment)
    #         # Hier wird angenommen, dass p_set eine Serie ist, die die Lasten für jede Stunde enthält
    #         # power = pd.Series(power[:len(grid.snapshots)], index=grid.snapshots)
    #         print("Snapshots grid:", grid.snapshots)
    #         print("Index power:", power.index)
    #         print("Sind sie gleich? ", power.index.equals(grid.snapshots))

    #         print(f"Load {bus}_load wird hinzugefügt.")
    #         grid.add("Load",
    #                 name=bus + "_load",
    #                 bus=bus,
    #                 carrier="AC",
    #                 p_set=power)
    #         print(f"Load {bus}_load jetzt hinzugefügt.")
    #     else:
    #         print(f"Load für {bus} existiert bereits.")
    

    print("Lasten hinzugefügt.")

    # Setzen von p_set
    grid.generators_t.p_set = pd.DataFrame(index=grid.snapshots)
    """
    Carrier und Type komplett egal?
    """

    """
    Alle Solargeneratoren erstmal löschen? Weil von ding0 und dann alle einheitlich?
    """
    # #solar_buses = buses.index[buses["Power_solar"].notna()]
    # solar_buses = buses.index[buses["Power_solar"] != 0]
    
    # for bus in solar_buses:
    #     print(f"Prüfe, ob Generator für {bus} existiert...")
    #     existing = grid.generators[(grid.generators['bus'] == bus)]

    #     """
    #     Power für Solar ist immer 0, warum?
    #     Im Test gab es schöne Kurven
    #     """
    #     power = demand_load.create_pv(peakpower=buses.loc[bus, 'Power_solar'], index=snapshots, env=environment)
    #     if existing.empty:
    #         # Generator hinzufügen
            
    #         grid.add("Generator",
    #                 name=bus + "_solar",
    #                 bus=bus,
    #                 carrier="solar",
    #                 type="solar",
    #                 p_nom=buses.loc[bus, 'Power_solar'])
            

    #         grid.generators_t.p_max_pu[bus + "_solar"] = power.values

    #         print(f"Generator {bus}_solar hinzugefügt.")

    #     else:
    #         # Load zu existierendem Generator hinzufügen
    #         grid.generators_t.p_max_pu[bus + "_solar"] = power.values

    #         print(f"Generator für {bus} existiert bereits.")



    print("Solargeneratoren hinzugefügt.")

    #HP_amb_buses = buses.index[buses["Power_HP_ambient"].notna()]
    HP_amb_buses = buses.index[buses["Power_HP_ambient"] != 0]
    for bus in HP_amb_buses:
        # Generator hinzufügen
        power = demand_load.create_hp(index=snapshots, env=environment)
        print(len(power), "Power:", power)
        
        grid.add("Generator",
                name=bus + "_HP_ambient",
                bus=bus,
                carrier="HP_ambient", # carrier definieren, damit es nicht mit anderen HPs kollidiert
                type="HP_ambient") # eventuell nicht so wichtig
        grid.generators_t.p_max_pu[bus + "_HP_ambient"] = power.values

        print(f"Generator {bus}_HP_ambient hinzugefügt.")



    # """
    # HP Geothermal und Ambient sind gleich, nur Carrier unterschiedlich
    # """
    # #HP_geo_buses = buses.index[buses["Power_HP_geothermal"].notna()]
    # HP_geo_buses = buses.index[buses["Power_HP_geothermal"] != 0]
    # for bus in HP_geo_buses:
    #     # Generator hinzufügen
    #     power = demand_load.create_hp(index=snapshots, env=environment)
    #     grid.add("Generator",
    #             name=bus + "_HP_geothermal",
    #             bus=bus,
    #             carrier="HP_geothermal",
    #             type="HP_geothermal")
    #     grid.generators_t.p_max_pu[bus + "_HP_geothermal"] = power.values

    return grid

def pypsa_vorbereiten(grid):

    # Kosten für Solaranlagen müssen wahrscheinlich allgemein gesetzt werden

    grid.generators.loc[grid.generators['type'] == 'solar', 'marginal_cost'] = 100
    grid.generators.loc[grid.generators['type'] == 'solar', 'capital_cost'] = 1000
    grid.generators.loc[grid.generators['type'] == 'solar', 'efficiency'] = 0.9
    grid.generators.loc[grid.generators['type'] == 'solar', 'p_nom_extendable'] = False
    grid.generators.loc[grid.generators['type'] == 'solar', 'p_max_pu'] = 1

    # Kosten für Kabel müssen wahrscheinlich allgemein gesetzt werden
    grid.lines.loc[grid.lines['carrier'] == 'AC', 'capital_cost'] = 50
    grid.lines.loc[grid.lines['carrier'] == 'AC', 's_nom_max'] = 1000
    grid.lines.loc[grid.lines['carrier'] == 'AC', 's_nom'] = 100
    grid.lines.loc[grid.lines['carrier'] == 'AC', 'r'] = 0.001
    grid.lines.loc[grid.lines['carrier'] == 'AC', 'x'] = 0.01
    grid.lines.loc[grid.lines['carrier'] == 'AC', 's_nom_extendable'] = True


    # Carrier müssen am Ende allgemein gesetzt werden
    grid.add("Carrier", "gas", co2_emissions=0, color="orange")
    grid.add("Carrier", "solar", co2_emissions=0, color="yellow")
    grid.add("Carrier", "wind", co2_emissions=0, color="cyan")
    grid.add("Carrier", "battery", co2_emissions=0, color="gray")
    grid.add("Carrier", "AC", co2_emissions=0, color="black")  # Für Busse, Lasten, Leitungen


    # Allen Tranformatoren einen Generator hinzufügen, um Strom von außerhalb zuzulassen

    for i, trafo in grid.transformers.iterrows():
        # lv bus extrahieren
        bus_lv = trafo['bus1']  
        # name setzen
        gen_name = f"Generator_am_{i}"

        grid.add("Generator",
                name=gen_name,
                bus=bus_lv,
                carrier="gas",
                p_nom=100,
                p_nom_extendable=True,
                capital_cost=500,
                marginal_cost=50,
                efficiency=0.4)
        
    return grid

