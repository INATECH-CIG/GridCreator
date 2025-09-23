
import functions as func
import ding0_grid_generators as ding0
import data_combination as dc
import pandas as pd
import numpy as np
from tqdm import tqdm
from data import agg_dict
import os
import demand_and_load as demand_load
from pycity_base.classes.timer import Timer
from pycity_base.classes.weather import Weather
from pycity_base.classes.prices import Prices
from pycity_base.classes.environment import Environment

# import für typehints
import pypsa
import geopandas as gpd


def daten_laden(ordner: str) -> pd.DataFrame:
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


def ding0_grid(bbox: list[float], grids_dir: str, output_file_grid: str) -> tuple[ding0.Grid, list[float]]:
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


def osm_data(net: pypsa.Network, buses_df: pd.DataFrame, bbox_neu: list[float], buffer: float) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
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


def daten_zuordnung(buses: pd.DataFrame, bundesland_data: pd.DataFrame, ordner: str) -> pd.DataFrame:
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
    buses = func.zensus_ID(buses, ordner)

    # Alten Index sichern
    buses = buses.reset_index() 

    buses = func.zensus_laden(buses, ordner)

    # Alten Index wiederherstellen
    buses.set_index("Bus", inplace=True)

    return buses



def bundesland_zensus(ordner: str, datei: str, bundesland: str) -> pd.DataFrame:
    """
    Lädt die Bundesland-Daten aus einer GeoJSON-Datei und ordnet sie den Zensusdaten zu.
    
    Args:
        zensus (pd.DataFrame): DataFrame mit den Zensusdaten.
        datei (str): Pfad zur GeoJSON-Datei mit den Bundesland-Daten.

    Returns:
        pd.DataFrame: DataFrame mit den zugeordneten Bundesland-Daten.
    """

    lan = func.load_shapes(datei, bundesland)

    buses = pd.DataFrame()
    buses["GITTER_ID_100m"] = func.bundesland_zuordnung(ordner, lan)

    df = func.zensus_laden(buses, ordner)

    df = func.bundesland_summieren(df, bundesland)

    df = df.T.reset_index().rename(columns={"index": "GEN"}) 

    return df




def technik_zuordnen(buses: pd.DataFrame, file_Faktoren: str, file_solar: str, file_ecar: str, file_hp: str, technik_arr: list[str]) -> tuple[pd.DataFrame, list[float]]:
    """
    Ordnet den Bussen im Netzwerk verschiedene Techniken basierend auf Zensus- und Bevölkerungsdaten zu.
    
    Args:
        buses (pd.DataFrame): DataFrame mit den Busdaten.
        file_Faktoren (str): Pfad zur CSV-Datei mit den Technik-Faktoren.
        file_solar (str): Pfad zur CSV-Datei mit den Solar-Bevölkerungsdaten.
        file_ecar (str): Pfad zur CSV-Datei mit den E-Car-Bevölkerungsdaten.
        file_hp (str): Pfad zur CSV-Datei mit den Wärmepumpen-Bevölkerungsdaten.
        technik_arr (list[str]): Liste der Techniken, die zugeordnet werden sollen.
    
    Returns:
        tuple: Ein Tupel bestehend aus dem aktualisierten DataFrame mit den Busdaten und einer Liste der berechneten Faktoren für jede Technik.
    """
    
    
    Technik_Faktoren = pd.read_csv(file_Faktoren, sep=";")
    Technik_Faktoren = Technik_Faktoren.set_index("Technik")

    Bev_data_solar = pd.read_csv(file_solar, sep=",")
    Bev_data_solar["PLZ"] = Bev_data_solar["PLZ"].astype(int).astype(str).str.zfill(5)
    Bev_data_solar.set_index("PLZ", inplace=True)


    Bev_data_ecar = pd.read_csv(file_ecar, sep=",")
    Bev_data_ecar.set_index("id_5km", inplace=True)


    Bev_data_hp = pd.read_csv(file_hp, sep=",")
    Bev_data_hp.set_index("GEN", inplace=True)

    #Bev_data_Zensus.set_index("GEN", inplace=True)

    buses_zensus = buses[[col for col in buses.columns if col.startswith("Zensus")]].copy()
    buses_zensus.drop(columns=["Zensus_Einwohner"], inplace=True)
    # Kommas durch Punkte ersetzen, damit pd.to_numeric klappt; In float umwandeln; Fehlende Werte auf 0 setzen
    buses_zensus = (buses_zensus.astype(str).replace(",", ".", regex=True).apply(pd.to_numeric, errors="coerce").fillna(0.0))
    bbox_zensus = buses_zensus.agg(agg_dict)

    

    factor_bbox = np.array([0.0] * len(technik_arr))
    
    # Berechnung Solar
    if 'solar' in technik_arr:
        buses_plz = buses['plz_code'].copy().to_frame()
        # buses_plz.reset_index(drop=True, inplace=True)
        print(buses_plz)
        technik = 'solar'
        i = technik_arr.index(technik)
        buses['Factor_' + technik], factor_bbox[i]  = func.faktoren(buses_zensus, Technik_Faktoren, Bev_data_solar, bbox_zensus, 'solar', buses_plz)


    # Berechnung E-Car

    '''
    Die ID muss den buses noch hinzugefügt werden
    '''
    if 'E_car' in technik_arr:
        buses_5km = func.raster_5_id(buses)
        technik = 'E_car'
        i = technik_arr.index(technik)
        buses['Factor_' + technik], factor_bbox[i]  = func.faktoren(buses_zensus, Technik_Faktoren, Bev_data_ecar, bbox_zensus, 'E_car', buses_5km)



    # Berechnung HP
    if 'HP' in technik_arr:
        buses_land = buses['lan_name'].copy().to_frame()
        technik = 'HP'
        i = technik_arr.index(technik)
        buses['Factor_' + technik], factor_bbox[i]  = func.faktoren(buses_zensus, Technik_Faktoren, Bev_data_hp, bbox_zensus, 'HP', buses_land)


    return buses, factor_bbox



def technik_fill(buses: pd.DataFrame, Technik: list[str], p_total: list[float]) -> pd.DataFrame:
    """
    Füllt das Grid-Objekt mit den Techniken basierend auf den gegebenen Anteilen.
    
    Args:
        grid: Das Grid-Objekt, das die Buslinien enthält.
        Technik (list): Liste der Techniken, die zugeordnet werden sollen.
        p_total (list): Liste der Anteile für jede Technik.
        
    Returns:
        grid: Das aktualisierte Grid-Objekt mit den zugeordneten Techniken.
    """

    pv_plz = pd.read_csv("input/mastr_values_per_plz.csv", sep=",").set_index("PLZ")
    plz = int(buses['plz_code'].values[0])
    solar_power = pv_plz.loc[plz, 'Solar_Installed_Capacity_[MW]']

    for tech, p in zip(Technik, p_total):
        buses = func.technik_sortieren(buses, tech, p, solar_power)

    buses = func.solar_ausrichtung(buses, plz, pv_plz)

    storage_pv = pv_plz.loc[plz, 'Storage_per_PV']
        
    buses = func.storage(buses, storage_pv)

    return buses



def loads_zuordnen(grid: pd.DataFrame, buses: pd.DataFrame, bbox: pd.DataFrame, env=None) -> pypsa.Network:
    """
    Fügt dem PyPSA-Netzwerk Lasten, Solargeneratoren und Wärmepumpen basierend auf den Busdaten hinzu.
    
    Args:
        grid (pypsa.Network): Das PyPSA-Netzwerk, dem die Lasten und Generatoren hinzugefügt werden sollen.
        buses (pd.DataFrame): DataFrame mit den Busdaten.
        bbox (list): Liste mit den Koordinaten der Bounding Box in der Form [left, bottom, right, top].
        env (Environment, optional): Die Umgebung, die für die Last- und Generatorerstellung verwendet wird. Wenn None, wird eine neue Umgebung erstellt.
        
    Returns:
        pypsa.Network: Das aktualisierte PyPSA-Netzwerk mit den hinzugefügten Lasten und Generatoren.
    """

    if env is None:
        environment = func.env_wetter(bbox)
    else:
        environment = env
        

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
    # Liste zum Hinzufügen von Loads
    load_cols = {}
    # Hinzufügen von buses
    e_auto_buses = buses.index[buses["Power_E_car"].notna()]
    e_auto_cols = {}
    
    # for bus in buses.index:
    #     print(f"Prüfe, ob Load für {bus} existiert...")
    #     existing = grid.loads[(grid.loads['bus'] == bus)]
    #     """
    #     Doch alle Loads erste entfehrnen, dann alle neu hinzufügen?
    #     Würde e-auto erleichtern
    #     """
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
    #                 carrier="AC")
    #         print(f"Load {bus}_load jetzt hinzugefügt.")
    #         load_cols[bus + "_load"] = power

    #         if bus in e_auto_buses:
    #                     power = demand_load.create_e_car(occ = occupants, index=snapshots, env=environment)
    #                     grid.add("StorageUnit",
    #                             name=bus + "_E_Auto",
    #                             bus=bus,
    #                             carrier="E_Auto")
    #                     e_auto_cols[bus + "_E_Auto"] = power
    #                     print(f"Generator {bus}_E_Auto hinzugefügt.")

    #     else:
    #         print(f"Load für {bus} existiert bereits.")

    # # Alle neuen Spalten zu p_max_pu hinzufügen
    # grid.loads_t.p_set = pd.concat([grid.loads_t.p_set, pd.DataFrame(load_cols)], axis=1)
    # grid.storage_units_t.p = pd.concat([grid.storage_units_t.p, pd.DataFrame(e_auto_cols)], axis=1)

    print("Lasten und E-Auto hinzugefügt.")
    """
    Carrier und Type komplett egal?
    """

    """
    Alle Solargeneratoren erstmal löschen? Weil von ding0 und dann alle einheitlich?
    """
    #solar_buses = buses.index[buses["Power_solar"].notna()]
    solar_buses = buses.index[buses["Power_solar"] != 0]
    solar_cols = {}
    for bus in solar_buses:
        print(f"Prüfe, ob Generator für {bus} existiert...")
        existing = grid.generators[(grid.generators['bus'] == bus)]

        """
        Power für Solar ist immer 0, warum?
        Im Test gab es schöne Kurven
        """
        beta = buses.loc[bus, 'HauptausrichtungNeigungswinkel_Anteil']
        gamma = buses.loc[bus, 'Hauptausrichtung_Anteil']
        print(f"Beta: {beta}, Gamma: {gamma}")

        # Ost-Weste wird in zwei halbe PV aufgeteilt
        if gamma == 'Ost-West':
            gamma_1 = 'Ost'
            gamma_2 = 'West'
            power_1 = demand_load.create_pv(peakpower=100*buses.loc[bus, 'Power_solar']*0.5, beta=beta, gamma=gamma_1, index=snapshots, env=environment)
            power_2 = demand_load.create_pv(peakpower=100*buses.loc[bus, 'Power_solar']*0.5, beta=beta, gamma=gamma_2, index=snapshots, env=environment)

            power = power_1 + power_2

        else:
            power = demand_load.create_pv(peakpower=100*buses.loc[bus, 'Power_solar'], beta=beta, gamma=gamma, index=snapshots, env=environment)
        
        if existing.empty:
            # Generator hinzufügen
            
            grid.add("Generator",
                    name=bus + "_solar",
                    bus=bus,
                    carrier="solar",
                    type="solar",
                    p_nom=buses.loc[bus, 'Power_solar'])
            

            solar_cols[bus + "_solar"] = power.values



        else:
            # Load zu existierendem Generator hinzufügen
            solar_cols[bus + "_solar"] = power.values




    if solar_cols:
        grid.generators_t.p_max_pu = pd.concat([grid.generators_t.p_max_pu, pd.DataFrame(solar_cols, index=snapshots)], axis=1)

    print("Solargeneratoren hinzugefügt.")

    #HP_amb_buses = buses.index[buses["Power_HP_ambient"].notna()]
    HP_buses = buses.index[buses["Power_HP"] != 0]
    hp_cols = {}
    for bus in HP_buses:
        # Generator hinzufügen
        power = demand_load.create_hp(index=snapshots, env=environment)

        
        grid.add("Generator",
                name=bus + "_HP",
                bus=bus,
                carrier="HP",
                type="HP")
        hp_cols[bus + "_HP"] = power.values


    if hp_cols:
        grid.generators_t.p_max_pu = pd.concat([grid.generators_t.p_max_pu, pd.DataFrame(hp_cols, index=snapshots)], axis=1)
    
    return grid

def pypsa_vorbereiten(grid: pypsa.Network) -> pypsa.Network:
    """
    Bereitet das PyPSA-Netzwerk vor, indem es Kosten und Eigenschaften für verschiedene Komponenten setzt.

    Args:
        grid (pypsa.Network): Das PyPSA-Netzwerk, das vorbereitet werden soll.

    Returns:
        pypsa.Network: Das vorbereitete PyPSA-Netzwerk.
    """


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

