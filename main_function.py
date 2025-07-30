
import functions as func
import ding0_grid_generators as ding0
import data_combination as dc
import pandas as pd
import numpy as np
from tqdm import tqdm
import os



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
    pd_Zensus_Bevoelkerung_100m.rename(columns={"Einwohner": "Einwohner_sum"}, inplace=True)

    pd_Zensus_Durchschn_Nettokaltmiete_100m = pd.read_csv(ordner + "/Zensus2022_Durchschn_Nettokaltmiete_100m-Gitter.csv", sep=";")
    pd_Zensus_Durchschn_Nettokaltmiete_100m.rename(columns={"durchschnMieteQM": "durchschnMieteQM_mw"}, inplace=True)

    pd_Zensus_Eigentuemerquote_100m = pd.read_csv(ordner + "/Zensus2022_Eigentuemerquote_100m-Gitter.csv", sep=";")
    pd_Zensus_Eigentuemerquote_100m.rename(columns={"Eigentuemerquote": "Eigentuemerquote_mw"}, inplace=True)

    pd_Zensus_Heizungsart_100m = pd.read_csv(ordner + "/Zensus2022_Heizungsart_100m-Gitter.csv", sep=";", encoding="cp1252")
    pd_Zensus_Heizungsart_100m.rename(columns={"Insgesamt_Heizungsart": "Insgesamt_Heizungsart_sum",
                                               "Fernheizung": "Fernheizung_sum",
                                               "Etagenheizung": "Etagenheizung_sum",
                                               "Blockheizung": "Blockheizung_sum",
                                               "Zentralheizung": "Zentralheizung_sum",
                                               "Einzel_Mehrraumoefen": "Einzel_Mehrraumoefen_sum",
                                               "keine_Heizung": "keine_Heizung_sum"}, inplace=True)

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


def osm_data(net, bbox_neu, buffer):
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
    #%% osm Data abrufen
    Area, Area_features = func.get_osm_data(bbox_osm)
    # Speichern der OSM Daten
    Area_features.to_file("Area_features.geojson", driver="GeoJSON")

    Area_features_df = Area_features.reset_index()

    #%% Daten kombinieren
    net = dc.data_combination(net, Area_features_df)

    return net, Area, Area_features


def daten_zuordnung(net, bundesland_data, zensus_data):
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
    net.buses = func.bundesland(net.buses, bundesland_data)

    # Zensus Daten
    net.buses = func.daten_zuordnen(net.buses, zensus_data)

    return net



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




def technik_zuordnen(grid, factors, kategorien_eigenschaften, Bev_data_Zensus, Bev_data_Technik, technik_arr):
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

    Bev_data = pd.merge(Bev_data_Zensus, Bev_data_Technik, on="GEN", how="left")

    df_land = grid.buses['lan_name'].copy().to_frame()

    df_zensus = grid.buses[[col for col in grid.buses.columns if col.startswith("Zensus")]].copy()
    # Kommas durch Punkte ersetzen, damit pd.to_numeric klappt; In float umwandeln; Fehlende Werte auf 0 setzen
    df_zensus = (df_zensus.astype(str).replace(",", ".", regex=True).apply(pd.to_numeric, errors="coerce").fillna(0.0))

    df_eigenschaften = pd.concat([df_land, df_zensus], axis=1)

    factor_bbox = np.array([0.0] * len(technik_arr))
    for i, technik in enumerate(technik_arr):
        factors_technik = factors[technik]
        grid.buses['Factor_' + technik], factor_bbox[i] = func.calculate_factors(df_eigenschaften, factors_technik, kategorien_eigenschaften, Bev_data, technik)

    return grid, factor_bbox



def technik_fill(grid, Technik, p_total):
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
        grid.buses = func.technik_sortieren(grid.buses, tech, p)

    grid.buses = func.storage(grid.buses)
    
    return grid



def loads_zuordnen(grid):
    
    # Powerflow zu bestehendem Netz
    grid.pf()

    # Zeit sollte auch schon integriert sein, wenn Lasten schon im Netz sind
    grid.set_snapshots(range(24))  # z.B. 24 Stunden

    # Lasten sollten schon im Netz integriert sein, hier nur Beispielwerte
    grid.add("Load", "Load_1",
                bus="BranchTee_mvgd_36165_lvgd_1884820002_building_28969207",       # in MW (bei Zeitreihe: Liste oder Serie)
                carrier="AC", 
                p__set = pd.Series([8,7,6,0,4,5,6,7,9,10,12,0,15,13,11,10,0,8,7,6,5,5,6,7], index=grid.snapshots))

    grid.add("Load", "Load_2",
                bus="BranchTee_mvgd_36165_lvgd_1884820002_28",       # in MW (bei Zeitreihe: Liste oder Serie)
                carrier="AC",
                p_set = pd.Series([18,17,16,15,14,15,16,17,19,20,22,24,25,23,21,20,19,18,17,16,15,15,16,17], index=grid.snapshots))

    grid.add("Load", "Load_3",
                bus="BranchTee_mvgd_36165_lvgd_1884820002_building_28969655",       # in MW (bei Zeitreihe: Liste oder Serie)
                carrier="AC", 
                p__set = pd.Series([8,7,0,5,4,0,6,0,9,0,12,0,15,0,11,0,9,8,7,6,5,5,6,7], index=grid.snapshots))

    grid.add("Load", "Load_4",
                bus="BranchTee_mvgd_36165_lvgd_1884820002_building_28969221",       # in MW (bei Zeitreihe: Liste oder Serie)
                carrier="AC",
                p_set = pd.Series([18,0,0,0,14,15,16,17,19,20,22,0,25,0,21,20,19,18,17,16,15,15,16,17], index=grid.snapshots))

    grid.add("Load", "Load_5",
                bus="BranchTee_mvgd_36165_lvgd_1884820002_building_28968489",       # in MW (bei Zeitreihe: Liste oder Serie)
                carrier="AC", 
                p__set = pd.Series([8,7,6,5,4,5,6,7,0,0,0,0,15,0,11,10,9,8,7,6,5,5,6,7], index=grid.snapshots))

    grid.add("Load", "Load_6",
                bus="BranchTee_mvgd_36165_lvgd_1884820002_building_28969176",       # in MW (bei Zeitreihe: Liste oder Serie)
                carrier="AC",
                p_set = pd.Series([18,17,16,15,0,0,0,0,0,20,22,24,25,23,0,0,0,0,17,16,15,15,16,17], index=grid.snapshots))

    grid.add("Load", "Load_7",
                bus="BranchTee_mvgd_36165_lvgd_1884820002_building_28969592",       # in MW (bei Zeitreihe: Liste oder Serie)
                carrier="AC", 
                p__set = pd.Series([8,7,0,50,0,0,0,7,9,10,12,14,15,13,11,10,0,0,7,6,5,5,6,7], index=grid.snapshots))

    grid.add("Load", "Load_8",
                bus="BranchTee_mvgd_36165_lvgd_1884820002_building_28969224",       # in MW (bei Zeitreihe: Liste oder Serie)
                carrier="AC",
                p_set = pd.Series([18,17,0,0,0,0,0,0,0,0,0,0,0,0,201,20,0,0,0,0,0,15,16,17], index=grid.snapshots))

    grid.add("Load", "Load_9",
                bus="BranchTee_mvgd_36165_lvgd_1884820002_building_28969634",       # in MW (bei Zeitreihe: Liste oder Serie)
                carrier="AC", 
                p__set = pd.Series([8,7,6,5,4,5,6,7,9,10,12,14,15,13,11,0,0,0,0,0,0,0,0,0], index=grid.snapshots))

    grid.add("Load", "Load_10",
                bus="BranchTee_mvgd_36165_lvgd_1884820002_18",       # in MW (bei Zeitreihe: Liste oder Serie)
                carrier="AC",
                p_set = pd.Series([0,0,0,0,0,0,0,0,0,0,202,0,0,0,0,0,0,0,0,0,0,150,0,17], index=grid.snapshots))

    grid.add("Load", "Load_11",
                bus="BranchTee_mvgd_36165_lvgd_1884820002_14",       # in MW (bei Zeitreihe: Liste oder Serie)
                carrier="AC", 
                p__set = pd.Series([8,7,6,5,4,5,6,7,9,10,12,14,15,13,11,10,9,8,7,6,5,5,6,7], index=grid.snapshots))

    grid.add("Load", "Load_12",
                bus="BranchTee_mvgd_36165_lvgd_1884820002_building_28969253",       # in MW (bei Zeitreihe: Liste oder Serie)
                carrier="AC",
                p_set = pd.Series([18,17,16,15,14,15,16,17,19,20,22,24,25,23,21,20,19,18,17,16,15,15,16,17], index=grid.snapshots))

    grid.add("Load", "Load_13",
                bus="BranchTee_mvgd_36165_lvgd_1884820002_building_28969616",       # in MW (bei Zeitreihe: Liste oder Serie)
                carrier="AC", 
                p__set = pd.Series([8,7,6,5,4,5,6,7,9,10,12,14,15,13,11,10,9,8,7,6,5,5,6,7], index=grid.snapshots))

    grid.add("Load", "Load_14",
                bus="BranchTee_mvgd_36165_lvgd_1884820002_71",       # in MW (bei Zeitreihe: Liste oder Serie)
                carrier="AC",
                p_set = pd.Series([18,17,16,15,14,15,16,17,19,20,22,24,25,23,21,20,19,18,17,16,15,15,16,17], index=grid.snapshots))

    grid.add("Load", "Load_15",
                bus="BranchTee_mvgd_36165_lvgd_1884820002_building_28969777",       # in MW (bei Zeitreihe: Liste oder Serie)
                carrier="AC", 
                p__set = pd.Series([8,7,6,5,4,5,6,7,9,0,12,14,105,13,11,10,9,8,7,6,5,5,6,7], index=grid.snapshots))

    grid.add("Load", "Load_16",
                bus="BranchTee_mvgd_36165_lvgd_1884820002_building_28968480",       # in MW (bei Zeitreihe: Liste oder Serie)
                carrier="AC",
                p_set = pd.Series([18,17,16,15,0,0,0,0,19,20,22,24,0,23,0,20,19,18,17,16,15,15,16,17], index=grid.snapshots))

    grid.loads['carrier'] = 'AC'

    # Wetter sollte schon hinzugefügt sein, hier nur Beispielwerte
    solar_profile = pd.Series([0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0.8, 0, 0, 0, 0,
                            0, 0, 0, 0], index=grid.snapshots)
    for gen in grid.generators.index[grid.generators['type'] == 'solar']:
        grid.generators_t[gen, 'p_max_pu'] = solar_profile

        
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

