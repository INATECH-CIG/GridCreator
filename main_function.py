
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

    grid.export_to_netcdf(output_file_grid)

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
