#%% Import
 
import geopandas as gpd
from pyproj import Transformer
import pandas as pd
import polars as pl
import osmnx as ox
import scipy.spatial as spatial
import numpy as np
import json
from tqdm import tqdm
from functools import reduce
import numpy as np
import ast
import random
from data_old import agg_dict

import cdsapi
from scipy import datasets
import xarray as xr
import os

from pycity_base.classes.timer import Timer
from pycity_base.classes.weather import Weather
from pycity_base.classes.prices import Prices
from pycity_base.classes.environment import Environment

# Import für TypeHints
import pypsa

#%% Funktionen zum aufrufen


#%% Funktionen zum aufrufen

def get_osm_data(bbox: list) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Ruft OSM-Daten für die gegebene Bounding Box ab und gibt sie zurück.

    Args:
        bbox (list): Eine Liste mit den Koordinaten der Bounding Box in der Form [left, bottom, right, top].

    Returns:
        tuple: Ein Tupel bestehend aus den OSM-Daten und den OSM-Feature-Daten.
    """

    tags = {
    "aerialway": True,        # Seilbahnen
    "aerodrome": True,        # Flugplätze
    "amenity": True,          # Annehmlichkeiten
    "aeroway": True,          # Luftfahrt-Infrastruktur
    "barrier": True,          # Barrieren
     "building": True,         # Gebäude
    "bus_station": True,      # Bushaltestellen
    "church": True,           # Kirchen
    "craft": True,            # Handwerk
    "emergency": True,        # Notfallinfrastruktur
    "geological": True,       # Geologische Merkmale
    "healthcare": True,       # Gesundheitseinrichtungen
    "historic": True,         # Historische Orte
    "highway": True,          # Straßen
    "leisure": True,          # Freizeiteinrichtungen
    "landcover": True,        # Landbedeckung
    "landuse": True,          # Landnutzung
    "Man Made": True,         # Menschliche Strukturen
    "man_made": True,         # Menschliche Strukturen
    "military": True,         # Militärische Einrichtungen
    "natural": True,          # Natürliche Merkmale
    "office": True,           # Büros
    "place": True,            # Orte
    "power": True,            # Energieinfrastruktur
    "public_transport": True, # Öffentliche Verkehrsmittel
    "railway": True,          # Eisenbahninfrastruktur
    "route": True,            # Routen
    "shop": True,             # Geschäfte
    "school": True,           # Schulen
    "tourism": True,          # Touristische Orte
    'telecom': True,         # Telekommunikationseinrichtungen
    "waterway": True,         # Gewässer
    }
    Area = ox.graph.graph_from_bbox(bbox, network_type="all", retain_all=True)
    Area_features = ox.features.features_from_bbox(bbox, tags=tags)
    return Area, Area_features


def compute_bbox_from_buses(net: pypsa.Network) -> list[float]:
    """
    Compute the bounding box from the bus coordinates in a PyPSA network.

    Args:
        net (pypsa.Network): The PyPSA network object containing bus coordinates.

    Returns:
        list: A list containing the bounding box coordinates in the form [left, bottom, right, top].
    """
    x_min = net.buses['x'].min()
    x_max = net.buses['x'].max()
    y_min = net.buses['y'].min()
    y_max = net.buses['y'].max()

    # Optional als [left, bottom, right, top]
    bbox = [x_min, y_min, x_max, y_max]
    return bbox


def bundesland(net_buses: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """
    Weist den Bussen im Netzwerk Bundesland-Daten zu.

    Args:
        net_buses (pd.DataFrame): DataFrame mit den Busdaten des Netzwerks.
        data (pd.DataFrame): DataFrame mit den Bundesland-Daten.
    
    Returns:
        pd.DataFrame: DataFrame mit den zugeordneten Bundesland-Daten.
    """

    # 1. Referenzpunkte (bundesland) vorbereiten
    data["geo_point_2d"] = data["geo_point_2d"].apply(json.loads)
    ref_lon = data["geo_point_2d"].apply(lambda d: d["lon"])
    ref_lat = data["geo_point_2d"].apply(lambda d: d["lat"])
    ref_points = np.vstack((ref_lon, ref_lat)).T

    # 2. Zielpunkte (buses) vorbereiten
    bus_lon = net_buses["x"]
    bus_lat = net_buses["y"]
    bus_points = np.vstack((bus_lon, bus_lat)).T

    # 3. KD-Tree bauen & Zuordnung
    tree = spatial.cKDTree(ref_points)
    _, idx = tree.query(bus_points)

    # Daten hinzufügen
    zuordnen = ["lan_name", "plz_name", "plz_code", "krs_code", "lan_code", "krs_name"]
    for spalte in zuordnen:
        net_buses[spalte] = data.iloc[idx][spalte].values


    return net_buses


def zensus_ID(buses_df: pd.DataFrame, ordner: str) -> pd.DataFrame:
    """
    Ordnet den Bussen im Netzwerk die GITTER_ID_100m aus den Zensusdaten zu.
    
    Args:
        buses_df (pd.DataFrame): DataFrame mit den Busdaten des Netzwerks.
        ordner (str): Der Pfad zum Ordner, der die Zensusdaten im CSV-Format enthält.

    Returns:
        pd.DataFrame: DataFrame mit den zugeordneten GITTER_ID_100m.
    """

    zensus = pd.read_csv(ordner + "/Zensus2022_Bevoelkerungszahl_100m-Gitter.csv", sep=";")

    buses = buses_df

    x_array, y_array = epsg4326_zu_epsg3035(buses["x"], buses["y"])

    # Referenzpunkte in der Zensus-Tabelle
    reference_points = np.vstack((zensus['x_mp_100m'], zensus['y_mp_100m'])).T
    tree = spatial.cKDTree(reference_points)

    # Zielpunkte aus dem Netz
    query_points = np.vstack((x_array, y_array)).T

    # Nächste Nachbarn finden
    distances, indices = tree.query(query_points)



    columns = np.array(zensus["GITTER_ID_100m"])[indices]

    buses['GITTER_ID_100m'] = columns

    return buses


def zensus_laden(buses_df: pd.DataFrame, ordner: str) -> pd.DataFrame:
    """
    Lädt Zensusdaten aus CSV-Dateien in einem angegebenen Ordner und gibt sie als DataFrame zurück.
    
    Args:
        ordner (str): Der Pfad zum Ordner, der die Zensusdaten im CSV-Format enthält.
        
    Returns:
        pd.DataFrame: Ein DataFrame, das die kombinierten Zensusdaten enthält.
    """

    columns = buses_df['GITTER_ID_100m']


    # Manuell laden
    Zensus2022_Bevoelkerungszahl_100m = (pl.scan_csv(ordner + "/Zensus2022_Bevoelkerungszahl_100m-Gitter.csv", separator=";")
                                         .filter(pl.col("GITTER_ID_100m")
                                                 .is_in(columns)).select("GITTER_ID_100m",
                                                                         "Einwohner").collect()
                                        )
    Zensus2022_Bevoelkerungszahl_100m = Zensus2022_Bevoelkerungszahl_100m.to_pandas()
    Zensus2022_Bevoelkerungszahl_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Bevoelkerungszahl_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses_df = buses_df.merge(Zensus2022_Bevoelkerungszahl_100m, on="GITTER_ID_100m", how="left")


    Zensus2022_Durchschn_Nettokaltmiete_Anzahl_der_Wohnungen_100m = (pl.scan_csv(ordner + "/Zensus2022_Durchschn_Nettokaltmiete_Anzahl_der_Wohnungen_100m-Gitter.csv", separator=";")
                                                    .filter(pl.col("GITTER_ID_100m").is_in(columns))
                                                    .select("GITTER_ID_100m",
                                                            "durchschnMieteQM",
                                                            ).collect()
                                                )
    Zensus2022_Durchschn_Nettokaltmiete_Anzahl_der_Wohnungen_100m = Zensus2022_Durchschn_Nettokaltmiete_Anzahl_der_Wohnungen_100m.to_pandas()
    Zensus2022_Durchschn_Nettokaltmiete_Anzahl_der_Wohnungen_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Durchschn_Nettokaltmiete_Anzahl_der_Wohnungen_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses_df = buses_df.merge(Zensus2022_Durchschn_Nettokaltmiete_Anzahl_der_Wohnungen_100m, on="GITTER_ID_100m", how="left")




    Zensus2022_Groesse_des_privaten_Haushalts_100m = (pl.scan_csv(ordner + "/Zensus2022_Groesse_des_privaten_Haushalts_100m-Gitter.csv", separator=";")
                                                        .filter(pl.col("GITTER_ID_100m").is_in(columns))
                                                        .select("GITTER_ID_100m",
                                                                "1_Person",
                                                                "2_Personen",
                                                                "3_Personen",
                                                                "4_Personen",
                                                                "5_Personen",
                                                                "6_Personen_und_mehr"
                                                                ).collect()
                                                        )
    Zensus2022_Groesse_des_privaten_Haushalts_100m = Zensus2022_Groesse_des_privaten_Haushalts_100m.to_pandas()
    Zensus2022_Groesse_des_privaten_Haushalts_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Groesse_des_privaten_Haushalts_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses_df = buses_df.merge(Zensus2022_Groesse_des_privaten_Haushalts_100m, on="GITTER_ID_100m", how="left")






    Zensus2022_Staatsangehoerigkeit_Gruppen_100m = (pl.scan_csv(ordner + "/Zensus2022_Staatsangehoerigkeit_Gruppen_100m-Gitter.csv", separator=";")
                                                    .filter(pl.col("GITTER_ID_100m").is_in(columns))
                                                    .select("GITTER_ID_100m",
                                                            "EU27_Land",
                                                            ).collect()
                                                    )
    Zensus2022_Staatsangehoerigkeit_Gruppen_100m = Zensus2022_Staatsangehoerigkeit_Gruppen_100m.to_pandas()
    Zensus2022_Staatsangehoerigkeit_Gruppen_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Staatsangehoerigkeit_Gruppen_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses_df = buses_df.merge(Zensus2022_Staatsangehoerigkeit_Gruppen_100m, on="GITTER_ID_100m", how="left")


    Zensus2022_Typ_der_Kernfamilie_nach_Kindern_100m = (pl.scan_csv(ordner + "/Zensus2022_Typ_der_Kernfamilie_nach_Kindern_100m-Gitter.csv", separator=";")
                                                            .filter(pl.col("GITTER_ID_100m").is_in(columns))
                                                            .select("GITTER_ID_100m",
                                                                    "Ehep_Kinder_ab18",
                                                                    "NichtehelLG_mind_1Kind_unter18",
                                                                    ).collect()
                                                            )
    Zensus2022_Typ_der_Kernfamilie_nach_Kindern_100m = Zensus2022_Typ_der_Kernfamilie_nach_Kindern_100m.to_pandas()
    Zensus2022_Typ_der_Kernfamilie_nach_Kindern_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Typ_der_Kernfamilie_nach_Kindern_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses_df = buses_df.merge(Zensus2022_Typ_der_Kernfamilie_nach_Kindern_100m, on="GITTER_ID_100m", how="left")





    Zensus2022_Flaeche_der_Wohnung_10m2_Intervalle_100m = (pl.scan_csv(ordner + "/Zensus2022_Flaeche_der_Wohnung_10m2_Intervalle_100m-Gitter.csv", separator=";")
                                                    .filter(pl.col("GITTER_ID_100m").is_in(columns))
                                                    .select("GITTER_ID_100m",
                                                            "30bis39",
                                                            "40bis49",
                                                            ).collect()
                                                )
    Zensus2022_Flaeche_der_Wohnung_10m2_Intervalle_100m = Zensus2022_Flaeche_der_Wohnung_10m2_Intervalle_100m.to_pandas()
    Zensus2022_Flaeche_der_Wohnung_10m2_Intervalle_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Flaeche_der_Wohnung_10m2_Intervalle_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses_df = buses_df.merge(Zensus2022_Flaeche_der_Wohnung_10m2_Intervalle_100m, on="GITTER_ID_100m", how="left")

    Zensus2022_Wohnung_Gebaeudetyp_Groesse_100m = (pl.scan_csv(ordner + "/Zensus2022_Wohnung_Gebaeudetyp_Groesse_100m-Gitter.csv", separator=";")
                                                    .filter(pl.col("GITTER_ID_100m").is_in(columns))
                                                    .select("GITTER_ID_100m",
                                                            "Freist_ZFH",
                                                            ).collect()
                                                )
    Zensus2022_Wohnung_Gebaeudetyp_Groesse_100m = Zensus2022_Wohnung_Gebaeudetyp_Groesse_100m.to_pandas()
    Zensus2022_Wohnung_Gebaeudetyp_Groesse_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Wohnung_Gebaeudetyp_Groesse_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses_df = buses_df.merge(Zensus2022_Wohnung_Gebaeudetyp_Groesse_100m, on="GITTER_ID_100m", how="left")



    """
    Baujahr hatte Gitter_ID_100M !!!!
    in csv geändert
    """

    Zensus2022_Baujahr_JZ_100m = (pl.scan_csv(ordner + "/Zensus2022_Baujahr_JZ_100m-Gitter.csv", separator=";")
                                                    .filter(pl.col("GITTER_ID_100m").is_in(columns))
                                                    .select("GITTER_ID_100m",
                                                            "Vor1919",
                                                            "a1970bis1979",
                                                            ).collect()
                                                )
    Zensus2022_Baujahr_JZ_100m = Zensus2022_Baujahr_JZ_100m.to_pandas()
    Zensus2022_Baujahr_JZ_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Baujahr_JZ_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses_df = buses_df.merge(Zensus2022_Baujahr_JZ_100m, on="GITTER_ID_100m", how="left")


    Zensus2022_Gebaeude_nach_Anzahl_der_Wohnungen_100m = (pl.scan_csv(ordner + "/Zensus2022_Gebaeude_nach_Anzahl_der_Wohnungen_100m-Gitter.csv", separator=";")
                                                    .filter(pl.col("GITTER_ID_100m").is_in(columns))
                                                    .select("GITTER_ID_100m",
                                                            "1_Wohnung",
                                                            "2_Wohnungen",
                                                            "3bis6_Wohnungen",
                                                            "7bis12_Wohnungen",
                                                            "13undmehr_Wohnungen"
                                                            ).collect()
                                                )
    Zensus2022_Gebaeude_nach_Anzahl_der_Wohnungen_100m = Zensus2022_Gebaeude_nach_Anzahl_der_Wohnungen_100m.to_pandas()
    Zensus2022_Gebaeude_nach_Anzahl_der_Wohnungen_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Gebaeude_nach_Anzahl_der_Wohnungen_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses_df = buses_df.merge(Zensus2022_Gebaeude_nach_Anzahl_der_Wohnungen_100m, on="GITTER_ID_100m", how="left")

    Zensus2022_Geb_Gebaeudetyp_Groesse_100m = (pl.scan_csv(ordner + "/Zensus2022_Geb_Gebaeudetyp_Groesse_100m-Gitter.csv", separator=";")
                                                    .filter(pl.col("GITTER_ID_100m").is_in(columns))
                                                    .select("GITTER_ID_100m",
                                                            "FreiEFH",
                                                            ).collect()
                                                )
    Zensus2022_Geb_Gebaeudetyp_Groesse_100m = Zensus2022_Geb_Gebaeudetyp_Groesse_100m.to_pandas()
    Zensus2022_Geb_Gebaeudetyp_Groesse_100m.rename(columns={
                                           "FreiEFH": "Geb_FreiEFH",
                                                        }, inplace=True)
    Zensus2022_Geb_Gebaeudetyp_Groesse_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Geb_Gebaeudetyp_Groesse_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses_df = buses_df.merge(Zensus2022_Geb_Gebaeudetyp_Groesse_100m, on="GITTER_ID_100m", how="left")



    Zensus2022_Energietraeger_100m = (pl.scan_csv(ordner + "/Zensus2022_Energietraeger_100m-Gitter_utf8.csv", separator=";")
                                                    .filter(pl.col("GITTER_ID_100m").is_in(columns))
                                                    .select("GITTER_ID_100m",
                                                            "Gas",
                                                            "Holz_Holzpellets",
                                                            "Kohle",
                                                            ).collect()
                                                )
    Zensus2022_Energietraeger_100m = Zensus2022_Energietraeger_100m.to_pandas()
    Zensus2022_Energietraeger_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Energietraeger_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses_df = buses_df.merge(Zensus2022_Energietraeger_100m, on="GITTER_ID_100m", how="left")


    Zensus2022_Gebaeude_nach_Energietraeger_der_Heizung_100m = (pl.scan_csv(ordner + "/Zensus2022_Gebaeude_nach_Energietraeger_der_Heizung_100m-Gitter.csv", separator=";")
                                                    .filter(pl.col("GITTER_ID_100m").is_in(columns))
                                                    .select("GITTER_ID_100m",
                                                            "Holz_Holzpellets",
                                                            "Kohle",
                                                            "Fernwaerme",
                                                            ).collect()
                                                )
    Zensus2022_Gebaeude_nach_Energietraeger_der_Heizung_100m = Zensus2022_Gebaeude_nach_Energietraeger_der_Heizung_100m.to_pandas()
    Zensus2022_Gebaeude_nach_Energietraeger_der_Heizung_100m.rename(columns={
                                           "Holz_Holzpellets": "Geb_Holz_Holzpellets",
                                           "Kohle": "Geb_Kohle",
                                           "Fernwaerme": "Geb_Fernwaerme",
                                                        }, inplace=True)
    Zensus2022_Gebaeude_nach_Energietraeger_der_Heizung_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Gebaeude_nach_Energietraeger_der_Heizung_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses_df = buses_df.merge(Zensus2022_Gebaeude_nach_Energietraeger_der_Heizung_100m, on="GITTER_ID_100m", how="left")

    
    return buses_df


def epsg4326_zu_epsg3035(lon: float, lat: float) -> tuple[float, float] :
    """
    Umrechnung von WGS84 (EPSG:4326) zu ETRS89 (EPSG:3035)

    Args:
        lon (float or pd.Series): Längengrad in WGS84.
        lat (float or pd.Series): Breitengrad in WGS84.
    
    Returns:
        tuple: Ein Tupel mit den umgerechneten Koordinaten (rechtswert, hochwert) in ETRS89 (EPSG:3035).
    """

    transformer_zu_EPSG3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    rechtswert, hochwert = transformer_zu_EPSG3035.transform(lon, lat)
    return rechtswert, hochwert

def epsg32632_zu_epsg3035(lon: float, lat: float) -> tuple[float, float]:
    """
    Umrechnung von WGS84 (EPSG:32632) zu ETRS89 (EPSG:3035)
    Args:
        lon (float or pd.Series): Längengrad in WGS84.
        lat (float or pd.Series): Breitengrad in WGS84.
    
    Returns:
        tuple: Ein Tupel mit den umgerechneten Koordinaten (rechtswert, hochwert) in ETRS89 (EPSG:3035).
    """

    transformer_zu_EPSG3035 = Transformer.from_crs("EPSG:32632", "EPSG:3035", always_xy=True)
    rechtswert, hochwert = transformer_zu_EPSG3035.transform(lon, lat)
    return rechtswert, hochwert


def zensus_df_verbinden(daten_zensus: list) -> pd.DataFrame:
    """
    Verbindet mehrere DataFrames basierend auf der GITTER_ID_100m Spalte.
    
    Args:
        daten_zensus (list): Eine Liste von DataFrames, die auf GITTER_ID_100m basieren.

    Returns:
        pd.DataFrame: Ein DataFrame, das die Daten aus allen DataFrames kombiniert.
    """

    # Sicherstellen, dass alle DFs die gleichen drei Schlüsselspalten haben
    basis_spalten = ["GITTER_ID_100m", "x_mp_100m", "y_mp_100m"]

    # DataFrames per GITTER_ID_100m zusammenführen
    df_vereint = reduce(lambda left, right: pd.merge(left, right, on=basis_spalten, how='outer'), daten_zensus)
    df_vereint = df_vereint.rename(columns={col: f"Zensus_{col}" for col in df_vereint.columns if col not in basis_spalten})
    return df_vereint




def daten_zuordnen(net_buses: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """
    Ordnet die Zensusdaten den Bussen im Netzwerk zu, basierend auf den Koordinaten.
    
    Args:
        net_buses (pd.DataFrame): DataFrame mit den Busdaten des Netzwerks.
        data (pd.DataFrame): DataFrame mit den Zensusdaten.
        
    Returns:
        pd.DataFrame: DataFrame mit den zugeordneten Zensusdaten.
    """

    x_array, y_array = epsg4326_zu_epsg3035(net_buses["x"], net_buses["y"])

    # Referenzpunkte in der Zensus-Tabelle
    reference_points = np.vstack((data['x_mp_100m'].values, data['y_mp_100m'].values)).T
    tree = spatial.cKDTree(reference_points)

    # Zielpunkte aus dem Netz
    query_points = np.vstack((x_array, y_array)).T

    # Nächste Nachbarn finden
    distances, indices = tree.query(query_points)

    # Spaltennamen ändern
    # data = data.rename(columns=lambda col: f"Zensus_{col}").copy()
    
    # Passende Zeilen aus data holen
    matched_data = data.iloc[indices].copy()

    # Index an net_buses anpassen, damit concat funktioniert
    matched_data.index = net_buses.index

    # DataFrames nebeneinander anfügen
    result = pd.concat([net_buses, matched_data], axis=1)

    return result



def load_shapes(datei: str, bundesland: str) -> gpd.GeoDataFrame:
    """
    Lädt die Bundesland-Geometrien aus einer GeoJSON-Datei und filtert sie auf die relevanten Bundesländer.
    Args:
        datei (str): Pfad zur GeoJSON-Datei mit den Bundesland-Geometrien.
        
    Returns:
        gpd.GeoDataFrame: Ein GeoDataFrame mit den Geometrien der relevanten Bundesländer.
    """

    # Laden der Geometrien aus der Datei
    shapes = gpd.read_file(datei, layer="vg5000_lan")
    
    # Filtern der Geometrien nach Bundesländern
    land = shapes[shapes["GEN"].isin([bundesland])]
    # Filtern der Geometrien
    land_aggregiert = land.dissolve(by="GEN")
    land_umrisse = land_aggregiert[["geometry"]]
    land_3035 = land_umrisse.to_crs(epsg=3035)

    return land_3035


# def sum_mw(df):
#     """
#     Summiert die Werte in einem DataFrame und berechnet den Mittelwert für Spalten,
#     die mit '_mw' enden. Für andere Spalten wird die Summe berechnet
    
#     Args:
#         df (pd.DataFrame): DataFrame mit den zu summierenden Werten.
        
#     Returns:
#         pd.DataFrame: Ein DataFrame mit den summierten Werten.
#     """

#     result = pd.DataFrame(columns=df.columns)
#     row = {}

#     for col in df.columns:
#         values = df[col]
#         if col.endswith('_mw'):
#             row[col] = values.mean()
#         else:
#             row[col] = values.sum()

#     result.loc[0] = row
#     return result



def bundesland_zuordnung(ordner: str, shapes: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Ordnet die Zensusdaten den Bundesländern zu, basierend auf den Geometrien der Bundesländer.
    Args:
        zensus (pd.DataFrame): DataFrame mit den Zensusdaten.
        shapes (gpd.GeoDataFrame): GeoDataFrame mit den Geometrien der Bundesländer.
        
    Returns:
        pd.DataFrame: Ein DataFrame mit den zugeordneten Bundesland-Daten."""

    # Zensusdaten laden
    zensus = pd.read_csv(ordner + "/Zensus2022_Bevoelkerungszahl_100m-Gitter.csv", sep=";")
    
    # Punkte-Geometrie erstellen
    geometry = gpd.GeoSeries(gpd.points_from_xy(zensus['x_mp_100m'], zensus['y_mp_100m']))
    
    # Boolean-Maske: welche Punkte liegen innerhalb der Bundesländer
    mask = geometry.within(shapes.unary_union)
    
    # Nur diese Zeilen auswählen
    zensus_filtered = zensus[mask].copy()

    gitter_id = zensus_filtered["GITTER_ID_100m"]

    return gitter_id



def bundesland_summieren(df: pd.DataFrame, bundesland: str) -> pd.DataFrame:
    """
    Summiert die Zensusdaten für ein bestimmtes Bundesland.
    
    Args:
        df (pd.DataFrame): DataFrame mit den Zensusdaten.
        bundesland (str): Der Name des Bundeslands, für das die Daten summiert werden
        
    Returns:
        pd.DataFrame: Ein DataFrame mit den summierten Zensusdaten für das Bundesland
    """

    
    numerische_spalten = [col for col in df.columns if col.startswith("Zensus")]

    for col in numerische_spalten:
        # Komma durch Punkt ersetzen, Strings in float konvertieren
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
        # Ungültige Zeichen wie "–" in NaN umwandeln
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # NaN durch 0 ersetzen
        df[col] = df[col].fillna(0)

    print(df)
    df_agg = df[numerische_spalten].agg(agg_dict)
    df_neu = pd.DataFrame()
    df_neu[bundesland] = df_agg

    return df_neu


def gewichtungsfaktor(land: str, kategorien_eigenschaften: pd.DataFrame, factors: dict, row: pd.Series, technik_faktoren: pd.Series, gesamtgewicht_dict: dict) -> float:
    """
    Berechnet den Gewichtungsfaktor für eine bestimmte Technik und ein bestimmtes Land basierend auf den Eigenschaften und Faktoren.
    
    Args:
        land (str): Der Name des Landes, für das der Faktor berechnet wird.
        kategorien_eigenschaften (pd.DataFrame): DataFrame mit den Kategorien der Eigenschaften.
        factors (dict): Dictionary mit den Faktoren für jede Kategorie.
        row (pd.Series): Eine Zeile des DataFrames, die die Eigenschaften enthält.
        technik_faktoren (pd.Series): Serie mit den Technik-Faktoren für das Land.
        gesamtgewicht_dict (dict): Dictionary mit den Gesamtgewichten für jede Kategorie und Land.
        
    Returns:
        float: Der berechnete Gewichtungsfaktor für die Technik im angegebenen Land.
    """

    weighted_sum = 1.0
    # Iteriere über die Kategorien
    for kat in kategorien_eigenschaften.columns:
        # Filtere die Eigenschaften für die aktuelle Kategorie
        eigenschaften = kategorien_eigenschaften[kat].dropna()

        # hole nur die relevanten Faktoren und Werte
        faktoren_kat = [factors[attr] for attr in eigenschaften]
        werte_kat = row[eigenschaften]

        # numerische Multiplikation
        sum_bus = (werte_kat * faktoren_kat).sum()
        sum_land = gesamtgewicht_dict.get((land, kat), 0)

        # Berechnung von Gewichtungsprodukt
        if sum_land != 0:
            weighted_sum *= sum_bus / sum_land
        else:
            weighted_sum *= 0

    print("Technik Faktoren:", technik_faktoren)

    return weighted_sum * float(technik_faktoren.iloc[0])




def calculate_factors(df: pd.DataFrame, factors: dict, kategorien_eigenschaften: pd.DataFrame, Bev_data: pd.DataFrame, technik: str) -> pd.DataFrame:

    """
    Berechnet den Faktor für jede Zeile im DataFrame basierend auf den Eigenschaften und Bevölkerungsdaten.

    Parameters:
        df (pd.DataFrame): DataFrame mit den Eigenschaften der Busse.
        factors (dict): Dictionary mit den Faktoren für die Eigenschaften.
        kategorien_eigenschaften (pd.DataFrame): DataFrame mit den Eigenschaften, die gefiltert werden sollen.
        Bev_data (pd.DataFrame): DataFrame mit den Bevölkerungsdaten.
        technik (str): Technik, für die der Faktor berechnet werden soll.

    Returns:
        pd.DataFrame: DataFrame mit den berechneten Faktoren für die jeweilige Technik.
    """

    #Bev_data = Bev_data.set_index('GEN')
    #  Extrahiere die Technik-Faktoren für jedes Bundesland -> verhindert mehrfaches Suchen in For Loop
    technik_faktoren = Bev_data[technik]

    # Dictionary für die Gesamtgewichte -> verhindert mehrfaches Suchen in For Loop
    gesamtgewicht_dict = {}
    for kat in kategorien_eigenschaften.columns:
        eigenschaften = kategorien_eigenschaften[kat].dropna()
        faktoren_kat = [factors[attr] for attr in eigenschaften]
        
        for land in Bev_data.index:
            land_werte = Bev_data.loc[land, eigenschaften]
            gewicht = (land_werte * faktoren_kat).sum()
            gesamtgewicht_dict[(land, kat)] = gewicht

    # Leeres DataFrame mit Zeilen wie df, für jedes Land eine Spalte
    faktor_pro_bus = pd.Series(index=df.index, dtype=float)

    # Iteriere über jeden Bus im DataFrame
    for idx, row in df.iterrows():
        land = row['lan_name']

        faktor_pro_bus.at[idx] = gewichtungsfaktor(land, kategorien_eigenschaften, factors, row, technik_faktoren, gesamtgewicht_dict)


    # Berechnung für Bundesland
    summen = df.iloc[:, 1:].agg(agg_dict)
    erste_spalte = {df.columns[0]: df.iloc[0, 0]}
    neue_zeile = {**erste_spalte, **summen.to_dict()}    
    zeile = pd.DataFrame([neue_zeile])

    land_bbox = zeile.iloc[0]['lan_name']
    faktor_bbox = gewichtungsfaktor(land_bbox, kategorien_eigenschaften, factors, zeile.iloc[0], technik_faktoren, gesamtgewicht_dict)

    return faktor_pro_bus, faktor_bbox



def zulassung(net_buses: pd.DataFrame, pfad: str) -> pd.DataFrame:
    """
    Ordnet den Bussen im Netzwerk die ID_5km aus den 5km Rasterdaten zu.
    
    Args:
        net_buses (pd.DataFrame): DataFrame mit den Busdaten des Netzwerks.
        
    Returns:
        pd.DataFrame: DataFrame mit den zugeordneten ID_5km.
    """


    data = gpd.read_file(f"{pfad}/input/FZ Pkw mit Elektroantrieb Zulassungsbezirk_-8414538009745447927.geojson")

    # Bus Koordinaten in GeoDataFrame umwandeln
    gdf_buses = gpd.GeoDataFrame(net_buses.copy(), geometry=gpd.points_from_xy(net_buses["x"], net_buses["y"]), crs=data.crs)

    # Shape, Punkt Join
    gdf_joined = gpd.sjoin(gdf_buses, data[["Schluessel_Zulbz", "geometry"]], how="left", predicate="within")
    print("gdf_joined:", gdf_joined)
    # Duplikate droppen
    gdf_joined = gdf_joined[~gdf_joined.index.duplicated(keep='first')]
    # ID filtern für jeden Bus
    id = gdf_joined["Schluessel_Zulbz"].copy()

    return id


def faktoren(buses_zensus: pd.DataFrame, Technik_Faktoren: pd.DataFrame, Bev_data: pd.DataFrame, bbox_zensus: pd.Series, technik: str, id_df: pd.Series) -> tuple[pd.Series, float]:
        """
        Berechnet die Faktoren für die Busse und das Bounding Box basierend auf den Zensusdaten und Bevölkerungsdaten.
        
        Args:
            buses_zensus (pd.DataFrame): DataFrame mit den Zensusdaten der Busse.
            Technik_Faktoren (pd.DataFrame): DataFrame mit den Technik-Faktoren.
            Bev_data (pd.DataFrame): DataFrame mit den Bevölkerungsdaten.
            bbox_zensus (pd.Series): Serie mit den Zensusdaten der Bounding Box.
            technik (str): Die Technik, für die die Faktoren berechnet werden.
            id_df (pd.Series): Serie mit den IDs der Busse.

        Returns:
            tuple: Ein Tupel mit zwei Elementen:
                - pd.Series: Serie mit den berechneten Faktoren für die Busse.
                - float: Der berechnete Faktor für die Bounding Box.
        """


        arr_factor = pd.Series(0.0, index=buses_zensus.index)
        #land = buses_land.iloc[0, 0]
        Bev_data_Zensus = Bev_data[[col for col in Bev_data.columns if col.startswith("Zensus")]].copy()
        
        '''
        Auf Cluster unnötig
        '''
        print('Technik: ', technik)
        print("Bev_data_Zensus.columns vor:", Bev_data_Zensus.columns)
        print("Technik_Faktoren.columns vor:", Technik_Faktoren.columns)
        Bev_data_Zensus = Bev_data_Zensus[list(Technik_Faktoren.columns)]
        buses_zensus = buses_zensus[list(Technik_Faktoren.columns)]

        Bev_data_Technik = Bev_data[technik].copy()
        id = id_df.iloc[0]
        factor_area = Bev_data_Zensus.loc[id] @ Technik_Faktoren.loc[technik]
        for j, zensus in buses_zensus.iterrows():
            id = id_df.loc[j]
            factor_bus = zensus @ Technik_Faktoren.loc[technik]
            if factor_bus < 0:
                factor_bus = 0
            arr_factor.loc[j] = factor_bus / factor_area * Bev_data_Technik.loc[id]

        factor_bbox = bbox_zensus @ Technik_Faktoren.loc[technik] / factor_area * Bev_data_Technik.loc[id]
        if factor_bbox < 0:
            factor_bbox = 0
        return arr_factor, factor_bbox
        


def technik_sortieren(buses: pd.DataFrame, Technik: str, amount_total: float, solar_power: float) -> pd.DataFrame:
    """
    Sortiert die Busse nach der Technik und verteilt die Leistung gleichmäßig auf die Busse
    
    Args:
        buses (pd.DataFrame): DataFrame mit den Busdaten.
        Technik (str): Die Technik, die verteilt werden soll.
        amount_total (float): Die gesamte Anzahl, die verteilt werden soll.
        solar_power (float): Die installierte Solarleistung für die PLZ.

    Returns:
        pd.DataFrame: DataFrame mit den verteilten Leistungen für die Technik.
    """

    # Neue Spalte vorbereiten
    buses['Power_' + Technik] = 0.0  # Initialisierung mit 0

    # buses nach Faktor gruppieren und als dict sortiert speichern
    buses_grouped = buses.groupby('Factor_'+Technik)
    groups_dict = {key: group for key, group in sorted(buses_grouped, key=lambda x: x[0], reverse=True)}

    #print("groups_dict:", groups_dict)

    # Series zum späteren gemeinsamen Hinzufügen
    power_col = pd.Series(0.0, index=buses.index)

    if Technik == 'solar' and "type_1" in buses.columns:
        print("Jetzt solar von type_1 übernehmen")
        # Allen buses mit type_1 == Technik solar_power zuordnen
        mask_type1 = buses["type_1"] == Technik
        #print("mask_type1:", mask_type1)


        power_col.loc[mask_type1] = solar_power
        # buses.loc[mask_type1, 'Power_' + Technik] = solar_power


        '''
        Passt die p_nom_1 größe zu den Daten aus dem MArktstammdatenregister, oder übernehmen wir nur
        dass dort eine pv ist, aber größe setzen wir selbst?
        '''

        # Bestimmung der Anzahl der gesetzten solaranlagen
        amount = mask_type1.sum()

        # Reduzieren von amount_total um die schon vorhandene Leistung
        amount_total -= amount

        # Wenn schon genug Solaranlagen vorhanden sind, abbrechen
        if amount_total <= 0:
            print("Keine Solaranlagen zu verteilen, da schon genug vorhanden sind.")
            buses['Power_' + Technik] = power_col
            return buses
        # if amount_total < 0:
        #     amount_total = 0
        #     print("Achtung, es sind mehr Solaranlagen im Netz als verteilt werden sollen!")
        # #print("Anzahl der schon vorhandenen Solaranlagen:", amount)
        #print("Anzahl der noch zu verteilenden Solaranlagen:", amount_total)


    
    p = 0
    for key in groups_dict.keys():
        #print("key:", key)
        # buszahl aus group random ziehen, zahl folgt aus wert des keys
        group = groups_dict[key]

        liste = []
        if Technik == 'E_car':
            print("E-Car, Haushalte berücksichtigen")
            for bus in group.index:
                # Bus index hinzufügen, so oft wie Haushalte existieren
                liste.extend([bus] * int(buses.loc[bus, 'Haushalte']))
                print("Bus:", bus, "Haushalte:", int(buses.loc[bus, 'Haushalte']))

        else:
            # Bus index hinzufügen
            print("Andere Technik, Haushalte nicht berücksichtigen")
            liste = group.index.tolist()
        
        
        print("Liste für Technik", Technik, ":", liste)

        #print("group:", group)

        # 5 zeilen random ziehen
        sample_size = int(min(key, len(liste))) # Sicherstellen, dass die Stichprobengröße nicht größer als die Gruppengröße ist
        sampled_buses_index = random.sample(liste, sample_size)  # Zufällige Auswahl von Bussen aus der Gruppe
        #print("sampled_buses:", sampled_buses)

        # setzen von ['Factor_Technik'] in buses auf Faktor
        if Technik == 'solar':
            print("Setze solar_power:", solar_power, "für buses als Power")
            power_col.loc[sampled_buses_index] = solar_power
            #print("Setze solar_power:", solar_power, "für buses:", sampled_buses.index.tolist())
        else:
            print("Setze:", key, "für buses als Power")
            power_col.loc[sampled_buses_index] += key

        p +=sample_size
        if p > amount_total:
            rest = int(p - amount_total)
            print("p:", p   , "amount_total:", amount_total)
            print("Rest wird abgezogen:", rest)
            # rest zahl aus sample buses wieder random auf 0 setzen
            rest_buses_index = random.sample(sampled_buses_index, rest)
            power_col.loc[rest_buses_index] = 0
            break

    # gemeinsame Spalte hinzufügen
    buses['Power_' + Technik] = power_col

    return buses


def solar_ausrichtung(buses: pd.DataFrame, plz: str, pv_plz: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt den Bussen mit Solartechnik die Ausrichtung und Neigung basierend auf den PLZ-Daten hinzu.
    
    Args:
        buses (pd.DataFrame): DataFrame mit den Busdaten, das die Spalte 'Power_solar' enthält.
        plz (str): Die Postleitzahl, für die die Ausrichtung und Neigung zugeordnet werden soll.
        pv_plz (pd.DataFrame): DataFrame mit den PLZ-Daten, das die Spalten 'Hauptausrichtung_Anteil' und 'HauptausrichtungNeigungswinkel_Anteil' enthält.
        
    Returns:
        pd.DataFrame: DataFrame mit den neuen Spalten 'Hauptausrichtung_Anteil' und 'HauptausrichtungNeigungswinkel_Anteil' für die Busse mit Solartechnik.
    """


    # Ausrichtung und Neigung für PV
    ausrichtung = ast.literal_eval(pv_plz.loc[plz, 'Hauptausrichtung_Anteil'])
    neigung = ast.literal_eval(pv_plz.loc[plz, 'HauptausrichtungNeigungswinkel_Anteil'])

    print("Neigung:", neigung)

    # Filtern der buses mit solar
    df = buses[buses["Power_solar"] != 0].copy()

    # Zuordnung der Ausrichtung
    df["Hauptausrichtung_Anteil"] = np.random.choice(
        list(ausrichtung.keys()),          # mögliche Buchstaben
        size=len(df),                  # Anzahl = Zeilenanzahl
        p=list(ausrichtung.values())       # Wahrscheinlichkeiten
    )

    # Zuordnung der Neigung
    df["HauptausrichtungNeigungswinkel_Anteil"] = np.random.choice(
        list(neigung.keys()),          # mögliche Buchstaben
        size=len(df),                  # Anzahl = Zeilenanzahl
        p=list(neigung.values())       # Wahrscheinlichkeiten
    )

    # Zuordnung von Ausrichtung und Neigung 
    buses.loc[df.index, 'Hauptausrichtung_Anteil'] = df['Hauptausrichtung_Anteil']
    buses.loc[df.index, 'HauptausrichtungNeigungswinkel_Anteil'] = df['HauptausrichtungNeigungswinkel_Anteil']

    return buses


def storage(buses: pd.DataFrame, storage_pv: float) -> pd.DataFrame:
    """
    Fügt eine Spalte 'speicher' zu den Bussen hinzu, die den Speicherbedarf für Solarenergie berechnet.
    
    Args:
        buses (pd.DataFrame): DataFrame mit den Busdaten, das die Spalten 'type_1' und 'p_nom_1' enthält.
    
    Returns:
        pd.DataFrame: DataFrame mit der neuen Spalte 'speicher', die den Speicher bedarf für Solarenergie enthält.
    """
    
    '''
    Prob folgt aus Marktstammdatenregister:
    '''
    prob = storage_pv 
    
    # np.random.seed(seed) # Setze einen Seed für Reproduzierbarkeit
    buses = buses.copy()
    buses['speicher'] = 0.0
    # Für alle Zeilen mit Power_solar, Speicherkapazität = Power_solar * 1
    solar_index = buses[buses['Power_solar'] != 0].sample(frac=prob).index
    buses.loc[solar_index, 'speicher'] = buses.loc[solar_index, 'Power_solar'] * 1 # 1 kWp PV-Leistung = 1 kWh Speicher


    return buses




def relative_humidity(t: float, td: float) -> float:
    """
    Berechnet die relative Luftfeuchtigkeit basierend auf der Temperatur und dem Taupunkt.
    
    Args:
        t (float): Die Temperatur in Grad Celsius.
        td (float): Der Taupunkt in Grad Celsius.
        
    Returns:
        float: Die relative Luftfeuchtigkeit in Prozent.
    """

    # t und td in °C
    es = 6.112 * np.exp((17.67 * t) / (t + 243.5))  # Sättigungsdampfdruck
    e = 6.112 * np.exp((17.67 * td) / (td + 243.5)) # Dampfdruck
    rh = 100 * e / es
    return rh


def env_wetter(bbox: list, pfad: str, time_discretization: int = 3600, timesteps_horizon: int = 8760, timesteps_used_horizon: int = 8760, timesteps_total: int = 8760) -> Environment: #, year):
    """
    Lädt Wetterdaten von ERA5 für eine gegebene Bounding Box und berechnet relevante Umweltparameter.
    
    Args:
        bbox (list): Liste mit den Koordinaten der Bounding Box im Format [W, S, E, N].
        time_discretization (int): Zeitdiskretisierung in Sekunden (Standard: 3600 Sekunden = 1 Stunde).
        timesteps_horizon (int): Anzahl der Zeitschritte im Planungshorizont (Standard: 8760 für ein Jahr).
        timesteps_used_horizon (int): Anzahl der Zeitschritte, die tatsächlich verwendet werden (Standard: 8760 für ein Jahr).
        timesteps_total (int): Gesamtanzahl der Zeitschritte (Standard: 8760 für ein Jahr).
        year (int): Jahr der Wetterdaten (z.B. 2013).
        
    Returns:
        Environment: Ein Environment-Objekt mit den geladenen Wetterdaten und berechneten Parametern.
    """


    # Definierung der gebrauchten Tabellen
    variables = [
        ("10m_u_component_of_wind", "u10"),
        ("10m_v_component_of_wind", "v10"),
        ("2m_dewpoint_temperature", "d2m"),
        ("2m_temperature", "t2m"),
        ("surface_pressure", "sp"),
        ("surface_solar_radiation_downwards", "ssrd"),
        ("total_sky_direct_solar_radiation_at_surface", "fdir"),
        ("total_cloud_cover", "tcc")
    ]

    # # cdsapi Client initialisieren
    # client = cdsapi.Client()
    # # Daten für jede Variable anfordern
    # for var, cod in variables:
    #     dataset = "reanalysis-era5-single-levels"
    #     request = {
    #         "product_type": "reanalysis",
    #         "variable": var,
    #         "year": str(year),
    #         "month": [
    #             "01", "02", "03",
    #             "04", "05", "06",
    #             "07", "08", "09",
    #             "10", "11", "12"
    #         ],
    #         "day": [
    #             "01", "02", "03",
    #             "04", "05", "06",
    #             "07", "08", "09",
    #             "10", "11", "12",
    #             "13", "14", "15",
    #             "16", "17", "18",
    #             "19", "20", "21",
    #             "22", "23", "24",
    #             "25", "26", "27",
    #             "28", "29", "30",
    #             "31"
    #         ],
    #         "time": [
    #             "00:00", "01:00", "02:00",
    #             "03:00", "04:00", "05:00",
    #             "06:00", "07:00", "08:00",
    #             "09:00", "10:00", "11:00",
    #             "12:00", "13:00", "14:00",
    #             "15:00", "16:00", "17:00",
    #             "18:00", "19:00", "20:00",
    #             "21:00", "22:00", "23:00"
    #         ],
    #         "format": "netcdf",
    #         "area": bbox
    #     }

    #     # Anfrage an den CDS stellen und Daten speichern
    #     client.retrieve(dataset, request, var+'.nc')

    # Daten in xarray Datasets laden
    # und in einem Dictionary speichern
    datasets = {}
    for var, cod in variables:
        filename = 'GER_' +var + '.nc'
        ds = xr.open_dataset(os.path.join(pfad, 'input/weather_2013', filename))
        datasets[var] = ds
        print(f"Variable {var} verarbeitet.")

    

    # # Daten löschen
    # for var, cod in variables:
    #     os.remove(var + '.nc')
    
    
    # Mittelpunkt von bbox
    # Berechnet den Mittelpunkt der Bounding Box.
    # bbox: [N, W, S, E] [left, bottom, right, top]
    lat_target = (bbox[3] + bbox[1]) / 2
    lon_target = (bbox[0] + bbox[2]) / 2

    data_dict = {}

    for var, cod in variables:
        data = datasets[var].sel(latitude=lat_target, longitude=lon_target, method='nearest')
        df = data.to_dataframe()
        data_dict[var] = df[cod]


    # Umrechnung der Einheiten
    # Temperatur von Kelvin zu Celsius umwandeln
    data_dict["2m_temperature"] = data_dict["2m_temperature"] - 273.15
    data_dict["2m_dewpoint_temperature"] = data_dict["2m_dewpoint_temperature"] - 273.15

    # Luftdruck von Pa zu hPa umwandeln
    data_dict["surface_pressure"] = data_dict["surface_pressure"] / 100  # Umwandlung von Pa zu hPa

    # Rel. Humidity  berechnen
    data_dict["2m_relative_humidity"] = relative_humidity(data_dict["2m_temperature"], data_dict["2m_dewpoint_temperature"])

    # Windgeschwindigkeit berechnen
    data_dict["10m_wind_speed"] = (data_dict["10m_u_component_of_wind"]**2 + data_dict["10m_v_component_of_wind"]**2)**0.5

    # Strahlungswerte berechnen: Umwandlung von J/m² zu W/m²
    data_dict["surface_solar_radiation_downwards"] = data_dict["surface_solar_radiation_downwards"] / 3600
    data_dict["total_sky_direct_solar_radiation_at_surface"] = data_dict["total_sky_direct_solar_radiation_at_surface"] / 3600

    # Diffuse Strahlung berechnen
    data_dict["surface_diffuse_solar_radiation_at_surface"] = data_dict["surface_solar_radiation_downwards"] - data_dict["total_sky_direct_solar_radiation_at_surface"]

    # Bedeckungsgrad berechnen
    data_dict["total_cloud_cover"] = (data_dict["total_cloud_cover"] * 8).round().clip(lower=0, upper=8)

    # Neue Variablen definieren
    variables_neu = [
        "10m_wind_speed",
        "2m_relative_humidity",
        "2m_temperature",
        "surface_pressure",
        "surface_diffuse_solar_radiation_at_surface",
        "total_sky_direct_solar_radiation_at_surface",
        "total_cloud_cover"
    ]

    # Speichern der Daten in CSV-Dateien
    for var in variables_neu:
        filename = f"{var}_nearest.txt"
        data_dict[var].to_csv(filename, sep="\t", index=False, header=False)


    # Erstellen der Environment
    timer = Timer(
        time_discretization=time_discretization,
        timesteps_horizon=timesteps_horizon,
        timesteps_used_horizon=timesteps_used_horizon,
        timesteps_total=timesteps_total
    )


    weather = Weather(timer,
                    path_TRY=None, path_TMY3=None,
                    path_temperature="2m_temperature_nearest.txt",
                    path_direct_radiation="total_sky_direct_solar_radiation_at_surface_nearest.txt",
                    path_diffuse_radiation="surface_diffuse_solar_radiation_at_surface_nearest.txt",
                    path_wind_speed="10m_wind_speed_nearest.txt",
                    path_humidity="2m_relative_humidity_nearest.txt",
                    path_pressure="surface_pressure_nearest.txt",
                    path_cloudiness="total_cloud_cover_nearest.txt",
                    time_discretization=3600,
                    delimiter="\t",
                    use_TRY=None, use_TMY3=False,
                    location=(50.76, 6.07), height_velocity_measurement=10,
                    altitude=152.0, time_zone=1)
    prices = Prices()
    environment = Environment(timer, weather, prices)

    for var in variables_neu:
        filename = f"{var}_nearest.txt"
        os.remove(filename)
        print(f"{filename} wurde gelöscht.")

    return environment

# %%
