import geopandas as gpd
from pyproj import Transformer
import pandas as pd
import polars as pl
import osmnx as ox
import scipy.spatial as spatial
import numpy as np
import json
from functools import reduce
import numpy as np
import ast
import random
import xarray as xr
import os

from pycity_base.classes.timer import Timer
from pycity_base.classes.weather import Weather
from pycity_base.classes.prices import Prices
from pycity_base.classes.environment import Environment

# Import for type hints
import pypsa

#%% Funktionen zum aufrufen





'''
Anpassen an Abfrage zu commercial
'''





def get_osm_data(bbox: list) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Fetches OpenStreetMap (OSM) data for a given bounding box.

    Args:
        bbox (list): Bounding box coordinates in the format [left, bottom, right, top].

    Returns:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
            A tuple containing:
            - The OSM graph for the area.
            - The OSM features (buildings, land use, roads, etc.) as GeoDataFrames.
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
    area_graph = ox.graph.graph_from_bbox(bbox, network_type="all", retain_all=True)
    area_features = ox.features.features_from_bbox(bbox, tags=tags)
    return area_graph, area_features


def compute_bbox_from_buses(net: pypsa.Network) -> list[float]:
    """
    Compute the bounding box from the bus coordinates in a PyPSA network.

    Args:
        net (pypsa.Network): The PyPSA network object containing bus coordinates (x, y).

    Returns:
        list[float]: Bounding box coordinates in the format [left, bottom, right, top].
    """

    # Extract bus coordinates
    x_min = net.buses['x'].min()
    x_max = net.buses['x'].max()
    y_min = net.buses['y'].min()
    y_max = net.buses['y'].max()

    # Create bounding box
    bbox = [x_min, y_min, x_max, y_max]

    return bbox


def federal_state(buses: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns federal state and regional data to network buses based on spatial proximity.

    Args:
        net_buses (pd.DataFrame): DataFrame containing network bus coordinates (columns: 'x', 'y').
        data (pd.DataFrame): DataFrame containing reference federal state and postal region data,
                             with a 'geo_point_2d' column containing JSON strings of {"lat": ..., "lon": ...}.

    Returns:
        pd.DataFrame: Updated bus DataFrame with added columns:
                      ['lan_name', 'plz_name', 'plz_code', 'krs_code', 'lan_code', 'krs_name'].
    """

    # Prepare reference points (federal state data)
    data = data.copy()
    data["geo_point_2d"] = data["geo_point_2d"].apply(json.loads)
    ref_lon = data["geo_point_2d"].apply(lambda d: d["lon"])
    ref_lat = data["geo_point_2d"].apply(lambda d: d["lat"])
    ref_points = np.vstack((ref_lon, ref_lat)).T

    # Prepare target points (bus coordinates)
    bus_lon = buses["x"]
    bus_lat = buses["y"]
    bus_points = np.vstack((bus_lon, bus_lat)).T

    # Build KD-tree and find nearest reference point for each bus
    tree = spatial.cKDTree(ref_points)
    _, idx = tree.query(bus_points)

    # Assign attributes from nearest reference entry
    cols_to_assign = ["lan_name", "plz_name", "plz_code", "krs_code", "lan_code", "krs_name"]
    for col in cols_to_assign:
        buses[col] = data.iloc[idx][col].values

    return buses


def zensus_ID(buses: pd.DataFrame, folder: str) -> pd.DataFrame:
    """
    Assigns the 100m grid ID (GITTER_ID_100m) from census data to network buses.

    Args:
        buses_df (pd.DataFrame): DataFrame containing bus coordinates ('x', 'y').
        folder (str): Path to the folder containing the census CSV file.

    Returns:
        pd.DataFrame: Updated bus DataFrame with an added 'GITTER_ID_100m' column.
    """

    # Load zensus data
    zensus = pd.read_csv(folder + "/Zensus2022_Bevoelkerungszahl_100m-Gitter.csv", sep=";")
    buses = buses.copy()

    # Convert bus coordinates from EPSG:4326 to EPSG:3035
    x_array, y_array = epsg4326_zu_epsg3035(buses["x"], buses["y"])

    # Build KD-tree from zensus reference points
    reference_points = np.vstack((zensus['x_mp_100m'], zensus['y_mp_100m'])).T
    tree = spatial.cKDTree(reference_points)

    # Query points: buses in projected coordinates
    query_points = np.vstack((x_array, y_array)).T

    # Find nearest neighbors
    distances, indices = tree.query(query_points)

    # Assign GITTER_ID_100m from zensus to buses
    buses['GITTER_ID_100m'] = np.array(zensus["GITTER_ID_100m"])[indices]

    return buses


def load_zensus(buses: pd.DataFrame, folder: str) -> pd.DataFrame:
    """
    Loads census data from CSV files in the specified folder and returns them as a DataFrame.

    Args:
        buses (pd.DataFrame): DataFrame containing the network bus data, which must include the column 'GITTER_ID_100m'.
        folder (str): Path to the folder containing the census CSV files.

    Returns:
        pd.DataFrame: A DataFrame containing the combined census data.
    """

    columns = buses['GITTER_ID_100m']
    buses = buses.copy()

    # Loading the census data from the CSV files in the specified folder
    Zensus2022_Bevoelkerungszahl_100m = (pl.scan_csv(folder + "/Zensus2022_Bevoelkerungszahl_100m-Gitter.csv", separator=";")
                                         .filter(pl.col("GITTER_ID_100m")
                                                 .is_in(columns)).select("GITTER_ID_100m",
                                                                         "Einwohner").collect()
                                        )
    Zensus2022_Bevoelkerungszahl_100m = Zensus2022_Bevoelkerungszahl_100m.to_pandas()
    Zensus2022_Bevoelkerungszahl_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Bevoelkerungszahl_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses = buses.merge(Zensus2022_Bevoelkerungszahl_100m, on="GITTER_ID_100m", how="left")


    Zensus2022_Durchschn_Nettokaltmiete_Anzahl_der_Wohnungen_100m = (pl.scan_csv(folder + "/Zensus2022_Durchschn_Nettokaltmiete_Anzahl_der_Wohnungen_100m-Gitter.csv", separator=";")
                                                    .filter(pl.col("GITTER_ID_100m").is_in(columns))
                                                    .select("GITTER_ID_100m",
                                                            "durchschnMieteQM",
                                                            ).collect()
                                                )
    Zensus2022_Durchschn_Nettokaltmiete_Anzahl_der_Wohnungen_100m = Zensus2022_Durchschn_Nettokaltmiete_Anzahl_der_Wohnungen_100m.to_pandas()
    Zensus2022_Durchschn_Nettokaltmiete_Anzahl_der_Wohnungen_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Durchschn_Nettokaltmiete_Anzahl_der_Wohnungen_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses = buses.merge(Zensus2022_Durchschn_Nettokaltmiete_Anzahl_der_Wohnungen_100m, on="GITTER_ID_100m", how="left")




    Zensus2022_Groesse_des_privaten_Haushalts_100m = (pl.scan_csv(folder + "/Zensus2022_Groesse_des_privaten_Haushalts_100m-Gitter.csv", separator=";")
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
    buses = buses.merge(Zensus2022_Groesse_des_privaten_Haushalts_100m, on="GITTER_ID_100m", how="left")






    Zensus2022_Staatsangehoerigkeit_Gruppen_100m = (pl.scan_csv(folder + "/Zensus2022_Staatsangehoerigkeit_Gruppen_100m-Gitter.csv", separator=";")
                                                    .filter(pl.col("GITTER_ID_100m").is_in(columns))
                                                    .select("GITTER_ID_100m",
                                                            "EU27_Land",
                                                            ).collect()
                                                    )
    Zensus2022_Staatsangehoerigkeit_Gruppen_100m = Zensus2022_Staatsangehoerigkeit_Gruppen_100m.to_pandas()
    Zensus2022_Staatsangehoerigkeit_Gruppen_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Staatsangehoerigkeit_Gruppen_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses = buses.merge(Zensus2022_Staatsangehoerigkeit_Gruppen_100m, on="GITTER_ID_100m", how="left")


    Zensus2022_Typ_der_Kernfamilie_nach_Kindern_100m = (pl.scan_csv(folder + "/Zensus2022_Typ_der_Kernfamilie_nach_Kindern_100m-Gitter.csv", separator=";")
                                                            .filter(pl.col("GITTER_ID_100m").is_in(columns))
                                                            .select("GITTER_ID_100m",
                                                                    "Ehep_Kinder_ab18",
                                                                    "NichtehelLG_mind_1Kind_unter18",
                                                                    ).collect()
                                                            )
    Zensus2022_Typ_der_Kernfamilie_nach_Kindern_100m = Zensus2022_Typ_der_Kernfamilie_nach_Kindern_100m.to_pandas()
    Zensus2022_Typ_der_Kernfamilie_nach_Kindern_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Typ_der_Kernfamilie_nach_Kindern_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses = buses.merge(Zensus2022_Typ_der_Kernfamilie_nach_Kindern_100m, on="GITTER_ID_100m", how="left")





    Zensus2022_Flaeche_der_Wohnung_10m2_Intervalle_100m = (pl.scan_csv(folder + "/Zensus2022_Flaeche_der_Wohnung_10m2_Intervalle_100m-Gitter.csv", separator=";")
                                                    .filter(pl.col("GITTER_ID_100m").is_in(columns))
                                                    .select("GITTER_ID_100m",
                                                            "30bis39",
                                                            "40bis49",
                                                            ).collect()
                                                )
    Zensus2022_Flaeche_der_Wohnung_10m2_Intervalle_100m = Zensus2022_Flaeche_der_Wohnung_10m2_Intervalle_100m.to_pandas()
    Zensus2022_Flaeche_der_Wohnung_10m2_Intervalle_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Flaeche_der_Wohnung_10m2_Intervalle_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses = buses.merge(Zensus2022_Flaeche_der_Wohnung_10m2_Intervalle_100m, on="GITTER_ID_100m", how="left")

    Zensus2022_Wohnung_Gebaeudetyp_Groesse_100m = (pl.scan_csv(folder + "/Zensus2022_Wohnung_Gebaeudetyp_Groesse_100m-Gitter.csv", separator=";")
                                                    .filter(pl.col("GITTER_ID_100m").is_in(columns))
                                                    .select("GITTER_ID_100m",
                                                            "Freist_ZFH",
                                                            ).collect()
                                                )
    Zensus2022_Wohnung_Gebaeudetyp_Groesse_100m = Zensus2022_Wohnung_Gebaeudetyp_Groesse_100m.to_pandas()
    Zensus2022_Wohnung_Gebaeudetyp_Groesse_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Wohnung_Gebaeudetyp_Groesse_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses = buses.merge(Zensus2022_Wohnung_Gebaeudetyp_Groesse_100m, on="GITTER_ID_100m", how="left")



    Zensus2022_Baujahr_JZ_100m = (pl.scan_csv(folder + "/Zensus2022_Baujahr_JZ_100m-Gitter.csv", separator=";")
                                                    .filter(pl.col("GITTER_ID_100m").is_in(columns))
                                                    .select("GITTER_ID_100m",
                                                            "Vor1919",
                                                            "a1970bis1979",
                                                            ).collect()
                                                )
    Zensus2022_Baujahr_JZ_100m = Zensus2022_Baujahr_JZ_100m.to_pandas()
    Zensus2022_Baujahr_JZ_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Baujahr_JZ_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses = buses.merge(Zensus2022_Baujahr_JZ_100m, on="GITTER_ID_100m", how="left")


    Zensus2022_Gebaeude_nach_Anzahl_der_Wohnungen_100m = (pl.scan_csv(folder + "/Zensus2022_Gebaeude_nach_Anzahl_der_Wohnungen_100m-Gitter.csv", separator=";")
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
    buses = buses.merge(Zensus2022_Gebaeude_nach_Anzahl_der_Wohnungen_100m, on="GITTER_ID_100m", how="left")


    Zensus2022_Geb_Gebaeudetyp_Groesse_100m = (pl.scan_csv(folder + "/Zensus2022_Geb_Gebaeudetyp_Groesse_100m-Gitter.csv", separator=";")
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
    buses = buses.merge(Zensus2022_Geb_Gebaeudetyp_Groesse_100m, on="GITTER_ID_100m", how="left")



    Zensus2022_Energietraeger_100m = (pl.scan_csv(folder + "/Zensus2022_Energietraeger_100m-Gitter_utf8.csv", separator=";")
                                                    .filter(pl.col("GITTER_ID_100m").is_in(columns))
                                                    .select("GITTER_ID_100m",
                                                            "Gas",
                                                            "Holz_Holzpellets",
                                                            "Kohle",
                                                            ).collect()
                                                )
    Zensus2022_Energietraeger_100m = Zensus2022_Energietraeger_100m.to_pandas()
    Zensus2022_Energietraeger_100m.rename(columns={col: f"Zensus_{col}" for col in Zensus2022_Energietraeger_100m.columns if col != "GITTER_ID_100m"}, inplace=True)
    buses = buses.merge(Zensus2022_Energietraeger_100m, on="GITTER_ID_100m", how="left")


    Zensus2022_Gebaeude_nach_Energietraeger_der_Heizung_100m = (pl.scan_csv(folder + "/Zensus2022_Gebaeude_nach_Energietraeger_der_Heizung_100m-Gitter.csv", separator=";")
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
    buses = buses.merge(Zensus2022_Gebaeude_nach_Energietraeger_der_Heizung_100m, on="GITTER_ID_100m", how="left")

    return buses


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

# def epsg32632_zu_epsg3035(lon: float, lat: float) -> tuple[float, float]:
#     """
#     Umrechnung von WGS84 (EPSG:32632) zu ETRS89 (EPSG:3035)
#     Args:
#         lon (float or pd.Series): Längengrad in WGS84.
#         lat (float or pd.Series): Breitengrad in WGS84.
    
#     Returns:
#         tuple: Ein Tupel mit den umgerechneten Koordinaten (rechtswert, hochwert) in ETRS89 (EPSG:3035).
#     """

#     transformer_zu_EPSG3035 = Transformer.from_crs("EPSG:32632", "EPSG:3035", always_xy=True)
#     rechtswert, hochwert = transformer_zu_EPSG3035.transform(lon, lat)
#     return rechtswert, hochwert


# def zensus_df_verbinden(daten_zensus: list) -> pd.DataFrame:
#     """
#     Verbindet mehrere DataFrames basierend auf der GITTER_ID_100m Spalte.
    
#     Args:
#         daten_zensus (list): Eine Liste von DataFrames, die auf GITTER_ID_100m basieren.

#     Returns:
#         pd.DataFrame: Ein DataFrame, das die Daten aus allen DataFrames kombiniert.
#     """

#     # Sicherstellen, dass alle DFs die gleichen drei Schlüsselspalten haben
#     basis_spalten = ["GITTER_ID_100m", "x_mp_100m", "y_mp_100m"]

#     # DataFrames per GITTER_ID_100m zusammenführen
#     df_vereint = reduce(lambda left, right: pd.merge(left, right, on=basis_spalten, how='outer'), daten_zensus)
#     df_vereint = df_vereint.rename(columns={col: f"Zensus_{col}" for col in df_vereint.columns if col not in basis_spalten})
#     return df_vereint




# def daten_zuordnen(net_buses: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
#     """
#     Ordnet die Zensusdaten den Bussen im Netzwerk zu, basierend auf den Koordinaten.
    
#     Args:
#         net_buses (pd.DataFrame): DataFrame mit den Busdaten des Netzwerks.
#         data (pd.DataFrame): DataFrame mit den Zensusdaten.
        
#     Returns:
#         pd.DataFrame: DataFrame mit den zugeordneten Zensusdaten.
#     """

#     x_array, y_array = epsg4326_zu_epsg3035(net_buses["x"], net_buses["y"])

#     # Referenzpunkte in der Zensus-Tabelle
#     reference_points = np.vstack((data['x_mp_100m'].values, data['y_mp_100m'].values)).T
#     tree = spatial.cKDTree(reference_points)

#     # Zielpunkte aus dem Netz
#     query_points = np.vstack((x_array, y_array)).T

#     # Nächste Nachbarn finden
#     distances, indices = tree.query(query_points)

#     # Spaltennamen ändern
#     # data = data.rename(columns=lambda col: f"Zensus_{col}").copy()
    
#     # Passende Zeilen aus data holen
#     matched_data = data.iloc[indices].copy()

#     # Index an net_buses anpassen, damit concat funktioniert
#     matched_data.index = net_buses.index

#     # DataFrames nebeneinander anfügen
#     result = pd.concat([net_buses, matched_data], axis=1)

#     return result



# def load_shapes(datei: str, bundesland: str) -> gpd.GeoDataFrame:
#     """
#     Lädt die Bundesland-Geometrien aus einer GeoJSON-Datei und filtert sie auf die relevanten Bundesländer.
#     Args:
#         datei (str): Pfad zur GeoJSON-Datei mit den Bundesland-Geometrien.
        
#     Returns:
#         gpd.GeoDataFrame: Ein GeoDataFrame mit den Geometrien der relevanten Bundesländer.
#     """

#     # Laden der Geometrien aus der Datei
#     shapes = gpd.read_file(datei, layer="vg5000_lan")
    
#     # Filtern der Geometrien nach Bundesländern
#     land = shapes[shapes["GEN"].isin([bundesland])]
#     # Filtern der Geometrien
#     land_aggregiert = land.dissolve(by="GEN")
#     land_umrisse = land_aggregiert[["geometry"]]
#     land_3035 = land_umrisse.to_crs(epsg=3035)

#     return land_3035


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



# def bundesland_zuordnung(ordner: str, shapes: gpd.GeoDataFrame) -> pd.DataFrame:
#     """
#     Ordnet die Zensusdaten den Bundesländern zu, basierend auf den Geometrien der Bundesländer.
#     Args:
#         zensus (pd.DataFrame): DataFrame mit den Zensusdaten.
#         shapes (gpd.GeoDataFrame): GeoDataFrame mit den Geometrien der Bundesländer.
        
#     Returns:
#         pd.DataFrame: Ein DataFrame mit den zugeordneten Bundesland-Daten."""

#     # Zensusdaten laden
#     zensus = pd.read_csv(ordner + "/Zensus2022_Bevoelkerungszahl_100m-Gitter.csv", sep=";")
    
#     # Punkte-Geometrie erstellen
#     geometry = gpd.GeoSeries(gpd.points_from_xy(zensus['x_mp_100m'], zensus['y_mp_100m']))
    
#     # Boolean-Maske: welche Punkte liegen innerhalb der Bundesländer
#     mask = geometry.within(shapes.unary_union)
    
#     # Nur diese Zeilen auswählen
#     zensus_filtered = zensus[mask].copy()

#     gitter_id = zensus_filtered["GITTER_ID_100m"]

#     return gitter_id



# def bundesland_summieren(df: pd.DataFrame, bundesland: str) -> pd.DataFrame:
#     """
#     Summiert die Zensusdaten für ein bestimmtes Bundesland.
    
#     Args:
#         df (pd.DataFrame): DataFrame mit den Zensusdaten.
#         bundesland (str): Der Name des Bundeslands, für das die Daten summiert werden
        
#     Returns:
#         pd.DataFrame: Ein DataFrame mit den summierten Zensusdaten für das Bundesland
#     """

    
#     numerische_spalten = [col for col in df.columns if col.startswith("Zensus")]

#     for col in numerische_spalten:
#         # Komma durch Punkt ersetzen, Strings in float konvertieren
#         df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
#         # Ungültige Zeichen wie "–" in NaN umwandeln
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#         # NaN durch 0 ersetzen
#         df[col] = df[col].fillna(0)

#     print(df)
#     df_agg = df[numerische_spalten].agg(agg_dict)
#     df_neu = pd.DataFrame()
#     df_neu[bundesland] = df_agg

#     return df_neu


# def gewichtungsfaktor(land: str, kategorien_eigenschaften: pd.DataFrame, factors: dict, row: pd.Series, technik_faktoren: pd.Series, gesamtgewicht_dict: dict) -> float:
#     """
#     Berechnet den Gewichtungsfaktor für eine bestimmte Technik und ein bestimmtes Land basierend auf den Eigenschaften und Faktoren.
    
#     Args:
#         land (str): Der Name des Landes, für das der Faktor berechnet wird.
#         kategorien_eigenschaften (pd.DataFrame): DataFrame mit den Kategorien der Eigenschaften.
#         factors (dict): Dictionary mit den Faktoren für jede Kategorie.
#         row (pd.Series): Eine Zeile des DataFrames, die die Eigenschaften enthält.
#         technik_faktoren (pd.Series): Serie mit den Technik-Faktoren für das Land.
#         gesamtgewicht_dict (dict): Dictionary mit den Gesamtgewichten für jede Kategorie und Land.
        
#     Returns:
#         float: Der berechnete Gewichtungsfaktor für die Technik im angegebenen Land.
#     """

#     weighted_sum = 1.0
#     # Iteriere über die Kategorien
#     for kat in kategorien_eigenschaften.columns:
#         # Filtere die Eigenschaften für die aktuelle Kategorie
#         eigenschaften = kategorien_eigenschaften[kat].dropna()

#         # hole nur die relevanten Faktoren und Werte
#         faktoren_kat = [factors[attr] for attr in eigenschaften]
#         werte_kat = row[eigenschaften]

#         # numerische Multiplikation
#         sum_bus = (werte_kat * faktoren_kat).sum()
#         sum_land = gesamtgewicht_dict.get((land, kat), 0)

#         # Berechnung von Gewichtungsprodukt
#         if sum_land != 0:
#             weighted_sum *= sum_bus / sum_land
#         else:
#             weighted_sum *= 0

#     print("Technik Faktoren:", technik_faktoren)

#     return weighted_sum * float(technik_faktoren.iloc[0])




# def calculate_factors(df: pd.DataFrame, factors: dict, kategorien_eigenschaften: pd.DataFrame, Bev_data: pd.DataFrame, technik: str) -> pd.DataFrame:

#     """
#     Berechnet den Faktor für jede Zeile im DataFrame basierend auf den Eigenschaften und Bevölkerungsdaten.

#     Parameters:
#         df (pd.DataFrame): DataFrame mit den Eigenschaften der Busse.
#         factors (dict): Dictionary mit den Faktoren für die Eigenschaften.
#         kategorien_eigenschaften (pd.DataFrame): DataFrame mit den Eigenschaften, die gefiltert werden sollen.
#         Bev_data (pd.DataFrame): DataFrame mit den Bevölkerungsdaten.
#         technik (str): Technik, für die der Faktor berechnet werden soll.

#     Returns:
#         pd.DataFrame: DataFrame mit den berechneten Faktoren für die jeweilige Technik.
#     """

#     #Bev_data = Bev_data.set_index('GEN')
#     #  Extrahiere die Technik-Faktoren für jedes Bundesland -> verhindert mehrfaches Suchen in For Loop
#     technik_faktoren = Bev_data[technik]

#     # Dictionary für die Gesamtgewichte -> verhindert mehrfaches Suchen in For Loop
#     gesamtgewicht_dict = {}
#     for kat in kategorien_eigenschaften.columns:
#         eigenschaften = kategorien_eigenschaften[kat].dropna()
#         faktoren_kat = [factors[attr] for attr in eigenschaften]
        
#         for land in Bev_data.index:
#             land_werte = Bev_data.loc[land, eigenschaften]
#             gewicht = (land_werte * faktoren_kat).sum()
#             gesamtgewicht_dict[(land, kat)] = gewicht

#     # Leeres DataFrame mit Zeilen wie df, für jedes Land eine Spalte
#     faktor_pro_bus = pd.Series(index=df.index, dtype=float)

#     # Iteriere über jeden Bus im DataFrame
#     for idx, row in df.iterrows():
#         land = row['lan_name']

#         faktor_pro_bus.at[idx] = gewichtungsfaktor(land, kategorien_eigenschaften, factors, row, technik_faktoren, gesamtgewicht_dict)


#     # Berechnung für Bundesland
#     summen = df.iloc[:, 1:].agg(agg_dict)
#     erste_spalte = {df.columns[0]: df.iloc[0, 0]}
#     neue_zeile = {**erste_spalte, **summen.to_dict()}    
#     zeile = pd.DataFrame([neue_zeile])

#     land_bbox = zeile.iloc[0]['lan_name']
#     faktor_bbox = gewichtungsfaktor(land_bbox, kategorien_eigenschaften, factors, zeile.iloc[0], technik_faktoren, gesamtgewicht_dict)

#     return faktor_pro_bus, faktor_bbox



def permission(buses: pd.DataFrame, pfad: str) -> pd.DataFrame:
    """
    Assigns each bus in the network its corresponding 5 km grid ID based on registration district data.
    
    Args:
        buses (pd.DataFrame): DataFrame containing the bus data of the network.
        path (str): Path to the directory containing the GeoJSON file with the 5 km grid or registration zone data.
        
    Returns:
        pd.Series: A pandas Series containing the 'Schluessel_Zulbz' ID (registration district key) for each bus.
    """

    data = gpd.read_file(f"{pfad}/input/FZ Pkw mit Elektroantrieb Zulassungsbezirk_-8414538009745447927.geojson")

    # Convert bus coordinates into a GeoDataFrame
    gdf_buses = gpd.GeoDataFrame(buses.copy(), geometry=gpd.points_from_xy(buses["x"], buses["y"]), crs=data.crs)

    # Perform spatial join (find which district each bus is located in)
    gdf_joined = gpd.sjoin(gdf_buses, data[["Schluessel_Zulbz", "geometry"]], how="left", predicate="within")

    # Drop duplicate indices to keep only unique bus entries
    gdf_joined = gdf_joined[~gdf_joined.index.duplicated(keep='first')]

    # Extract the registration district ID for each bus
    id = gdf_joined["Schluessel_Zulbz"].copy()

    return id


def faktoren(buses: pd.DataFrame, gcp_factors: pd.DataFrame, data: pd.DataFrame, bbox_zensus: pd.Series, technik: str, id_df: pd.Series) -> tuple[pd.Series, float]:
    """
    Calculates technology distribution factors for buses and the bounding box 
    based on census and population data.
    
    Args:
        buses_zensus (pd.DataFrame): DataFrame containing census-related data for each bus.
        gcp_factors (pd.DataFrame): DataFrame with technology weighting factors for each census attribute.
        Bev_data (pd.DataFrame): DataFrame containing population data and associated census identifiers.
        bbox_zensus (pd.Series): Series with aggregated census data for the bounding box area.
        technik (str): The technology for which factors are to be calculated.
        id_df (pd.Series): Series mapping each bus to its census region ID.

    Returns:
        tuple:
            - pd.Series: A Series containing the calculated technology factors for each bus.
            - float: The calculated factor for the bounding box.
    """

    # Initialize Series for bus factors
    arr_factor = pd.Series(0.0, index=buses.index)

    # Filter population data for zensus-related columns
    data_zensus = data[[col for col in data.columns if col.startswith("Zensus")]].copy()

    # Align data columns with technology factor structure
    data_zensus = data_zensus[list(gcp_factors.columns)]
    buses = buses[list(gcp_factors.columns)]

    # Extract technology-specific population data
    data_gcp = data[technik].copy()
    # Compute factor for the area
    id = id_df.iloc[0]
    factor_area = data_zensus.loc[id] @ gcp_factors.loc[technik]

    # Calculate factors for each bus
    for j, zensus in buses.iterrows():
        id = id_df.loc[j]
        factor_bus = zensus @ gcp_factors.loc[technik]
        if factor_bus < 0:
            factor_bus = 0
        arr_factor.loc[j] = factor_bus / factor_area * data_gcp.loc[id]

    # Compute factor for the bounding box
    factor_bbox = bbox_zensus @ gcp_factors.loc[technik] / factor_area * data_gcp.loc[id]
    if factor_bbox < 0:
        factor_bbox = 0
    
    return arr_factor, factor_bbox
        


def sort_gcp(buses: pd.DataFrame, gcp: str, amount_total: float, solar_power: float) -> pd.DataFrame:
    """
    Distributes a given technology (e.g., solar, heat pump, e-car) across buses 
    in the network based on their factor values.

    Args:
        buses (pd.DataFrame): DataFrame containing bus data.
        gcp (str): The technology to be distributed (e.g., "solar", "HP", "E_car").
        amount_total (float): The total number or capacity to be distributed among buses.
        solar_power (float): The installed solar capacity (in kW) for the postal code area.

    Returns:
        pd.DataFrame: Updated DataFrame with the distributed technology power assigned.
    """

    # Initialize column for technology power
    buses['Power_' + gcp] = 0.0  # Initialisierung mit 0

    # Group buses by their gcp factor (e.g., Factor_solar) and sort descending
    buses_grouped = buses.groupby('Factor_' + gcp)
    groups_dict = {key: group for key, group in sorted(buses_grouped, key=lambda x: x[0], reverse=True)}

    # Temporary Series to collect power values
    power_col = pd.Series(0.0, index=buses.index)

    # Handle case where solar data already exists (from MaStR or similar)
    if gcp == 'solar' and "type_1" in buses.columns:
        mask_type1 = buses["type_1"] == gcp
        power_col.loc[mask_type1] = solar_power

        # Count already assigned solar units and adjust total amount
        amount = mask_type1.sum()

        # Reduce amount_total by the already existing capacity
        amount_total -= amount

        # If no more solar units to distribute, return early
        if amount_total <= 0:
            buses['Power_' + gcp] = power_col
            return buses

    # Distribute remaining gcp
    p = 0
    for key in groups_dict.keys():
        group = groups_dict[key]

        bus_list = []
        if gcp == 'E_car':
            for bus in group.index:
                # Add bus index according to the number of households
                bus_list.extend([bus] * int(buses.loc[bus, 'Haushalte']))


        else:
            # Add bus index
            bus_list = group.index.tolist()
        
        # Sample subset of buses randomly (sample size limited by available elements)
        sample_size = int(min(key, len(bus_list)))
        sampled_buses_index = random.sample(bus_list, sample_size)

        # Assign gcp power based on type
        if gcp == 'solar':
            power_col.loc[sampled_buses_index] = solar_power
        else:
            power_col.loc[sampled_buses_index] += key

        # Check if total amount exceeded and adjust
        p += sample_size
        if p > amount_total:
            rest = int(p - amount_total)
            # Randomly select 'rest' number of buses to set their power back to 0
            rest_buses_index = random.sample(sampled_buses_index, rest)
            power_col.loc[rest_buses_index] = 0
            break

    # Merge calculated power back into the buses DataFrame
    buses['Power_' + gcp] = power_col

    return buses


def solar_orientation(buses: pd.DataFrame, plz: str, pv_plz: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns orientation and tilt angles to buses with solar technology based on postal code (PLZ) statistics.

    Args:
        buses (pd.DataFrame): 
            DataFrame containing bus data, including the column 'Power_solar'.
        plz (str): 
            The postal code for which the solar orientation and tilt should be applied.
        pv_plz (pd.DataFrame): 
            DataFrame containing PLZ-level PV installation data with columns:
            - 'Hauptausrichtung_Anteil': dict-like string of orientation probabilities (e.g. {"S": 0.6, "SW": 0.3, "W": 0.1})
            - 'HauptausrichtungNeigungswinkel_Anteil': dict-like string of tilt angle probabilities.

    Returns:
        pd.DataFrame: 
            Updated DataFrame with two new columns for solar-equipped buses:
            - 'Hauptausrichtung_Anteil' (main orientation)
            - 'HauptausrichtungNeigungswinkel_Anteil' (main tilt angle)
    """


    # Extract orientation and tilt distributions for the specified PLZ
    orientation_dist = ast.literal_eval(pv_plz.loc[plz, 'Hauptausrichtung_Anteil'])
    tilt_dist = ast.literal_eval(pv_plz.loc[plz, 'HauptausrichtungNeigungswinkel_Anteil'])

    # Filter buses with solar
    df = buses[buses["Power_solar"] != 0].copy()

    # Randomly assign orientation and tilt based on probability distributions
    df["Hauptausrichtung_Anteil"] = np.random.choice(list(orientation_dist.keys()), size=len(df), p=list(orientation_dist.values()))
    df["HauptausrichtungNeigungswinkel_Anteil"] = np.random.choice(list(tilt_dist.keys()), size=len(df), p=list(tilt_dist.values()))

    # Write results back into main DataFrame
    buses.loc[df.index, 'Hauptausrichtung_Anteil'] = df['Hauptausrichtung_Anteil']
    buses.loc[df.index, 'HauptausrichtungNeigungswinkel_Anteil'] = df['HauptausrichtungNeigungswinkel_Anteil']

    return buses


def storage(buses: pd.DataFrame, storage_pv: float) -> pd.DataFrame:
    """
    Adds a 'storage' column to the buses DataFrame, representing the storage capacity 
    associated with solar installations.

    Args:
        buses (pd.DataFrame): 
            DataFrame containing bus data, including the column 'Power_solar'.
        storage_pv (float): 
            Probability (between 0 and 1) that a solar bus has an associated storage system. 
            Derived from the market master data register.

    Returns:
        pd.DataFrame: 
            Updated DataFrame with a new column 'storage', indicating assigned storage capacity (in kWh).
    """
    
    prob = storage_pv 
    buses = buses.copy()
    buses['storage'] = 0.0

    # Randomly assign storage systems to a fraction of solar-equipped buses
    solar_index = buses[buses['Power_solar'] != 0].sample(frac=prob).index
    # 1 kWp PV capacity corresponds to 1 kWh storage capacity
    buses.loc[solar_index, 'storage'] = buses.loc[solar_index, 'Power_solar'] * 1.0

    return buses




def relative_humidity(t: float, td: float) -> float:
    """
    Calculates the relative humidity based on the air temperature and dew point temperature.
    
    Args:
        t (float): Air temperature in degrees Celsius.
        td (float): Dew point temperature in degrees Celsius.
        
    Returns:
        float: Relative humidity as a percentage (%).
    """

    # Both t and td are in °C
    es = 6.112 * np.exp((17.67 * t) / (t + 243.5))  # Saturation vapor pressure
    e = 6.112 * np.exp((17.67 * td) / (td + 243.5)) # Vapor pressure
    rh = 100 * e / es

    return rh


def env_weather(bbox: list, path: str, time_discretization: int = 3600, timesteps_horizon: int = 8760, timesteps_used_horizon: int = 8760, timesteps_total: int = 8760) -> Environment: #, year):
    """
    Loads ERA5 weather data for a given bounding box and computes relevant environmental parameters.
    
    Args:
        bbox (list): Coordinates of the bounding box [W, S, E, N].
        time_discretization (int): Time step in seconds (default 3600s = 1 hour).
        timesteps_horizon (int): Number of timesteps in the planning horizon (default 8760 = 1 year).
        timesteps_used_horizon (int): Number of timesteps actually used (default 8760 = 1 year).
        timesteps_total (int): Total number of timesteps (default 8760 = 1 year).
        
    Returns:
        Environment: An Environment object containing weather data and computed parameters.
    """

    # Variables to load from ERA5 datasets
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

    # Load datasets into a dictionary
    datasets = {}
    for var, cod in variables:
        filename = 'GER_' +var + '.nc'
        ds = xr.open_dataset(os.path.join(path, 'input/weather_2013', filename))
        datasets[var] = ds
        print(f"Variable {var} verarbeitet.")

    # Compute bounding box center
    # bbox: [N, W, S, E] [left, bottom, right, top]
    lat_target = (bbox[3] + bbox[1]) / 2
    lon_target = (bbox[0] + bbox[2]) / 2

    data_dict = {}

    # Select nearest grid point for each variable
    for var, cod in variables:
        data = datasets[var].sel(latitude=lat_target, longitude=lon_target, method='nearest')
        df = data.to_dataframe()
        data_dict[var] = df[cod]


    # Unit conversions
    # Temperature: Kelvin → Celsius
    data_dict["2m_temperature"] = data_dict["2m_temperature"] - 273.15
    data_dict["2m_dewpoint_temperature"] = data_dict["2m_dewpoint_temperature"] - 273.15

    # Pressure: Pa → hPa
    data_dict["surface_pressure"] = data_dict["surface_pressure"] / 100

    # Compute relative humidity
    data_dict["2m_relative_humidity"] = relative_humidity(data_dict["2m_temperature"], data_dict["2m_dewpoint_temperature"])

    # Wind speed calculation
    data_dict["10m_wind_speed"] = (data_dict["10m_u_component_of_wind"]**2 + data_dict["10m_v_component_of_wind"]**2)**0.5

    # Convert solar radiation from J/m² to W/m²
    data_dict["surface_solar_radiation_downwards"] = data_dict["surface_solar_radiation_downwards"] / 3600
    data_dict["total_sky_direct_solar_radiation_at_surface"] = data_dict["total_sky_direct_solar_radiation_at_surface"] / 3600

    # Diffuse solar radiation calculation
    data_dict["surface_diffuse_solar_radiation_at_surface"] = data_dict["surface_solar_radiation_downwards"] - data_dict["total_sky_direct_solar_radiation_at_surface"]

    # Compute cloud cover: scale to 0-8 octas
    data_dict["total_cloud_cover"] = (data_dict["total_cloud_cover"] * 8).round().clip(lower=0, upper=8)

    # Select relevant variables for Environment
    variables_new = [
        "10m_wind_speed",
        "2m_relative_humidity",
        "2m_temperature",
        "surface_pressure",
        "surface_diffuse_solar_radiation_at_surface",
        "total_sky_direct_solar_radiation_at_surface",
        "total_cloud_cover"
    ]

    # Save intermediate CSV files for PyPSA Weather input
    for var in variables_new:
        filename = f"{var}_nearest.txt"
        data_dict[var].to_csv(filename, sep="\t", index=False, header=False)

    # Timer object
    timer = Timer(
        time_discretization=time_discretization,
        timesteps_horizon=timesteps_horizon,
        timesteps_used_horizon=timesteps_used_horizon,
        timesteps_total=timesteps_total
    )

    # Weather object for PyPSA
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

    # Clean up temporary CSV files
    for var in variables_new:
        filename = f"{var}_nearest.txt"
        os.remove(filename)

    return environment

# %%
