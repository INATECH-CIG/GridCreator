#%% Import
 
import geopandas as gpd
from pyproj import Transformer
import pandas as pd
import osmnx as ox
#import pystatis # noch nicht installiert im env
import math
import scipy.spatial as spatial
import numpy as np
import json

#%% Funktionen zum aufrufen



def get_osm_data(bbox):
    """
    Ruft OSM-Daten für ein definiertes Rechteck ab.
    
    Parameter:
    north (float): Nördliche Grenze des Rechtecks (Breitengrad)
    south (float): Südliche Grenze des Rechtecks (Breitengrad)
    east (float): Östliche Grenze des Rechtecks (Längengrad)
    west (float): Westliche Grenze des Rechtecks (Längengrad)
    
    Returns:
    GeoDataFrame: Enthält die OSM-Daten für das definierte Rechteck
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
    

    # tags ={"building": True,
    #        "addr:postcode": True
    #        }

    # Area_features = ox.features_from_bbox(bbox, tags)


    # # Nur bestimmte Spalten behalten
    # # welche spalten wollen wir alles? Ansonsten entsteht Fehler, weil Listen in Tabelle nicht als geojason gespeichert werden können
    # columns_to_keep = ['geometry', 'building', 'name', 'addr:postcode']
    # Area_features_clean = Area_features[columns_to_keep].copy()

    Area = ox.graph.graph_from_bbox(bbox, network_type="all")
    Area_features = ox.features_from_bbox(bbox, tags=tags)
    return Area, Area_features


def compute_bbox_from_buses(net):
    """
    Berechnet die Bounding Box (bbox) aus den Koordinaten der Busse im Netz.

    Parameters:
    net (object): Ein Netzobjekt, das eine Attribut 'buses' enthält, welches ein DataFrame mit den Spalten 'x' und 'y' hat.

    Returns:
    list: Eine Liste mit den Koordinaten der Bounding Box in der Form [left, bottom, right, top].
    """
    x_min = net.buses['x'].min()
    x_max = net.buses['x'].max()
    y_min = net.buses['y'].min()
    y_max = net.buses['y'].max()

    # Optional als [left, bottom, right, top]
    bbox = [x_min, y_min, x_max, y_max]
    return bbox



def Bundesland(net_buses, data):

    """
    Weist jedem Bus im Netz das Bundesland zu, in dem er sich befindet.
    
    Parameters:
    net_buses (DataFrame): DataFrame mit den Bussen des Netzes, die die Spalten 'x' und 'y' enthalten.
    data (DataFrame): DataFrame mit den Referenzpunkten, die die Spalten 'geo_point_2d' (als JSON-String) und 'lan_name' enthalten.
    
    Returns:
    DataFrame: Ein DataFrame mit den Bussen des Netzes, erweitert um die Spalte 'Bundesland'.
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

    # 4. Bundeslandnamen zuordnen
    net_buses["Bundesland"] = data.iloc[idx]["lan_name"].values

    return net_buses



def WG_zu_Gauss(lon, lat):
    """
    Umrechnung von WGS84 (Längengrad, Breitengrad) zu Gauß-Krüger Zone 3 (EPSG:31467)

    Parameters:
    lon (float): Längengrad in WGS84
    lat (float): Breitengrad in WGS84

    Returns:
    tuple: (rechtswert, hochwert) in Gauß-Krüger Koordinaten
    """

    transformer_zu_Gauss = Transformer.from_crs("EPSG:4326", "EPSG:31467", always_xy=True)
    rechtswert, hochwert = transformer_zu_Gauss.transform(lon, lat)
    return rechtswert, hochwert


def Gauss_zu_WG(rechtswert, hochwert):
    """
    Umrechnung von Gauß-Krüger Zone 3 (EPSG:31467) zu WGS84 (Längengrad, Breitengrad)
    
    Parameters:
    rechtswert (float): Rechtswert in Gauß-Krüger Koordinaten
    hochwert (float): Hochwert in Gauß-Krüger Koordinaten
    
    Returns:
    tuple: (lon, lat) in WGS84 Koordinaten
    """

    transformer_zu_wg = Transformer.from_crs("EPSG:31467", "EPSG:4326", always_xy=True)
    lon, lat = transformer_zu_wg.transform(rechtswert, hochwert)
    return lon, lat



def gitter_ID(net_buses, data):

    """
    Weist jedem Bus im Netz die Gitter-ID des nächstgelegenen Referenzpunkts zu.

    Parameters:
    net_buses (DataFrame): DataFrame mit den Bussen des Netzes, die die Spalten 'x' und 'y' enthalten.
    data (DataFrame): DataFrame mit den Referenzpunkten, die die Spalten 'x_mp_100m', 'y_mp_100m' und 'GITTER_ID_100m' enthalten.
    
    Returns:
    numpy.ndarray: Ein Array mit den Gitter-IDs, die den Bussen zugeordnet
    """

    x_array, y_array = WG_zu_Gauss(net_buses["x"], net_buses["y"])

    # Referenzpunkte in der Zensus-Tabelle
    reference_points = np.vstack((data['x_mp_100m'].values, data['y_mp_100m'].values)).T
    tree = spatial.cKDTree(reference_points)

    # Zielpunkte aus dem Netz
    query_points = np.vstack((x_array, y_array)).T

    # Nächste Nachbarn finden
    distances, indices = tree.query(query_points)

    # Gitter-IDs extrahieren
    gitter_ids = data.iloc[indices]['GITTER_ID_100m'].values

    return gitter_ids



def zenus_daten(net_buses, data, spaltenname):

    """
    Weist jedem Bus im Netz die Zensus-Daten anhand der Gitter-ID zu.
    
    Parameters:
    net_buses (DataFrame): DataFrame mit den Bussen des Netzes, die die Spalte 'Zensus_ID' enthalten.
    data (DataFrame): DataFrame mit den Zensus-Daten, die die Spalten 'GITTER_ID_100m' und die gewünschte Spalte enthalten.
    spaltenname (str): Der Name der Spalte in den Zensus-Daten, die zugeordnet werden soll.
    
    Returns:
    DataFrame: Ein DataFrame mit den Bussen des Netzes, erweitert um die Zensus-Daten.
    """

    net_buses = net_buses.merge(data[["GITTER_ID_100m", spaltenname]], left_on="Zensus_ID", right_on="GITTER_ID_100m", how="left")
    # Umbenennen der neuen Spalte und löschen von GITTER_ID_100m
    net_buses = net_buses.rename(columns={spaltenname: f"Zensus_{spaltenname}"})
    net_buses = net_buses.drop(columns=["GITTER_ID_100m"])

    return net_buses
