#%% Import
 
import geopandas as gpd
from pyproj import Transformer
import pandas as pd
import osmnx as ox
#import pystatis # noch nicht installiert im env
import math

#%% Funktionen zum aufrufen



def get_osm_data(north, south, east, west):
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

#     tags = {
#     "aerialway": True,        # Seilbahnen
#     "aerodrome": True,        # Flugplätze
#     "amenity": True,          # Annehmlichkeiten
#     "aeroway": True,          # Luftfahrt-Infrastruktur
#     "barrier": True,          # Barrieren
#      "building": True,         # Gebäude
#     "bus_station": True,      # Bushaltestellen
#     "church": True,           # Kirchen
#     "craft": True,            # Handwerk
#     "emergency": True,        # Notfallinfrastruktur
#     "geological": True,       # Geologische Merkmale
#     "healthcare": True,       # Gesundheitseinrichtungen
#     "historic": True,         # Historische Orte
#     "highway": True,          # Straßen
#     "leisure": True,          # Freizeiteinrichtungen
#     "landcover": True,        # Landbedeckung
#     "landuse": True,          # Landnutzung
#     "Man Made": True,         # Menschliche Strukturen
#     "man_made": True,         # Menschliche Strukturen
#     "military": True,         # Militärische Einrichtungen
#     "natural": True,          # Natürliche Merkmale
#     "office": True,           # Büros
#     "place": True,            # Orte
#     "power": True,            # Energieinfrastruktur
#     "public_transport": True, # Öffentliche Verkehrsmittel
#     "railway": True,          # Eisenbahninfrastruktur
#     "route": True,            # Routen
#     "shop": True,             # Geschäfte
#     "school": True,           # Schulen
#     "tourism": True,          # Touristische Orte
#     'telecom': True,         # Telekommunikationseinrichtungen
#     "waterway": True,         # Gewässer
#     }

    tags ={"building": True,
           "addr:postcode": True
           }

    Area_features = ox.geometries_from_bbox(north, south, east, west, tags)


    # Nur bestimmte Spalten behalten
    # welche spalten wollen wir alles? Ansonsten entsteht Fehler, weil Listen in Tabelle nicht als geojason gespeichert werden können
    columns_to_keep = ['geometry', 'building', 'name', 'addr:postcode']
    Area_features_clean = Area_features[columns_to_keep].copy()
    return Area_features_clean



def Bundesland(daten, plz):
    """
    Ermittelt das Bundesland für eine gegebene Postleitzahl (PLZ) aus den Zensusdaten.

    Parameter:
    daten (DataFrame): DataFrame der Zensusdaten mit PLZ und Bundesland
    plz (str): Postleitzahl, für die das Bundesland ermittelt werden soll

    Returns:
    str: Name des Bundeslandes, das der angegebenen PLZ zugeordnet ist
    """

    return daten.loc[daten["name"] == plz, "lan_name"].values[0]



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



def gitter_ID(data, xpunkt, ypunkt):
    """
    Zuordnung von Nodes zu Gitter ID basierend auf den Koordinaten (xpunkt, ypunkt).

    Parameters:
    data (DataFrame): DataFrame der Zensusdaten.
    xpunkt (float): X-Koordinate des Punktes.
    ypunkt (float): Y-Koordinate des Punktes.

    Returns:
    tuple: (x_mp_100m, y_mp_100m, GITTER_ID_100m) Koordinaten der Mitte dder Zelle und Gitter ID.
    """

    distance = [math.dist((x,y), (xpunkt, ypunkt)) for x,y in zip(data['x_mp_100m'],data['y_mp_100m'])]
    point_index = distance.index(min(distance))
    
    return data['x_mp_100m'][point_index], data['y_mp_100m'][point_index], data['GITTER_ID_100m'][point_index]


"""
Einwohner je Gitter ID
"""
def get_population_count(data, gitter_id):
    """
    Gibt die Einwohnerzahl für eine bestimmte Gitter-ID zurück.

    Parameters:
    data (DataFrame): DataFrame der Zensusdaten
    gitter_id (int): Gitter-ID, für die die Einwohnerzahl abgefragt wird

    Returns:
    int: Einwohnerzahl für die angegebene Gitter-ID
    """

    return data.loc[data['GITTER_ID_100m'] == gitter_id, 'Einwohner'].values[0]
