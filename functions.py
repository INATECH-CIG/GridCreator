#%% Import
 
import geopandas as gpd
from pyproj import Transformer
import pandas as pd
import osmnx as ox
import scipy.spatial as spatial
import numpy as np
import json
from tqdm import tqdm
from functools import reduce

#%% Funktionen zum aufrufen


#%% Funktionen zum aufrufen

def get_osm_data(bbox):
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


def compute_bbox_from_buses(net):
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


def bundesland(net_buses, data):
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



def epsg4326_zu_epsg3035(lon, lat):
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

def epsg32632_zu_epsg3035(lon, lat):
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


def zensus_df_verbinden(daten_zensus):
    """
    Verbindet mehrere DataFrames basierend auf der GITTER_ID_100m Spalte.
    
    Args:
        daten_zensus (dict): Ein Dictionary, das DataFrames enthält, die auf GITTER_ID_100m basieren.
    
    Returns:
        pd.DataFrame: Ein DataFrame, das die Daten aus allen DataFrames kombiniert.
    """

    # Start mit den gemeinsamen Spalten aus einem beliebigen DF
    dfs = list(daten_zensus.values())

    # Sicherstellen, dass alle DFs die gleichen drei Schlüsselspalten haben
    basis_spalten = ["GITTER_ID_100m", "x_mp_100m", "y_mp_100m"]

    # Alle DFs vorbereiten (nur einmalige ID-Spalten behalten)
    df_bereinigt = []
    for df in tqdm(dfs, desc="DataFrames bereinigen"):
        # Doppelte Spalten entfernen, außer den 3 gemeinsamen
        eigene_spalten = [col for col in df.columns if col not in basis_spalten]
        df_neu = df[basis_spalten + eigene_spalten]
        df_bereinigt.append(df_neu)

    # DataFrames per GITTER_ID_100m zusammenführen
    df_vereint = reduce(lambda left, right: pd.merge(left, right, on=basis_spalten, how='outer'), df_bereinigt)

    return df_vereint




def daten_zuordnen(net_buses, data):
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
    data = data.rename(columns=lambda col: f"Zensus_{col}").copy()
    
    # Passende Zeilen aus data holen
    matched_data = data.iloc[indices].copy()

    # Index an net_buses anpassen, damit concat funktioniert
    matched_data.index = net_buses.index

    # DataFrames nebeneinander anfügen
    result = pd.concat([net_buses, matched_data], axis=1)

    return result



def load_shapes(datei):
    """
    Lädt die Bundesland-Geometrien aus einer GeoJSON-Datei und filtert sie auf die relevanten Bundesländer.
    Args:
        datei (str): Pfad zur GeoJSON-Datei mit den Bundesland-Geometrien.
        
    Returns:
        gpd.GeoDataFrame: Ein GeoDataFrame mit den Geometrien der relevanten Bundesländer.
    """

    # Laden der Geometrien aus der Datei
    shapes = gpd.read_file(datei, layer="vg5000_lan")
    
    bundeslaender = [
        'Schleswig-Holstein', 'Hamburg', 'Niedersachsen', 'Bremen',
        'Nordrhein-Westfalen', 'Hessen', 'Rheinland-Pfalz',
        'Baden-Württemberg', 'Bayern', 'Saarland', 'Berlin',
        'Brandenburg', 'Mecklenburg-Vorpommern', 'Sachsen',
        'Sachsen-Anhalt', 'Thüringen'
    ]
    # Filtern der Geometrien nach Bundesländern
    laender = shapes[shapes["GEN"].isin(bundeslaender)]
    # Filtern der Geometrien
    laender_aggregiert = laender.dissolve(by="GEN")
    laender_umrisse = laender_aggregiert[["geometry"]]
    laender_3035 = laender_umrisse.to_crs(epsg=3035)

    return laender_3035

"""
NICHT ANWENDBAR !!!!
NEUER MECHANISMUS WICHTIG

Zensus_Fernheizung sind Zahlen zwischen 0 un 100, aber absolut und nicht relativ!
Zensus_Etagenheizung enthält Zahlen größer 100 -> kann auch Glück haben

Wie unterscheide ich also sicher automatisch zwischen MW und SUMME?
"""
def sum_mw(df):
    """
    Bildet den Mittelwert der Werte in einem DataFrame, für Spalten mit Werten zwischen 0 und 1 oder 0 und 100.
    Ansonsten bildet es die Summe der Werte.

    Args:
        df (pd.DataFrame): DataFrame mit den zu summierenden Werten.
        
    Returns:
        pd.DataFrame: Ein DataFrame mit der Summe der Werte für jede Spalte.
    """

    result = pd.DataFrame(columns=df.columns)
    row = {}

    for col in df.columns:
        values = df[col]
        if values.between(0, 1).all() or values.between(0, 100).all():
            row[col] = values.mean()
        else:
            row[col] = values.sum()

    result.loc[0] = row
    return result



def bundesland_zuordnung(zensus, shapes):
    """
    Ordnet die Zensusdaten den Bundesländern zu, basierend auf den Geometrien der Bundesländer.
    Args:
        zensus (pd.DataFrame): DataFrame mit den Zensusdaten.
        shapes (gpd.GeoDataFrame): GeoDataFrame mit den Geometrien der Bundesländer.
        
    Returns:
        pd.DataFrame: Ein DataFrame mit den zugeordneten Bundesland-Daten."""

    # Zensusdaten vorbereiten -> Georeferenz einbauen 
    spalten_liste = zensus.columns[3:].tolist()
    zensus['geometry'] = gpd.points_from_xy(zensus['x_mp_100m'], zensus['y_mp_100m'])
    gdf_punkte = gpd.GeoDataFrame(zensus, geometry=zensus['geometry'], crs='EPSG:3035').copy()

    # Daten verbinden
    punkte_mit_bl = gpd.sjoin(gdf_punkte, shapes, how='left', predicate='within')
    punkte_mit_bl[spalten_liste] = punkte_mit_bl[spalten_liste].apply(lambda col: pd.to_numeric(col.astype(str).str.replace(',', '.', regex=False), errors='coerce')).fillna(0.0)

    # Jetzt gruppieren und aufsummieren
    # Statt reines aufsummieren Funktionsaufruf für MW oder SUMME
    # Also erst gruppieren und dann einzelne gruppen aufsummieren in for schleife?
    # Eine Gruppe/Bundesland = bbox

    #ergebnis = punkte_mit_bl.groupby('GEN')[spalten_liste].sum(min_count=1).reset_index()

    ergebnis = punkte_mit_bl.groupby('GEN')[spalten_liste].apply(sum_mw).reset_index()

    # Spalten umbenennen
    ergebnis.columns = [ergebnis.columns[0]] + [f"Zensus_{col}" for col in ergebnis.columns[1:]]

    return ergebnis


def gewichtungsfaktor(land, kategorien_eigenschaften, factors, row, technik_faktoren, gesamtgewicht_dict):
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
        faktoren_kat = pd.Series({attr: factors[kat][attr] for attr in eigenschaften})
        werte_kat = row[eigenschaften]

        # numerische Multiplikation
        sum_bus = (werte_kat * faktoren_kat).sum()
        sum_land = gesamtgewicht_dict.get((land, kat), 0)

        # Berechnung von Gewichtungsprodukt
        if sum_land != 0:
            weighted_sum *= sum_bus / sum_land
        else:
            weighted_sum *= 0  

    return weighted_sum * technik_faktoren.loc[land]




def calculate_factors(df, factors, kategorien_eigenschaften, Bev_data, technik):

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

    Bev_data = Bev_data.set_index('GEN')
    #  Extrahiere die Technik-Faktoren für jedes Bundesland -> verhindert mehrfaches Suchen in For Loop
    technik_faktoren = Bev_data[technik]

    # Dictionary für die Gesamtgewichte -> verhindert mehrfaches Suchen in For Loop
    gesamtgewicht_dict = {}
    for kat in kategorien_eigenschaften.columns:
        eigenschaften = kategorien_eigenschaften[kat].dropna()
        faktoren_kat = pd.Series({attr: factors[kat][attr] for attr in eigenschaften})
        
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

    # Berechnung von Zensus-Gesamt bbox
    # Lässt sich eventuell mit bundesland zensus [bundesland_zensus()] verbinden?
    # Ja siehe ChatGPT: ZENSUS VERBINDEN
    # summen müsste angepasst werden: Funktionsaufruf für MW oder SUMME
    # summen = df.iloc[:, 1:].sum()
    summen = sum_mw(df.iloc[:, 1:])
    erste_spalte = {df.columns[0]: df.iloc[0, 0]}
    # neue_zeile = {**erste_spalte, **summen.to_dict()}
    neue_zeile = {**erste_spalte, **summen.iloc[0].to_dict()}    
    zeile = pd.DataFrame([neue_zeile])


    faktor_bbox = gewichtungsfaktor(land, kategorien_eigenschaften, factors, zeile.iloc[0], technik_faktoren, gesamtgewicht_dict)

    return faktor_pro_bus, faktor_bbox



def technik_sortieren(grid, Technik, p_total):
    """
    Füllt das Grid-Objekt mit der angegebenen Technik basierend auf den gegebenen Anteilen.
    Args:
        grid (pypsa.Network): Das Grid-Objekt, das die Buslinien enthält.
        Technik (str): Die Technik, die zugeordnet werden soll.
        p_total (float): Der Gesamtanteil, der der Technik zugeordnet werden soll.
        
    Returns:
        pypsa.Network: Das aktualisierte Grid-Objekt mit den zugeordneten Techniken.
    """

    mask_type1 = grid.buses["type_1"] == Technik
    p_bbox = grid.buses.loc[mask_type1, "p_nom_1"].sum(min_count=1)
    amount = mask_type1.sum()

    if "type_2" in grid.buses.columns:
        mask_type2 = grid.buses["type_2"] == Technik
        p_bbox += grid.buses.loc[mask_type2, "p_nom_2"].dropna().sum()
        amount += mask_type2.sum()

    # Falls keine Busse mit Technik gefunden wurden:
    if amount == 0:
        return grid
    


    # Leistung, die noch verteilt werden muss
    p_rest = p_total - p_bbox
    if p_rest <= 0:
        # Alles verteilt, nichts zu tun
        return grid
    
    # Mittelwert der vorhandenen Leistung pro Bus
    p_mean = p_bbox / amount if amount > 0 else 0


    bus_sorted = grid.buses[(grid.buses['p_nom_1'].isna()) & (grid.buses['type_1'] != Technik)].copy()
    bus_sorted = bus_sorted.sort_values(by=['Factor_'+Technik], ascending = False)
    #to_fill = bus_sorted[bus_sorted['p_nom_1'].isna()].index.tolist()
    to_fill = bus_sorted.index.tolist()


    verteilte_leistung = 0

    for idx in to_fill:
        if verteilte_leistung + p_mean > p_rest:
            break
        grid.buses.at[idx, 'p_nom_1'] = p_mean
        verteilte_leistung += p_mean

    return grid

# %%
