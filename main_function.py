
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
from typing import List, Tuple
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


def ding0_grid(bbox: list[float], grids_dir: str, output_file_grid: str) -> tuple[pypsa.Network, list[float]]:
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


def osm_data(net: pypsa.Network, bbox_neu: list[float], buffer: float) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
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
    # Area_features.to_file("Area_features.geojson", driver="GeoJSON")

    Area_features_df = Area_features.reset_index()

    # Daten kombinieren
    buses_df = dc.data_combination(net, Area_features_df)

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




def technik_zuordnen(buses: pd.DataFrame, file_Faktoren: str, file_solar: str, file_ecar: str, file_hp: str, technik_arr: list[str], pfad: str) -> tuple[pd.DataFrame, list[float]]:
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
    
    
    Technik_Faktoren = pd.read_csv(file_Faktoren)
    Technik_Faktoren = Technik_Faktoren.set_index("Technik")

    Bev_data_solar = pd.read_csv(file_solar, sep=",")
    Bev_data_solar["PLZ"] = Bev_data_solar["PLZ"].astype(int).astype(str).str.zfill(5)
    Bev_data_solar.set_index("PLZ", inplace=True)


    Bev_data_ecar = pd.read_csv(file_ecar, sep=",")
    Bev_data_ecar["Schluessel_Zulbz"] = Bev_data_ecar["Schluessel_Zulbz"].astype(int).astype(str).str.zfill(5)
    Bev_data_ecar.set_index("Schluessel_Zulbz", inplace=True)


    Bev_data_hp = pd.read_csv(file_hp, sep=",")
    Bev_data_hp.set_index("GEN", inplace=True)

    #Bev_data_Zensus.set_index("GEN", inplace=True)

    buses_zensus = buses[[col for col in buses.columns if col.startswith("Zensus")]].copy()
    buses_zensus.drop(columns=["Zensus_Einwohner"], inplace=True)
    # Kommas durch Punkte ersetzen, damit pd.to_numeric klappt; In float umwandeln; Fehlende Werte auf 0 setzen
    buses_zensus = (buses_zensus.astype(str).replace(",", ".", regex=True).apply(pd.to_numeric, errors="coerce").fillna(0.0))
    bbox_zensus = buses_zensus.agg(agg_dict)

    # bbox Faktor folgt nur aus erster Zeile von jeder Gruppe
    # Nach Gitter-ID gruppieren
    buses_zensus['GITTER_ID_100m'] = buses['GITTER_ID_100m']
    buses_zensus_grouped = buses_zensus.groupby(['GITTER_ID_100m'])

    bbox_zensus_df = pd.DataFrame()
    for name, group in buses_zensus_grouped:
        df = group.iloc[[0]].copy()
        bbox_zensus_df = pd.concat([bbox_zensus_df, df])

    bbox_zensus = bbox_zensus_df.agg(agg_dict)

    # Gitter ID löschen
    buses_zensus.drop(columns=['GITTER_ID_100m'], inplace=True)

    factor_bbox = np.array([0.0] * len(technik_arr))
    
    # Berechnung Solar
    if 'solar' in technik_arr:
        buses_plz = buses['plz_code'].copy()
        technik = 'solar'
        i = technik_arr.index(technik)
        buses['Factor_' + technik], factor_bbox[i]  = func.faktoren(buses_zensus, Technik_Faktoren, Bev_data_solar, bbox_zensus, 'solar', buses_plz)


    # Berechnung E-Car

    '''
    Die ID muss den buses noch hinzugefügt werden
    '''
    if 'E_car' in technik_arr:
        buses_zulassung = func.zulassung(buses, pfad)
        technik = 'E_car'
        i = technik_arr.index(technik)
        buses['Factor_' + technik], factor_bbox[i]  = func.faktoren(buses_zensus, Technik_Faktoren, Bev_data_ecar, bbox_zensus, 'E_car', buses_zulassung)



    # Berechnung HP
    if 'HP' in technik_arr:
        buses_land = buses['lan_name'].copy()
        technik = 'HP'
        i = technik_arr.index(technik)
        buses['Factor_' + technik], factor_bbox[i]  = func.faktoren(buses_zensus, Technik_Faktoren, Bev_data_hp, bbox_zensus, 'HP', buses_land)


    return buses, factor_bbox



def wohnungen_zuordnen(buses):
        

    # Neue Spalten erstellen für Haustypen
    buses['Bus_1_Wohnung'] = 0
    buses['Bus_2_Wohnungen'] = 0
    buses['Bus_3bis6_Wohnungen'] = 0
    buses['Bus_7bis12_Wohnungen'] = 0
    buses['Bus_13undmehr_Wohnungen'] = 0
    buses['Bus_1_Person'] = 0
    buses['Bus_2_Personen'] = 0
    buses['Bus_3_Personen'] = 0
    buses['Bus_4_Personen'] = 0
    buses['Bus_5_Personen'] = 0
    buses['Bus_6_Personen_und_mehr'] = 0
    buses['Haushalte'] = 0
    buses['Bewohnerinnen'] = 0

    # gruppieren nach GITER_ID_100m
    buses = buses.fillna(0)
    buses = buses.replace("–", 0)
    buses_grouped = buses.groupby('GITTER_ID_100m')
    # Haustypverteilung berechnen


    for name, group in buses_grouped:

        print('GITTER_ID_100m: ', name)
        print('Gruppe, ', group.index)

        node_anzahl = len(group)
        wohnungs_anzahl = sum([float(group.iloc[0]['Zensus_1_Wohnung']),
                        float(group.iloc[0]['Zensus_2_Wohnungen']),
                        float(group.iloc[0]['Zensus_3bis6_Wohnungen']),
                        float(group.iloc[0]['Zensus_7bis12_Wohnungen']),
                        float(group.iloc[0]['Zensus_13undmehr_Wohnungen'])])
        print('wohnungs_anzahl, ', wohnungs_anzahl)
        if wohnungs_anzahl == 0:

            '''
            Wenn keine Angaben da sind, werden die Bewohnerinnen gleichmäßig aufgeteilt
            '''

            bew_pro_node = int(group.iloc[0]['Zensus_Einwohner']) / node_anzahl
            print('bew_pro_node, ', bew_pro_node)
            for i in group.index:
                buses.at[i, 'Bewohnerinnen'] = bew_pro_node
            
            print('Nächste Gruppe')
            continue
            

        else:

            '''
            Wenn Angaben da sind, werden die Bewohnerinnen anteilig aufgeteilt
            Ablauf:
            1. Wohnungsgrößenanteile bestimmen
            2. Jedem Bus eine Wohnungsanzahl zuweisen
            3. Liste mit den Index der Busse erstellen, je nach Wohnungsanzahl
            4. Personenanteile bestimmen
            5. Personen auf Wohnungen anteilig verteilen, bis Einwohnerzahl erreicht ist
            '''

            
            group = group.replace("–", 0)
            typ1_anteil = float(group.iloc[0]['Zensus_1_Wohnung']) / wohnungs_anzahl
            typ2_anteil = float(group.iloc[0]['Zensus_2_Wohnungen']) / wohnungs_anzahl
            typ3_anteil = float(group.iloc[0]['Zensus_3bis6_Wohnungen']) / wohnungs_anzahl
            typ4_anteil = float(group.iloc[0]['Zensus_7bis12_Wohnungen']) / wohnungs_anzahl
            typ5_anteil = float(group.iloc[0]['Zensus_13undmehr_Wohnungen']) / wohnungs_anzahl
            print('typ1_anteil, ', typ1_anteil)
            print('typ2_anteil, ', typ2_anteil)
            print('typ3_anteil, ', typ3_anteil)
            print('typ4_anteil, ', typ4_anteil)
            print('typ5_anteil, ', typ5_anteil)

            # jeden bus zufällig einen Haustypen zuweisen
            # Haustyp dabei je nach Anteil verteilen
            for i in group.index:
                print('i, ', i)
                random_value = np.random.rand()
                print('random_value, ', random_value)
                if random_value < typ1_anteil:
                    buses.at[i, 'Bus_1_Wohnung'] = 1
                    print('typ1_anteil, ', typ1_anteil)
                    print('typ1')
                elif random_value < typ1_anteil + typ2_anteil:
                    buses.at[i, 'Bus_2_Wohnungen'] = 1
                    print('typ1_anteil + typ2_anteil, ', typ1_anteil + typ2_anteil)
                    print('typ2')
                elif random_value < typ1_anteil + typ2_anteil + typ3_anteil:
                    buses.at[i, 'Bus_3bis6_Wohnungen'] = 1
                    print('typ1_anteil + typ2_anteil + typ3_anteil, ', typ1_anteil + typ2_anteil + typ3_anteil)
                    print('typ3')
                elif random_value < typ1_anteil + typ2_anteil + typ3_anteil + typ4_anteil:
                    buses.at[i, 'Bus_7bis12_Wohnungen'] = 1
                    print('typ1_anteil + typ2_anteil + typ3_anteil + typ4_anteil, ', typ1_anteil + typ2_anteil + typ3_anteil + typ4_anteil)
                    print('typ4')
                else:
                    buses.at[i, 'Bus_13undmehr_Wohnungen'] = 1
                    print('typ1_anteil + typ2_anteil + typ3_anteil + typ4_anteil + typ5_anteil, ', typ1_anteil + typ2_anteil + typ3_anteil + typ4_anteil + typ5_anteil)
                    print('typ5')

            print('Fertig mit den Wohnungen')
            # liste mit den index der buse erstellen
            # wenn typ1_ = 1 dann einmal in der liste, type2 = 1, dann zweimal in der liste usw.
            list = []

            wohnungs_anzahl_verteilt = 0
            for bus_index in group.index:
                print('bus_index, ', bus_index)
                if buses.at[bus_index, 'Bus_1_Wohnung'] == 1:
                    print('1 Wohnung')
                    list.extend([bus_index] * 1)
                    buses.at[bus_index, 'Haushalte'] = 1
                    buses.at[bus_index, 'Bewohnerinnen'] = 1
                    wohnungs_anzahl_verteilt += 1
                if buses.at[bus_index, 'Bus_2_Wohnungen'] == 1:
                    print('2 Wohnungen')
                    list.extend([bus_index] * 2)
                    buses.at[bus_index, 'Haushalte'] = 2
                    buses.at[bus_index, 'Bewohnerinnen'] = 2
                    wohnungs_anzahl_verteilt += 2
                
                '''
                Annahme der kleinsten Anzahl an Wohnungen
                '''

                if buses.at[bus_index, 'Bus_3bis6_Wohnungen'] == 1:
                    print('3 bis 6 Wohnungen')
                    list.extend([bus_index] * 3)
                    buses.at[bus_index, 'Haushalte'] = 3
                    buses.at[bus_index, 'Bewohnerinnen'] = 3
                    wohnungs_anzahl_verteilt += 3
                if buses.at[bus_index, 'Bus_7bis12_Wohnungen'] == 1:
                    print('7 bis 12 Wohnungen')
                    list.extend([bus_index] * 7)
                    buses.at[bus_index, 'Haushalte'] = 7
                    buses.at[bus_index, 'Bewohnerinnen'] = 7
                    wohnungs_anzahl_verteilt += 7
                if buses.at[bus_index, 'Bus_13undmehr_Wohnungen'] == 1:
                    print('13 und mehr Wohnungen')
                    list.extend([bus_index] * 13)
                    buses.at[bus_index, 'Haushalte'] = 13
                    buses.at[bus_index, 'Bewohnerinnen'] = 13
                    wohnungs_anzahl_verteilt += 13

            print('List, ', list)
            print('Wohnungen verteilt, Anzahl: ', wohnungs_anzahl_verteilt)
            # Einwohnerzahl zuordnen
            einwohnerzahl = group.iloc[0]['Zensus_Einwohner']

            personen_anzahl = sum([float(group.iloc[0]['Zensus_1_Person']),
                    float(group.iloc[0]['Zensus_2_Personen']),
                    float(group.iloc[0]['Zensus_3_Personen']),
                    float(group.iloc[0]['Zensus_4_Personen']),
                    float(group.iloc[0]['Zensus_5_Personen']),
                    float(group.iloc[0]['Zensus_6_Personen_und_mehr'])])
            print('einwohnerzahl, ', einwohnerzahl)
            print('personen_anzahl, ', personen_anzahl)
            if personen_anzahl == 0:
                '''
                Wenn keine Angaben da sind, werden die Bewohnerinnen gleichmäßig aufgeteilt
                '''

                bew_pro_wohnung = int(group.iloc[0]['Zensus_Einwohner']) / wohnungs_anzahl_verteilt
                for i in group.index:
                    buses.at[i, 'Bewohnerinnen'] = bew_pro_wohnung * buses.at[i, 'Haushalte']
                print('Nächste Gruppe')
                continue


            # anteile der bew pro Haushalttyp
            group = group.replace("–", 0)
            bew_1_anteil = float(group.iloc[0]['Zensus_1_Person'])/personen_anzahl
            bew_2_anteil = float(group.iloc[0]['Zensus_2_Personen'])/personen_anzahl
            bew_3_anteil = float(group.iloc[0]['Zensus_3_Personen'])/personen_anzahl
            bew_4_anteil = float(group.iloc[0]['Zensus_4_Personen'])/personen_anzahl
            bew_5_anteil = float(group.iloc[0]['Zensus_5_Personen'])/personen_anzahl
            bew_6_anteil = float(group.iloc[0]['Zensus_6_Personen_und_mehr'])/personen_anzahl
            print('bew_1_anteil, ', bew_1_anteil)
            print('bew_2_anteil, ', bew_2_anteil)
            print('bew_3_anteil, ', bew_3_anteil)
            print('bew_4_anteil, ', bew_4_anteil)
            print('bew_5_anteil, ', bew_5_anteil)
            print('bew_6_anteil, ', bew_6_anteil)

            # Einwohner solange auf wohnungen ateilig verteilen bis Einwohnerzahl erreicht ist
            # Jeder bus benötigt dabei aber auf jeden Fall mindestens einen Einwohner
            einwohner_verteilt = wohnungs_anzahl_verteilt
            while einwohner_verteilt < einwohnerzahl:
                print('einwohner_verteilt, ', einwohner_verteilt)
                if len(list) == 0:
                    print("Anzahl an Einwohner, die nicht verteilt werden konnten: ", einwohnerzahl - einwohner_verteilt)
                    break
                # zufälligen index aus list auswählen
                random_index = np.random.choice(list)
                # random_index aus list löschen, damit nicht immer der gleiche bus ausgewählt wird
                list.remove(random_index)

                random_value = np.random.rand()
                print('random_value, ', random_value)
                print('Anteile ', bew_1_anteil, bew_2_anteil, bew_3_anteil, bew_4_anteil, bew_5_anteil, bew_6_anteil)
                if random_value < bew_1_anteil:
                    buses.at[random_index, 'Bewohnerinnen'] += 0
                    buses.at[random_index, 'Bus_1_Person'] += 1
                    einwohner_verteilt += 0
                    print('bew_1_anteil, ', bew_1_anteil)

                elif random_value < bew_1_anteil + bew_2_anteil:
                    buses.at[random_index, 'Bewohnerinnen'] += 1
                    buses.at[random_index, 'Bus_2_Personen'] += 1
                    einwohner_verteilt += 1
                    print('bew_1_anteil + bew_2_anteil, ', bew_1_anteil + bew_2_anteil)
                elif random_value < bew_1_anteil + bew_2_anteil + bew_3_anteil:
                    buses.at[random_index, 'Bewohnerinnen'] += 2
                    buses.at[random_index, 'Bus_3_Personen'] += 1
                    einwohner_verteilt += 2
                    print('bew_1_anteil + bew_2_anteil + bew_3_anteil, ', bew_1_anteil + bew_2_anteil + bew_3_anteil)
                elif random_value < bew_1_anteil + bew_2_anteil + bew_3_anteil + bew_4_anteil:
                    buses.at[random_index, 'Bewohnerinnen'] += 3
                    buses.at[random_index, 'Bus_4_Personen'] += 1
                    einwohner_verteilt += 3
                    print('bew_1_anteil + bew_2_anteil + bew_3_anteil + bew_4_anteil, ', bew_1_anteil + bew_2_anteil + bew_3_anteil + bew_4_anteil)
                elif random_value < bew_1_anteil + bew_2_anteil + bew_3_anteil + bew_4_anteil + bew_5_anteil:
                    buses.at[random_index, 'Bewohnerinnen'] += 4
                    buses.at[random_index, 'Bus_5_Personen'] += 1
                    einwohner_verteilt += 4
                    print('bew_1_anteil + bew_2_anteil + bew_3_anteil + bew_4_anteil + bew_5_anteil, ', bew_1_anteil + bew_2_anteil + bew_3_anteil + bew_4_anteil + bew_5_anteil)
                else:
                    '''
                    Auch bei 6 Personen Haushalt nur 5 Personen
                    '''
                    buses.at[random_index, 'Bewohnerinnen'] += 4
                    buses.at[random_index, 'Bus_6_Personen_und_mehr'] += 1
                    einwohner_verteilt += 5
                    print('bew_1_anteil + bew_2_anteil + bew_3_anteil + bew_4_anteil + bew_5_anteil + bew_6_anteil, ', bew_1_anteil + bew_2_anteil + bew_3_anteil + bew_4_anteil + bew_5_anteil + bew_6_anteil)

            # Anpassung der Anzahl der 1 Personenn Wohnung, um die Anzahl der Einwohner zu erreichen
            # Einwohner werden zu Beginn auf die Wohnungen verteilt, sodass jede Wohnung mindestens einen Einwohner hat
            # Dieser wird aber später nicht hinzugefügt, wenn es um die Verteilung der Personen pro Wohnung geht
            for bus in group.index:
                dif = int(buses.at[bus, 'Bewohnerinnen']) - (buses.at[bus, 'Bus_1_Person'] + buses.at[bus, 'Bus_2_Personen']*2 + buses.at[bus, 'Bus_3_Personen']*3 + buses.at[bus, 'Bus_4_Personen']*4 + buses.at[bus, 'Bus_5_Personen']*5 + buses.at[bus, 'Bus_6_Personen_und_mehr']*6)
                print('dif, ', dif)
                if dif > 0:
                    buses.at[bus, 'Bus_1_Person'] += dif
                    print('Anpassung um ', dif, ' auf 1 Person')

            print('Fertig mit den Einwohnern')
            


            print('Nächste Gruppe')
    # Löschen von Spalten, die nicht mehr benötigt werden
    buses.drop(columns=['Bus_1_Wohnung', 'Bus_2_Wohnungen', 'Bus_3bis6_Wohnungen', 'Bus_7bis12_Wohnungen', 'Bus_13undmehr_Wohnungen'], inplace=True)
    return buses



def technik_fill(buses: pd.DataFrame, Technik: List[str], p_total: List[float], pfad: str) -> pd.DataFrame:
    """
    Füllt das Grid-Objekt mit den Techniken basierend auf den gegebenen Anteilen.
    
    Args:
        grid: Das Grid-Objekt, das die Buslinien enthält.
        Technik (list): Liste der Techniken, die zugeordnet werden sollen.
        p_total (list): Liste der Anteile für jede Technik.
        
    Returns:
        grid: Das aktualisierte Grid-Objekt mit den zugeordneten Techniken.
    """

    pv_plz = pd.read_csv(f"{pfad}/input/mastr_values_per_plz.csv", sep=",").set_index("PLZ")
    plz = int(buses['plz_code'].values[0])
    solar_power = pv_plz.loc[plz, 'Mean_Solar_Installed_Capacity_[MW]'] * 1000  # in kW

    for tech, p in zip(Technik, p_total):
        buses = func.technik_sortieren(buses, tech, p, solar_power)

    buses = func.solar_ausrichtung(buses, plz, pv_plz)

    storage_pv = pv_plz.loc[plz, 'Storage_per_PV']
        
    buses = func.storage(buses, storage_pv)

    return buses




def loads_zuordnen(grid: pypsa.Network, buses: pd.DataFrame, bbox: List[float], pfad: str, env=None):
    if env is None:
        environment = func.env_wetter(bbox, pfad)
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
    Alle Loads und StorageUnits erstmal löschen
    """
    grid.loads.drop(grid.loads.index, inplace=True)

    # StorageUnits löschen
    grid.storage_units.drop(grid.storage_units.index, inplace=True)

    # Liste zum Hinzufügen von Loads
    load_cols = {}
    # Hinzufügen von buses
    e_auto_buses = buses.index[buses["Power_E_car"].notna()].tolist()
    e_auto_cols = {}

    # Haushaltsgrößen in einem Dictionary
    household_types = {
        "Bus_1_Person": 1,
        "Bus_2_Personen": 2,
        "Bus_3_Personen": 3,
        "Bus_4_Personen": 4,
        "Bus_5_Personen": 5,
        "Bus_6_Personen_und_mehr": 5,
                        }

    def haus_auto(typ, anz, buses, bus, bew, load_cols, e_auto_cols, e_auto_buses, aufruf, grid, snapshots, environment):
        aufruf += 1
        power, occupants = demand_load.create_haus(people=anz, index=snapshots, env=environment)
        grid.add("Load", name=f"{bus}_load_{aufruf}", bus=bus, carrier="AC")
        load_cols[f"{bus}_load_{aufruf}"] = power
        bew -= anz
        if bus in e_auto_buses:
            e_auto_power = demand_load.create_e_car(occ = occupants, index=snapshots, env=environment)
            grid.add("StorageUnit", name=bus + f"{bus}_E_Auto_{aufruf}", bus=bus, carrier="E_Auto")
            e_auto_cols[f"{bus}_E_Auto_{aufruf}"] = e_auto_power
            if buses.loc[bus, 'Power_E_car'] == buses.loc[bus, 'Factor_E_car']:
                # bus aus Liste entfernen
                e_auto_buses.remove(bus)
            else:
                buses.loc[bus, 'Power_E_car'] -= buses.loc[bus, 'Factor_E_car']
        
        return buses, bew, load_cols, e_auto_cols, e_auto_buses, aufruf

    
    for bus in buses.index:
        print('-----------------------------------')
        print('Neuer Bus')
        print('-----------------------------------')
        print(f"Prüfe, ob Load für {bus} existiert...")
        existing = grid.loads[(grid.loads['bus'] == bus)]
        """
        Doch alle Loads erste entfehrnen, dann alle neu hinzufügen?
        Würde e-auto erleichtern
        """
        if existing.empty:

            bew = int(round(buses.loc[bus, 'Bewohnerinnen']))
            aufruf = 0

            for typ, anz in household_types.items():
                n_haushalte = int(buses.loc[bus, typ])
                if n_haushalte > 0:
                    for i in range(n_haushalte):
                        print(f"{n_haushalte}× {anz}-Personenhaushalt(e) in {bus}")
                        buses, bew, load_cols, e_auto_cols, e_auto_buses, aufruf = haus_auto(typ, anz, buses, bus, bew, load_cols, e_auto_cols, e_auto_buses, aufruf, grid, snapshots, environment)


            while bew > 0:
                anz = min(bew, 5)  # Nimm maximal 5 Personen für den Haushalt
                print(f"Es sind noch {bew} Bewohner  an Bus {bus} übrig, die keinem Haushaltstyp zugeordnet wurden.")
                buses, bew, load_cols, e_auto_cols, e_auto_buses, aufruf = haus_auto(f"Rest_{anz}_Personen", anz, buses, bus, bew, load_cols, e_auto_cols, e_auto_buses, aufruf, grid, snapshots, environment)

        else:
            print(f"Load für {bus} existiert bereits.")

    # Alle neuen Spalten zu p_max_pu hinzufügen
    grid.loads_t.p_set = pd.concat([grid.loads_t.p_set, pd.DataFrame(load_cols)], axis=1)
    grid.storage_units_t.p = pd.concat([grid.storage_units_t.p, pd.DataFrame(e_auto_cols)], axis=1)

    """
    Alle Solargeneratoren erstmal löschen
    """
    grid.generators.drop(grid.generators.index[grid.generators['type'] == 'solar'], inplace=True)

    #solar_buses = buses.index[buses["Power_solar"].notna()]
    solar_buses = buses.index[buses["Power_solar"] != 0]
    solar_cols = {}
    for bus in solar_buses:
        existing = grid.generators[(grid.generators['bus'] == bus)]

        beta = buses.loc[bus, 'HauptausrichtungNeigungswinkel_Anteil']
        gamma = buses.loc[bus, 'Hauptausrichtung_Anteil']


        # Ost-Weste wird in zwei halbe PV aufgeteilt
        if gamma == 'Ost-West':
            gamma_1 = 'Ost'
            gamma_2 = 'West'
            power_1 = demand_load.create_pv(peakpower=buses.loc[bus, 'Power_solar']*0.5, beta=beta, gamma=gamma_1, index=snapshots, env=environment)
            power_2 = demand_load.create_pv(peakpower=buses.loc[bus, 'Power_solar']*0.5, beta=beta, gamma=gamma_2, index=snapshots, env=environment)
            power = power_1 + power_2

        else:
            # power = demand_load.create_pv(peakpower=100*buses.loc[bus, 'Power_solar'], beta=beta, gamma=gamma, index=snapshots, env=environment)
            power = demand_load.create_pv(peakpower=buses.loc[bus, 'Power_solar'], beta=beta, gamma=gamma, index=snapshots, env=environment)
        
        if existing.empty:
            # Generator hinzufügen
            
            grid.add("Generator",
                    name=bus + "_solar",
                    bus=bus,
                    carrier="solar",
                    type="solar",
                    p_nom=buses.loc[bus, 'Power_solar'])
            

            #grid.generators_t.p_max_pu[bus + "_solar"] = power.values
            solar_cols[bus + "_solar"] = power.values

            print('Solar Power:', power.values)

        else:

            solar_cols[bus + "_solar"] = power.values


    if solar_cols:
        grid.generators_t.p_max_pu = pd.concat([grid.generators_t.p_max_pu, pd.DataFrame(solar_cols, index=snapshots)], axis=1)
        grid.generators_t.p_min_pu = pd.concat([grid.generators_t.p_min_pu, pd.DataFrame(solar_cols, index=snapshots)], axis=1)


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
        #grid.generators_t.p_max_pu[bus + "_HP"] = power.values
        hp_cols[bus + "_HP"] = power.values



    if hp_cols:
        grid.generators_t.p_max_pu = pd.concat([grid.generators_t.p_max_pu, pd.DataFrame(hp_cols, index=snapshots)], axis=1)
        grid.generators_t.p_min_pu = pd.concat([grid.generators_t.p_min_pu, pd.DataFrame(hp_cols, index=snapshots)], axis=1)

    print('------------------------------------')
    print('Gewerbe hinzufügen')
    print('------------------------------------')

    # Gewerbe hinzufügen
    if "osm_building" in buses.columns:
        gewerbe_buses = buses.index[buses["osm_building"] != 0]
        gewerbe_cols = {}

        gewerbe_dict = {
            'commercial': 'G0',
            'industrial': 'G0',
            'office': 'G1',
            'kiosk': 'G4',
            'retail': 'G4',
            'supermarket': 'G4',
            'warehouse': 'G0',
            'sports_hall': 'sports',
            'stadium': 'sports'
        }

        '''
        Wie viel Verbrauch?????
        '''

        for bus in gewerbe_buses:
            for gewerbe, typ in gewerbe_dict.items():
                if buses.loc[bus, 'osm_building'] == gewerbe:
                    power = demand_load.create_gewerbe(typ, demand_per_year=1000, index=snapshots, env=environment)
                    grid.add("Load", name=bus + "_Gewerbe", bus=bus, carrier="Gewerbe")
                    gewerbe_cols[bus + "_Gewerbe"] = power.values
            

    # Shops hinzufügen
    if "osm_shop" in buses.columns:
        shops_buses = buses.index[buses["osm_shop"] != 0]
        shops_cols = {}
        for bus in shops_buses:
            if buses.loc[bus, 'osm_shop'] == 'bakery':
                power = demand_load.create_gewerbe('G5', demand_per_year=1000, index=snapshots, env=environment)
            else:
                power = demand_load.create_gewerbe('G4', demand_per_year=1000, index=snapshots, env=environment)
            grid.add("Load", name=bus + "_Shop", bus=bus, carrier="Shop")
            shops_cols[bus + "_Shop"] = power.values
    
    if shops_cols:
        grid.loads_t.p_set = pd.concat([grid.loads_t.p_set, pd.DataFrame(shops_cols, index=snapshots)], axis=1)

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
        bus_mv = trafo['bus0']  
        # name setzen
        gen_name = f"Generator_am_{i}"

        grid.add("Generator",
                name=gen_name,
                bus=bus_mv,
                carrier="gas",
                p_nom=100,
                p_nom_extendable=True,
                capital_cost=500,
                marginal_cost=50,
                efficiency=0.4)

        storage_name = f"Storage_am_{i}"

        grid.add("Generator",
                name=storage_name,
                bus=bus_mv,
                carrier="battery",
                p_min = -1,
                p_max = 0,
                p_nom_extendable=True,
                capital_cost=500,
                marginal_cost=50,
                efficiency_store=0.9,
                efficiency_dispatch=0.9)  
        
    return grid

