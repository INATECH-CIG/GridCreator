# Liste aller dictionaries
import pandas as pd
import numpy as np



"""
Bevölkerungsdaten für Bundesländer laden
Technische Daten auf Bundeslandebene finden

Falls nicht Bundeslandebene möglich, dann Deutschland als Ganzes
Dann müssen allerdingsFaktoren angepasst werden!
"""


"""
Daten stammen zum Großteil aus Zensus
Daher alles für beliebige Referenzfläche abrufbar

Hier darf mehr stehen
"""
# Bev_data_Zensus = pd.DataFrame({
#     'GEN': ["Bayern", "Schleswig-Holstein", "NRW", "Deutschland"],
#     'Zensus_ID': [0.2, 0.3, 0.4, 0.5],
#     'Zensus_x_mp_100m': [0.7, 0.6, 0.5, 0.4],
#     'Zensus_y_mp_100m': [0.1, 0.1, 0.1, 0.1],
#     'Zensus_Einwohner': [100, 100, 100, 100],
#     'Zensus_Insgesamt_Heizungsart': [0.4, 0.5, 0.6, 0.7],
#     'Zensus_Fernheizung': [0.5, 324117, 0.3, 0.2],
#     'Zensus_Etagenheizung': [0.1, 77272, 0.1, 0.1],
#     'Zensus_Blockheizung': [0.3, 27386, 0.1, 0.2],
#     'Zensus_Zentralheizung': [0.5, 1075695, 0.3, 0.2],
#     'Zensus_Einzel_Mehrraumoefen': [0.7, 29578, 0.5, 0.4],
#     'Zensus_keine_Heizung': [0.9, 6327, 0.7, 0.6],
#     'Zensus_durchschnMieteQM': [0.5, 0.6, 0.7, 0.8],
#     'Zensus_Eigentuemerquote': [0.3, 0.4, 0.5, 0.6]
# })

solar_dict={
    'Süd': 0,
    'Süd-Ost': -45,
    'Süd-West': 45,
    'Ost': -90,
    'West': 90,
    'Nord-Ost': -135,
    'Nord-West': 135,
    'Nord': 180
}


agg_dict={
    'Zensus_Fernheizung': 'sum',
    'Zensus_Etagenheizung': 'sum',
    'Zensus_Blockheizung': 'sum',
    'Zensus_Zentralheizung': 'sum',
    'Zensus_Einzel_Mehrraumoefen': 'sum',
    'Zensus_keine_Heizung': 'sum',

    'Zensus_durchschnMieteQM': 'mean',
    'Zensus_Eigentuemerquote': 'mean'
    }

# """
# Hier darf mehr stehen
# """
# Bev_data_Technik = pd.DataFrame({
#     'GEN': ["Baden-Württemberg", "Bayern", "Berlin", "Brandenburg", "Bremen", "Hamburg", "Hessen", "Mecklenburg-Vorpommern", "Niedersachsen", "Nordrhein-Westfalen", "Rheinland-Pfalz", "Saarland", "Sachsen", "Sachsen-Anhalt", "Schleswig-Holstein", "Thüringen"],
#     'solar': [1011.8, 1817.7, 42.1, 401.8, 26, 28.4, 299.4, 264.7, 730.2, 918.2, 283.2, 50.5, 469.7, 24.7, 301.8, 135.1], # in MW: https://www.solarbranche.de/ausbau/deutschland/photovoltaik
#     'HP_ambient': [137000, 156000, 5100, 18000, 500, 2500, 10000, 7800, 31000, 87000, 41000, 4600, 23000, 13000, 12000, 17000], # Anzahl: https://www.waermepumpe.de/fileadmin/user_upload/Mediengalerie/Zahlen_und_Daten/Absatzzahlen_Marktanteile/619_WPinDeutschland_2018.PNG
#     'HP_geothermal': [29000, 49000, 3500, 21000, 450, 2300, 41000, 6800, 21000, 70000, 16000, 1100, 24000, 8000, 13000, 6500], # Anzahl: https://www.waermepumpe.de/fileadmin/user_upload/Mediengalerie/Zahlen_und_Daten/Absatzzahlen_Marktanteile/619_WPinDeutschland_2018.PNG
#     'E_car': [268123, 317954, 41988, 33278, 8337, 35572, 145946, 14620, 176774, 366309, 83379, 17071, 38571, 19343, 60539, 21921] # Anzahl: https://de.statista.com/statistik/daten/studie/75841/umfrage/bestand-an-personenkraftwagen-mit-elektroantrieb/
# })


"""
Allgemeingültig, unabhängig von Referenzfläche

Hier darf mehr stehen
"""
faktoren_technik = {
    'solar':{
        'Heizung': {
            'Zensus_Fernheizung_sum': 0.2,
            'Zensus_Etagenheizung_sum': 0.7,
            'Zensus_Blockheizung_sum': 0.1,
            'Zensus_Zentralheizung_sum': 0.5,
            'Zensus_Einzel_Mehrraumoefen_sum': 0.3,
            'Zensus_keine_Heizung_sum': 0.1
        },
        'Miete': {
            'Zensus_durchschnMieteQM_mw': 0.6,
            'Zensus_Eigentuemerquote_mw': 0.4
        }
    },
    'HP_ambient':{
        'Heizung': {
            'Zensus_Fernheizung_sum': 0.1,
            'Zensus_Etagenheizung_sum': 0.5,
            'Zensus_Blockheizung_sum': 0.2,
            'Zensus_Zentralheizung_sum': 0.4,
            'Zensus_Einzel_Mehrraumoefen_sum': 0.2,
            'Zensus_keine_Heizung_sum': 0.4
        },
        'Miete': {
            'Zensus_durchschnMieteQM_mw': 0.3,
            'Zensus_Eigentuemerquote_mw': 0.2
        }
    }, 
    'HP_geothermal':{
        'Heizung': {
            'Zensus_Fernheizung_sum': 0.3,
            'Zensus_Etagenheizung_sum': 0.2,
            'Zensus_Blockheizung_sum': 0.4,
            'Zensus_Zentralheizung_sum': 0.6,
            'Zensus_Einzel_Mehrraumoefen_sum': 0.1,
            'Zensus_keine_Heizung_sum': 0.3
        },
        'Miete': {
            'Zensus_durchschnMieteQM_mw': 0.5,
            'Zensus_Eigentuemerquote_mw': 0.3
        }
    },
    'E_car':{
        'Heizung': {
            'Zensus_Fernheizung_sum': 0.4,
            'Zensus_Etagenheizung_sum': 0.3,
            'Zensus_Blockheizung_sum': 0.2,
            'Zensus_Zentralheizung_sum': 0.5,
            'Zensus_Einzel_Mehrraumoefen_sum': 0.4,
            'Zensus_keine_Heizung_sum': 0.2
        },
        'Miete': {
            'Zensus_durchschnMieteQM_mw': 0.7,
            'Zensus_Eigentuemerquote_mw': 0.5
        }
    }
}

"""
Kategorien und deren Eigenschaften
Allgemeine Definition, unabhängig von allem

Alles was hier steht, muss überall vorkommen
"""
kategorien_eigenschaften = pd.DataFrame({
    'Heizung': ['Zensus_Fernheizung', 'Zensus_Etagenheizung', 'Zensus_Blockheizung', 'Zensus_Zentralheizung', 'Zensus_Einzel_Mehrraumoefen', 'Zensus_keine_Heizung'],
    'Miete': ['Zensus_durchschnMieteQM', 'Zensus_Eigentuemerquote', None, None, None, None]
})




