'''
Module containing dictionaries for solar orientation and aggregation methods.
'''

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
    'Zensus_Holz_Holzpellets': 'sum',
    'Zensus_Kohle': 'sum',
    'Zensus_Geb_Fernwaerme': 'sum',
    'Zensus_Gas': 'sum',
    'Zensus_Geb_Holz_Holzpellets': 'sum',
    'Zensus_Geb_Kohle': 'sum',
    'Zensus_durchschnMieteQM': 'mean',
    'Zensus_30bis39': 'sum',
    'Zensus_40bis49': 'sum',
    'Zensus_Geb_FreiEFH': 'sum',
    'Zensus_Freist_ZFH': 'sum',
    'Zensus_EU27_Land': 'sum',
    'Zensus_1_Person': 'sum',
    'Zensus_2_Personen': 'sum',
    'Zensus_3_Personen': 'sum',
    'Zensus_4_Personen': 'sum',
    'Zensus_5_Personen': 'sum',
    'Zensus_6_Personen_und_mehr': 'sum',
    'Zensus_Ehep_Kinder_ab18': 'sum',
    'Zensus_NichtehelLG_mind_1Kind_unter18': 'sum',
    'Zensus_Vor1919': 'sum',
    'Zensus_a1970bis1979': 'sum',
    'Zensus_1_Wohnung': 'sum',
    'Zensus_2_Wohnungen': 'sum',
    'Zensus_3bis6_Wohnungen': 'sum',
    'Zensus_7bis12_Wohnungen': 'sum',
    'Zensus_13undmehr_Wohnungen': 'sum',
    }


