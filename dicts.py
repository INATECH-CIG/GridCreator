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

generator_agg_dict = {'bus': 'first',
            'control' : 'last', # as first might be Slack
            'type' : 'first',
            'p_nom' : 'sum',
            'p_nom_mod' : 'sum',
            'p_nom_extendable' : 'first',
            'p_nom_min' : 'sum', # only for capacity expansion
            'p_nom_max' : 'sum',
            'p_nom_set' : 'sum',
            'p_min_pu' : 'mean', # small error: normally we should use weighted mean here
            'p_max_pu' : 'mean', # same here
            'p_set' : 'sum',
            'e_sum_min' : 'sum',
            'e_sum_max' : 'sum',
            'q_set' : 'sum', # ?
            'sign' : 'first',
            'carrier': 'first',
            'marginal_cost' : 'mean',
            'marginal_cost_quadratic' : 'sum',
            'active' : 'first',
            'build_year' : 'mean',
            'lifetime' : 'mean',
            'capital_cost' : 'mean',
            'efficiency': 'mean',
            'committable' : 'first',
            'start_up_cost' : 'mean',
            'shut_down_cost' : 'mean',
            'stand_by_cost' : 'mean',
            'min_up_time' : 'mean',
            'min_down_time' : 'mean',
            'up_time_before' : 'mean',
            'down_time_before' : 'mean',
            'ramp_limit_up' : 'mean',
            'ramp_limit_down' : 'mean',
            'ramp_limit_start_up' : 'mean',
            'ramp_limit_shut_down' : 'mean',
            'weight' : 'mean',
            'p_nom_opt': 'sum',
            'weather_cell_id': 'first',
            'subtype': 'first',
            'source_id': 'first',
            'voltage_level': 'first',
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


