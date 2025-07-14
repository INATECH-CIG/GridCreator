
import functions as func
import ding0_grid_generators as ding0
import data_combination as dc


def ding0_grid(bbox, grids_dir, output_file_grid):
    
    # Netz extrahieren
    grid = ding0.load_grid(bbox, grids_dir)

    # neue bbox für alle enthaltenen buses laden
    bbox_neu = func.compute_bbox_from_buses(grid)

    # Grid creation für erweiterte bbox
    grid = ding0.load_grid(bbox_neu, grids_dir)

    grid.export_to_netcdf(output_file_grid)

    return grid, bbox_neu


def osm_data(net, bbox_neu, buffer):
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

    # Bundesland
    net.buses = func.Bundesland(net.buses, bundesland_data)

    # Zensus ID
    net.buses["Zensus_ID"] = func.gitter_ID(net.buses, zensus_data)

    """
    Mittels der Gitter_ID sollten nun auch alle weiteren Daten zugeordnet werden können.
    """

    # Zensus: Einwohnerdaten
    spalte = "Einwohner"
    net.buses = func.zenus_daten(net.buses, zensus_data, spalte)

    return net