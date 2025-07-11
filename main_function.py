
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


def osm_data(net, bbox_neu):

    #%% osm Data abrufen
    Area, Area_features = func.get_osm_data(bbox_neu)
    # Speichern der OSM Daten
    Area_features.to_file("Area_features.geojson", driver="GeoJSON")

    Area_features_df = Area_features.reset_index()

    #%% Daten kombinieren
    net = dc.data_combination(net, Area_features_df)

    return net, Area, Area_features