#%%
import pypsa
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import contextily as ctx
import folium
import geopandas as gpd
from shapely.geometry import Point, LineString, box as create_box
import numpy as np

#%%
def plot_network_on_osm(network, figsize=(15, 12), zoom=None, 
                         bus_color='red', bus_size=50, 
                         line_color='blue', line_width=1.5,
                         transformer_line_color='orange', transformer_line_width=2,
                         transformer_box_color='purple', transformer_box_size=0.0002,
                         title='Network on OpenStreetMap'):
    """
    Plot a PyPSA network on an OpenStreetMap using contextily.
    
    Parameters:
    -----------
    network : pypsa.Network
        The PyPSA network to plot
    figsize : tuple
        Figure size (width, height) in inches
    zoom : int, optional
        Zoom level for the basemap. If None, auto-calculated
    bus_color : str
        Color for bus points
    bus_size : int
        Size of bus markers
    line_color : str
        Color for transmission lines
    line_width : float
        Width of transmission line strokes
    transformer_line_color : str
        Color for transformer lines (when buses are not overlapping)
    transformer_line_width : float
        Width of transformer line strokes
    transformer_box_color : str
        Color for transformer boxes (when buses overlap)
    transformer_box_size : float
        Size of transformer box in degrees (small value for compact squares)
    title : str
        Plot title
    """
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get bus positions
    buses = network.buses[['x', 'y']].copy()
    buses = buses.dropna()
    
    # Create GeoDataFrame for buses
    geometry = [Point(xy) for xy in zip(buses['x'], buses['y'])]
    gdf_buses = gpd.GeoDataFrame(geometry=geometry, crs='EPSG:4326')
    
    # Convert to Web Mercator (EPSG:3857) for contextily
    gdf_buses_mercator = gdf_buses.to_crs('EPSG:3857')
    
    # Get lines
    lines = network.lines[['bus0', 'bus1']].copy()
    
    # Create line geometries
    line_geoms = []
    for idx, row in lines.iterrows():
        bus0 = row['bus0']
        bus1 = row['bus1']
        
        if bus0 in buses.index and bus1 in buses.index:
            coords = [
                (buses.loc[bus0, 'x'], buses.loc[bus0, 'y']),
                (buses.loc[bus1, 'x'], buses.loc[bus1, 'y'])
            ]
            line_geoms.append(LineString(coords))
    
    gdf_lines = gpd.GeoDataFrame(geometry=line_geoms, crs='EPSG:4326')
    gdf_lines_mercator = gdf_lines.to_crs('EPSG:3857')
    
    # Plot lines first (so they appear below buses)
    gdf_lines_mercator.plot(ax=ax, color=line_color, linewidth=line_width, alpha=0.7)
    
    # Get transformers
    transformers = network.transformers[['bus0', 'bus1']].copy()
    
    # Separate transformers into overlapping and non-overlapping
    transformer_line_geoms = []
    transformer_box_geoms = []
    
    for idx, row in transformers.iterrows():
        bus0 = row['bus0']
        bus1 = row['bus1']
        
        if bus0 in buses.index and bus1 in buses.index:
            x0, y0 = buses.loc[bus0, 'x'], buses.loc[bus0, 'y']
            x1, y1 = buses.loc[bus1, 'x'], buses.loc[bus1, 'y']
            
            # Check if buses overlap (same coordinates within tolerance)
            if np.isclose(x0, x1, atol=1e-6) and np.isclose(y0, y1, atol=1e-6):
                # Buses overlap - create a box at the location
                box_geom = create_box(
                    x0 - transformer_box_size/2, y0 - transformer_box_size/2,
                    x0 + transformer_box_size/2, y0 + transformer_box_size/2
                )
                transformer_box_geoms.append(box_geom)
            else:
                # Buses don't overlap - create a line
                coords = [(x0, y0), (x1, y1)]
                transformer_line_geoms.append(LineString(coords))
    
    # Plot transformer lines
    if transformer_line_geoms:
        gdf_transformer_lines = gpd.GeoDataFrame(geometry=transformer_line_geoms, crs='EPSG:4326')
        gdf_transformer_lines_mercator = gdf_transformer_lines.to_crs('EPSG:3857')
        gdf_transformer_lines_mercator.plot(ax=ax, color=transformer_line_color, 
                                            linewidth=transformer_line_width, alpha=0.8)
    
    # Plot transformer boxes
    if transformer_box_geoms:
        gdf_transformer_boxes = gpd.GeoDataFrame(geometry=transformer_box_geoms, crs='EPSG:4326')
        gdf_transformer_boxes_mercator = gdf_transformer_boxes.to_crs('EPSG:3857')
        gdf_transformer_boxes_mercator.plot(ax=ax, facecolor=transformer_box_color, 
                                            edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Plot buses
    gdf_buses_mercator.plot(ax=ax, color=bus_color, markersize=bus_size, alpha=0.8)
    
    # Add OpenStreetMap basemap
    if zoom is None:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    else:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=bus_color, 
               markersize=8, label='Buses', alpha=0.8),
        Line2D([0], [0], color=line_color, linewidth=line_width, 
               label='Transmission Lines', alpha=0.7)
    ]
    
    if transformer_line_geoms:
        legend_elements.append(
            Line2D([0], [0], color=transformer_line_color, linewidth=transformer_line_width,
                   label='Transformers (separate buses)', alpha=0.8)
        )
    
    if transformer_box_geoms:
        legend_elements.append(
            Patch(facecolor=transformer_box_color, edgecolor='black', 
                  label='Transformers (overlapping buses)', alpha=0.8)
        )
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
              framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    return fig, ax


def plot_network_interactive(network,
                             bus_color='red', line_color='blue',
                             transformer_line_color='orange',
                             transformer_box_color='purple',
                             zoom_start=12):
    """
    Create an interactive Folium map of the PyPSA network with hover tooltips.
    
    Parameters:
    -----------
    network : pypsa.Network
        The PyPSA network to plot
    output_file : str
        HTML filename to save the interactive map
    bus_color : str
        Color for bus markers
    line_color : str
        Color for transmission lines
    transformer_line_color : str
        Color for transformer lines
    transformer_box_color : str
        Color for transformer boxes (overlapping buses)
    zoom_start : int
        Initial zoom level for the map
    
    Returns:
    --------
    folium.Map
        The Folium map object
    """
    
    # Get bus positions
    buses = network.buses[['x', 'y']].copy()
    buses = buses.dropna()
    
    # Calculate map center
    center_lat = buses['y'].mean()
    center_lon = buses['x'].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )
    
    # Add transmission lines
    lines = network.lines[['bus0', 'bus1']].copy()
    for idx, row in lines.iterrows():
        bus0 = row['bus0']
        bus1 = row['bus1']
        
        if bus0 in buses.index and bus1 in buses.index:
            coords = [
                [buses.loc[bus0, 'y'], buses.loc[bus0, 'x']],
                [buses.loc[bus1, 'y'], buses.loc[bus1, 'x']]
            ]
            
            # Get line info
            line_info = network.lines.loc[idx]
            tooltip_text = f"<b>Line:</b> {idx}<br>"
            tooltip_text += f"<b>From:</b> {bus0}<br>"
            tooltip_text += f"<b>To:</b> {bus1}<br>"
            if 'length' in line_info.index:
                tooltip_text += f"<b>Length:</b> {line_info['length']:.2f} km<br>"
            if 's_nom' in line_info.index:
                tooltip_text += f"<b>Capacity:</b> {line_info['s_nom']:.2f} MVA<br>"
            
            folium.PolyLine(
                coords,
                color=line_color,
                weight=2,
                opacity=0.7,
                tooltip=folium.Tooltip(tooltip_text)
            ).add_to(m)
    
    # Add transformers
    transformers = network.transformers[['bus0', 'bus1']].copy()
    for idx, row in transformers.iterrows():
        bus0 = row['bus0']
        bus1 = row['bus1']
        
        if bus0 in buses.index and bus1 in buses.index:
            x0, y0 = buses.loc[bus0, 'x'], buses.loc[bus0, 'y']
            x1, y1 = buses.loc[bus1, 'x'], buses.loc[bus1, 'y']
            
            # Get transformer info
            trafo_info = network.transformers.loc[idx]
            tooltip_text = f"<b>Transformer:</b> {idx}<br>"
            tooltip_text += f"<b>From:</b> {bus0}<br>"
            tooltip_text += f"<b>To:</b> {bus1}<br>"
            if 's_nom' in trafo_info.index:
                tooltip_text += f"<b>Capacity:</b> {trafo_info['s_nom']:.2f} MVA<br>"
            if 'type_info' in trafo_info.index:
                tooltip_text += f"<b>Type:</b> {trafo_info['type_info']}<br>"
            
            # Check if buses overlap
            if np.isclose(x0, x1, atol=1e-6) and np.isclose(y0, y1, atol=1e-6):
                # Overlapping buses - draw a square marker
                folium.RegularPolygonMarker(
                    location=[y0, x0],
                    fill=True,
                    fill_color=transformer_box_color,
                    fill_opacity=0.7,
                    color='black',
                    number_of_sides=4,
                    radius=8,
                    rotation=45,
                    tooltip=folium.Tooltip(tooltip_text)
                ).add_to(m)
            else:
                # Non-overlapping buses - draw a line
                coords = [[y0, x0], [y1, x1]]
                folium.PolyLine(
                    coords,
                    color=transformer_line_color,
                    weight=3,
                    opacity=0.8,
                    tooltip=folium.Tooltip(tooltip_text)
                ).add_to(m)
    
    # Add buses
    for bus_id, bus_data in buses.iterrows():
        bus_info = network.buses.loc[bus_id]
        tooltip_text = f"<b>Bus:</b> {bus_id}<br>"
        if 'v_nom' in bus_info.index:
            tooltip_text += f"<b>Voltage:</b> {bus_info['v_nom']:.2f} kV<br>"
        if 'carrier' in bus_info.index:
            tooltip_text += f"<b>Carrier:</b> {bus_info['carrier']}<br>"
        
        folium.CircleMarker(
            location=[bus_data['y'], bus_data['x']],
            radius=5,
            color='black',
            fill=True,
            fill_color=bus_color,
            fill_opacity=0.8,
            weight=1,
            tooltip=folium.Tooltip(tooltip_text)
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m