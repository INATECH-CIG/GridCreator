'''
basic plotting
'''
import matplotlib.pyplot as plt
import pypsa

def plot_gird(grid):
    fig, ax = plt.subplots(figsize=(10, 10))
    grid.lines.plot(ax=ax, color='gray', linewidth=0.5, label='Lines')
    grid.buses.plot(ax=ax, color='blue', markersize=20, label='Buses')
    grid.generators.plot(ax=ax, color='green', markersize=15, label='Generators')
    plt.legend()
    plt.title('Grid Overview')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()