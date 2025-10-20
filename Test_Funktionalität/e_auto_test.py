import pypsa
import pandas as pd

# %% Netzwerk
n = pypsa.Network()
snapshots = pd.date_range("2025-10-17", periods=10, freq="H")
n.set_snapshots(snapshots)

# Buses
n.add("Bus", "Node", carrier="AC")
n.add("Bus", "Battery", carrier="AC")

# Link
occupancy = pd.Series([1, 1, 1, 0, 0, 0, 1, 0, 0, 1], index=snapshots)
n.add("Link", "Charger", bus0="Node", bus1="Battery", p_nom = 120,
      p_nom_extendable=True, efficiency=0.95, p_max_pu = occupancy)

# Generator an Node (liefert nicht immer)
n.add("Generator",
      "Gen1",
      bus="Node",
      p_nom_extendable=True,
      marginal_cost=50,
      p_max_pu=pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], index=snapshots))


# Store an Battery

n.add("Store", "Store1",
      bus="Battery",
      e_nom=100,
      #p_nom_extendable=True,
      e_nom_extendable=True,
      p_initial=0,
      efficiency_store=0.9,
      efficiency_dispatch=0.9,
      self_discharge=0.01)

# Load an Battery
n.add("Load", "Load1",
      bus="Battery",
      p_set=pd.Series([0, 0, 0, 10, 10, 10, 0, 10, 10, 0], index=snapshots))

#%% Optimierung
n.optimize(solver_name="gurobi")

# %% Ergebnisse
print(n.generators_t.p)
print(n.store)
print(n.storage_units_t.state_of_charge)

# %%

n.add("Bus", f"{bus}E-Car", carrier="Battery")
n.add("Link", f"{bus}E-Car_Connector_charge", bus0=bus, bus1=f"{bus}E-Car", p_nom=charging_power, efficiency=0.95)
n.add("Link", f"{bus}E-Car_Connector_discharge", bus0=f"{bus}E-Car", bus1=bus, p_nom=charging_power, efficiency=0.95)

n.add("Load", f"{bus}E-Car_Load", bus=f"{bus}E-Car",
      p_set=spill, index=snapshots)

n.add("Store", f"{bus}E-Car_Storage",
      bus=f"{bus}E-Car",
      e_nom=77, # kWh https://www.enbw.com/blog/elektromobilitaet/laden/wie-gross-muss-die-batterie-fuer-mein-elektroauto-sein/
      p_initial=0,
      efficiency_store=0.9,
      efficiency_dispatch=0.9,
      self_discharge=0.01)

