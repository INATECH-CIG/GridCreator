import pypsa
import pandas as pd

# %% Netzwerk
n = pypsa.Network()
snapshots = pd.date_range("2025-10-17", periods=4, freq="H")
n.set_snapshots(snapshots)

# Buses
n.add("Bus", "Bus1", carrier="AC")
n.add("Bus", "Bus2", carrier="AC")

# Line
n.add("Line", "Line1", bus0="Bus1", bus1="Bus2", x=0.01, r=0.001, s_nom=1000, carrier="AC")

# Generator an Bus1 (liefert nicht immer)
n.add("Generator",
      "Gen1",
      bus="Bus1",
      p_nom=10000,
      marginal_cost=50,
      p_max_pu=pd.Series([1, 1, 1, 1], index=snapshots))

# Load an Bus2
n.add("Load", "Load1", bus="Bus2", p_set=80)

# Storage an Bus2
n.add("StorageUnit",
      "Storage1",
      bus="Bus2",
      p_nom=500,
      max_hours=3,
      efficiency_store=0.9,
      efficiency_dispatch=0.9)

# Storage flexibel
n.storage_units.at["Storage1", "cyclic_state_of_charge"] = False

n.storage_units_t.spill.loc[:, "Storage1"] = [0, 0.2, 0.2, 0]

n.storage_units_t.state_of_charge_set.loc[:, "Storage1"] = 1

#%% Optimierung
n.optimize(solver_name="gurobi")

# %% Ergebnisse
print(n.generators_t.p)
print(n.storage_units_t.p_dispatch)
print(n.storage_units_t.state_of_charge)

# %%
