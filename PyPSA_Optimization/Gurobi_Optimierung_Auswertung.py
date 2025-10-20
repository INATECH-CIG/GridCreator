#%%
import pypsa
import pandas as pd
from pathlib import Path

#%%

# Basis: das Verzeichnis zu GridCreator
this_dir = Path(__file__).parent

# Pfad zur .nc-Datei
file = this_dir / "output" / "grid_Schallstadt_GER_optimize_ecar.nc"

#%% Pypsa netzwerk einlesen aus .nc datei
network = pypsa.Network(file)

def analyze_infeasibility(network_path):
	print(f"Loading network from {network_path}...")
	n = pypsa.Network(network_path)
	print("Network loaded.")

	# Check for negative or NaN values in key components
	print("\nChecking for negative or NaN values in key components:")
	for comp in ['generators', 'loads', 'lines', 'links', 'storage_units']:
		df = getattr(n, comp, None)
		if df is not None and not df.empty:
			for col in df.columns:
				if pd.api.types.is_numeric_dtype(df[col]):
					nans = df[col].isna().sum()
					negatives = (df[col] < 0).sum()
					if nans > 0 or negatives > 0:
						print(f"{comp}.{col}: {nans} NaNs, {negatives} negatives")

	# Print the objective value if available
	if hasattr(n, 'objective'):
		print(f"\nObjective value: {getattr(n, 'objective', 'N/A')}")


if __name__ == "__main__":
	analyze_infeasibility(file)
	n = pypsa.Network(file)
	# n.generators_t.p_max_pu["BranchTee_mvgd_36165_lvgd_1884820002_building_28969804_solar"] = 1
	n.optimize(snapshots=n.snapshots[:], solver_name="gurobi")


#%%
# Pypsa netzwerk einlesen aus .nc datei
network = pypsa.Network(file)
#%%
# Carrier in StorageUnits und Generatoren
used_carriers = set(network.generators['carrier'].unique()) | set(network.storage_units['carrier'].unique())

# Definierte Carrier
defined_carriers = set(network.carriers.index)

# Nicht definierte Carrier
undefined_carriers = used_carriers - defined_carriers
print("Nicht definierte Carrier:", undefined_carriers)

#%%
import pypsa
import pandas as pd

def check_network_issues(network_path):
    n = pypsa.Network(network_path)
    print(f"Netzwerk geladen: {network_path}\n")

    problem_found = False

    # 1️⃣ Überprüfe numerische Spalten auf NaNs oder negative Werte
    for comp_name in ['generators', 'loads', 'lines', 'links', 'storage_units', 'transformers']:
        df = getattr(n, comp_name, None)
        if df is not None and not df.empty:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    nans = df[col].isna().sum()
                    negs = (df[col] < 0).sum()
                    if nans > 0 or negs > 0:
                        problem_found = True
                        print(f"[{comp_name}] Spalte '{col}': {nans} NaNs, {negs} negative Werte")

    # 2️⃣ Prüfe Generator- und Speicher-Carriers
    undefined_carriers = set()
    for comp_name in ['generators', 'storage_units']:
        df = getattr(n, comp_name, None)
        if df is not None and not df.empty:
            carriers = df['carrier'].unique()
            for c in carriers:
                if c not in n.carriers.index:
                    undefined_carriers.add(c)
    if undefined_carriers:
        problem_found = True
        print(f"Nicht definierte Carrier: {undefined_carriers}")
    else:
        print("Alle Carrier sind definiert.")

    # # 3️⃣ Prüfe Zeitspezifische Daten (generators_t, storage_units_t, loads_t)
    # for tname in ['generators_t', 'storage_units_t', 'loads_t']:
    #     if hasattr(n, tname):
    #         df_t = getattr(n, tname)
    #         if df_t is not None and not df_t.empty:
    #             nan_count = df_t.isna().sum().sum()
    #             if nan_count > 0:
    #                 problem_found = True
    #                 print(f"{tname} enthält {nan_count} NaN-Werte")

    if not problem_found:
        print("Keine offensichtlichen Probleme gefunden. Netzwerk sollte optimierbar sein.")

    return n

# Beispielaufruf
network_path = file
n = check_network_issues(network_path)

#%%
# Netzwerk speichern
network.export_to_netcdf(file)

#%%
network.loads.p_set = network.loads.p_set.replace(0, 0.1) 
#%%
# Gas generator an jeden bus setzen
for bus in network.buses.index:
	gen_name = f"GasGen_{bus}"
	network.add("Generator",
				name=gen_name,
				bus=bus,
				carrier="gas",
				p_nom=10,
				p_min_pu=0,
				p_max_pu=1,
				marginal_cost=50)  # Beispiel-Kosten
      
#%%
# Alle Leitungen auf erweiterbar setzen
for line in network.lines.index:
	network.lines.at[line, 's_nom_extendable'] = True

#%%
# an alle buses eine storage unit setzen
for bus in network.buses.index:
	su_name = f"Storage_{bus}"
	network.add("StorageUnit",
				name=su_name,
				bus=bus,
				capacity=10,
				max_hours=5,
				efficiency_store=0.9,
				efficiency_dispatch=0.9,
				self_discharge=0.01,
				marginal_cost=5,
                p_nom_extendable=True) 
      
#%%
# alle Transformer s_nom_extendable setzen
for trafo in network.transformers.index:
	network.transformers.at[trafo, 's_nom_extendable'] = True
      

#%%

network.generators.loc['Storage_am_Transformer_lv_grid_3165700001_1', 'sign'] = -1
network.generators.loc['Storage_am_Transformer_lv_grid_3165700006_1', 'sign'] = -1



#%%
# 
network.optimize(solver_name='gurobi', solver_options={'ResultFile':'model_all.ilp'}, snapshots=network.snapshots)

#%%

# Prüfe auf NaNs oder negative Werte
print(network.generators[['p_min_pu', 'p_max_pu', 'efficiency', 'marginal_cost']].describe())

# Prüfe ob alle Generatoren an Busse angeschlossen sind
print(network.generators.bus.isna().sum())

# Prüfe ob es Loads gibt
print(network.loads.p_set.describe())

# Prüfe auf Verbindungsprobleme
network.consistency_check()
# %%
