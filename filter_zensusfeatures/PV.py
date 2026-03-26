#%%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.optimize import least_squares

from dicts import agg_dict_all


#%%
'''
This is the scipt with which the census data was filtered and the factors for the downscaling of the photovoltaic data were calculated. The factors were calculated by minimizing the squared differences between the actual photovoltaic numbers per PLZ and the estimated photovoltaic numbers per PLZ, which were calculated by multiplying the factors with the respective census features and normalizing them by the sum of the respective census features per federal state.

It can be recalculated by copying the file into the main folder and running it. The factors will be stored in the "input/Faktoren_opt.csv" file.
'''
#%% Load data
zensus_data_pv = pd.read_csv("input/zensus_sum/data_pv.csv", sep=",")
mastr_values_per_plz = pd.read_csv("input/mastr_values_per_plz.csv")
Bev_data_Zensus_land_all = pd.read_csv("input/Bev_data_Zensus_land_all.csv")
georef_germany_postleitzahl = gpd.read_file("input/georef-germany-postleitzahl.geojson")

#%%
'''
Determine PV correlation
'''
#%%
# Data processing
pca_pv_zensus = (zensus_data_pv.rename(columns={"Unnamed: 0": "PLZ"}).assign(PLZ=lambda df: df["PLZ"].astype(int).astype(str).str.zfill(5)).set_index("PLZ")).copy()
pv_plz = (mastr_values_per_plz.assign(PLZ=lambda df: df['PLZ'].astype(int).astype(str).str.zfill(5))[['PLZ', 'Anzahl_Solar']].set_index('PLZ')).copy()

# Normalizing Mastr data
pv_plz_norm = pv_plz.div(pca_pv_zensus["Zensus_Einwohner"], axis=0)
# Merge both Dfs
pca_pv_zensus_merged = pd.merge(pv_plz_norm, pca_pv_zensus, left_index=True, right_index=True, how="inner")

# Columns that were summed, normalize by population
# Filter columns
sum_cols = [col for col, agg in agg_dict_all.items() if agg == "sum"]
# Normalizing
pca_pv_zensus_merged_norm = pca_pv_zensus_merged.copy()
pca_pv_zensus_merged_norm[sum_cols] = pca_pv_zensus_merged[sum_cols].div(pca_pv_zensus_merged["Zensus_Einwohner"], axis=0)

# Calculate correlations
correlations_pv_norm = pca_pv_zensus_merged_norm.corr()

#%%
# Sort correlations
correlations_pv_norm_sorted = correlations_pv_norm.sort_values(by="Anzahl_Solar", ascending=False)

# Sort by absolute values
corr_pv_sort_abs = correlations_pv_norm.reindex(correlations_pv_norm['Anzahl_Solar'].abs().sort_values(ascending=False).index)

# Select the top 20 percent
corr_pv_sort_abs.drop(index=['Anzahl_Solar'], inplace=True)
corr_pv_sort_abs_20 = corr_pv_sort_abs.head(int(0.2*len(corr_pv_sort_abs)))

# %%
'''
Calculation of Variance Inflation Factor
'''
# Relevant variables for VIF
vif_data_pv = pca_pv_zensus_merged_norm[corr_pv_sort_abs_20.index]

# Iterative removal of features with high VIF
while True:
    vif_pv = pd.DataFrame()
    vif_pv["VIF"] = [variance_inflation_factor(vif_data_pv.values, i) for i in range(vif_data_pv.shape[1])]
    vif_pv["Feature"] = vif_data_pv.columns
    vif_pv = vif_pv.sort_values("VIF", ascending=True)

    if vif_pv["VIF"].iloc[-1] < 10:
        break

    # Drop feature with highest VIF
    vif_data_pv = vif_data_pv.drop(columns=[vif_pv["Feature"].iloc[-1]])

# Save correlation coefficients of filtered features in a separate DF
corr_filtered_pv = correlations_pv_norm[vif_pv['Feature']].loc['Anzahl_Solar']

# %%
'''
Minimization with scipy.optimize.least_squares
'''

# Prepare data for optimization
Bev_data_Zensus_pv = Bev_data_Zensus_land_all.copy()
gpd_bundesland_pv = georef_germany_postleitzahl.copy()
data_zensus_plz_pv = zensus_data_pv.copy()

data_zensus_plz_pv = (data_zensus_plz_pv.rename(columns={"Unnamed: 0": "plz_code"}).reset_index(drop=True).assign(plz_code=lambda df: df["plz_code"].astype(str).str.zfill(5))).copy()
gpd_bundesland_pv['plz_code'] = gpd_bundesland_pv['plz_code'].astype(str).str.zfill(5)

# Merge on PLZ
zensus_buses_pv = pd.merge(data_zensus_plz_pv, gpd_bundesland_pv[['plz_code', 'lan_name']], on='plz_code', how='left')
zensus_buses_pv = pd.merge(zensus_buses_pv, pv_plz, left_on='plz_code', right_index=True, how='left')

# PV per Bundesland
pv_bundesland = zensus_buses_pv.groupby('lan_name')['Anzahl_Solar'].sum()
Bev_data_Zensus_pv = Bev_data_Zensus_pv.set_index('GEN')
pv_bundesland = pv_bundesland.reindex(Bev_data_Zensus_pv.index)

# Filter correct columns
columns_to_keep_pv = list(corr_filtered_pv.index) 
Bev_data_Zensus_pv = Bev_data_Zensus_pv.loc[:, columns_to_keep_pv]
columns_to_keep_pv.append('lan_name')
columns_to_keep_pv.append('Anzahl_Solar')
zensus_buses_pv = zensus_buses_pv.loc[:, columns_to_keep_pv]

# Initialize dataframe to store factors
F_pv = zensus_buses_pv[['Anzahl_Solar', 'lan_name']].copy()
zensus_buses_pv.drop(columns=['Anzahl_Solar'], inplace=True)
F_pv['factor'] = None

# Calculate factors: (Anzahl_Solar in plz / Anzahl_Solar in respective bundesland)
for idx, row in F_pv.iterrows():
    F_pv.at[idx, 'factor'] = (F_pv.at[idx, 'Anzahl_Solar'] / pv_bundesland.loc[row['lan_name']])


# Function for least_squares
def function_vectorized_ls(f, zensus_bus, Bev_data_Zensus, res):

    # Numerator per PLZ
    numerator = zensus_bus.drop(columns=['lan_name']).values @ f  

    # Denominator per Bundesland
    denominator_map = dict(zip(Bev_data_Zensus.index, Bev_data_Zensus.values @ f))

    # Assign correct denominator per PLZ
    denominator = np.array([denominator_map[name] for name in zensus_bus['lan_name']])

    # Calculate residuals
    x = numerator / denominator - res.to_numpy(dtype=float)

    return np.asarray(x, dtype=float).ravel()


# Prepare data for least_squares
zensus_buses_short_pv = zensus_buses_pv.copy()
faktoren_short_pv = F_pv['factor']
# Check for NaNs
nan_indices_pv = faktoren_short_pv[faktoren_short_pv.isna()].index
# Remove corresponding rows everywhere
if len(nan_indices_pv) > 0:
    faktoren_short_pv = faktoren_short_pv.drop(index=nan_indices_pv).reset_index(drop=True)
    zensus_buses_short_pv = zensus_buses_short_pv.drop(index=nan_indices_pv).reset_index(drop=True)

# Initial values for factors
num_f_pv = zensus_buses_short_pv.drop(columns=['lan_name']).shape[1]
f0_pv = np.ones(num_f_pv)

# Perform minimization
results_pv = least_squares(function_vectorized_ls, f0_pv, args=(zensus_buses_short_pv, Bev_data_Zensus_pv, faktoren_short_pv))


# %%
# Store results in CSV
columns = zensus_buses_short_pv.drop(columns=['lan_name']).columns
results = results_pv.x
# Load existing CSV
csv = pd.read_csv("input/Faktoren_opt.csv")
csv.set_index("Technik", inplace=True)

# Add columns with respective values from results
for col, res in zip(columns, results):
    csv.at['solar', col] = res

# Fill missing values with 0
csv.fillna(0, inplace=True)
csv.to_csv("input/Faktoren_opt.csv")
# %%
