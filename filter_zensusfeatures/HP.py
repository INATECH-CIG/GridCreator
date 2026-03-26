#%%
import pandas as pd
import geopandas as gpd
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np

import sklearn as sk
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.optimize import least_squares

from dicts import agg_dict_all


#%%
'''
This is the scipt with which the census data was filtered and the factors for the downscaling of the heat pump data were calculated. The factors were calculated by minimizing the squared differences between the actual heat pump numbers per PLZ and the estimated heat pump numbers per PLZ, which were calculated by multiplying the factors with the respective census features and normalizing them by the sum of the respective census features of Germany.

It can be recalculated by copying the file into the main folder and running it. The factors will be stored in the "input/Faktoren_opt.csv" file.
'''

#%% Load data
pca_hp_zensus = pd.read_csv("input/zensus_sum/data_hp.csv", sep=",")
pca_hp_zensus.rename(columns={"Unnamed: 0": "GEN"}, inplace=True)
pca_hp_zensus.set_index("GEN", inplace=True)
hp_land = pd.read_csv("input/Bev_data_Technik.csv")

#%%
'''
Determine HP correlation
'''
# Data processing
hp_land = pd.DataFrame(hp_land[["GEN", "HP"]]).set_index("GEN")

# Merge both Dfs
pca_hp_zensus_merged = pd.merge(hp_land, pca_hp_zensus, left_index=True, right_index=True, how="inner")

# Columns that were summed, normalize by population
# Filter columns
sum_cols = [col for col, agg in agg_dict_all.items() if agg == "sum"]

# Normalize
pca_hp_zensus_merged_norm = pca_hp_zensus_merged.copy()
pca_hp_zensus_merged_norm[sum_cols] = pca_hp_zensus_merged[sum_cols].div(pca_hp_zensus_merged["Zensus_Einwohner"], axis=0)

# Calculate correlations
correlations_hp = pca_hp_zensus_merged_norm.corr()

# Sort correlations
correlations_hp_sorted = correlations_hp.sort_values(by="HP", ascending=False)

# Sort by absolute values
corr_hp_sort_abs = correlations_hp.reindex(correlations_hp['HP'].abs().sort_values(ascending=False).index)

# Select the top 10 percent
corr_hp_sort_abs.drop(index=['HP'], inplace=True)
corr_hp_sort_abs_10 = corr_hp_sort_abs.head(int(0.1*len(corr_hp_sort_abs)))

# %%
'''
Calculation of Variance Inflation Factor
'''
#%% # Relevant variables for VIF
vif_data_hp = pca_hp_zensus_merged_norm[corr_hp_sort_abs_10.index]

# Perform VIF until no VIF is above 10
while True:
    vif_hp = pd.DataFrame()
    vif_hp["VIF"] = [variance_inflation_factor(vif_data_hp.values, i) for i in range(vif_data_hp.shape[1])]
    vif_hp["Feature"] = vif_data_hp.columns
    vif_hp = vif_hp.sort_values("VIF", ascending=True)

    if vif_hp["VIF"].iloc[-1] < 10:
        break

    # Remove the variable with the highest VIF
    vif_data_hp = vif_data_hp.drop(columns=[vif_hp["Feature"].iloc[-1]])

# Correlation coefficients of filtered variables saved in a separate DF
corr_filtered_hp = correlations_hp[vif_hp['Feature']].loc['HP']



# %%
'''
Minimization with scipy.optimize.least_squares
'''
#%%
# Load data
Bev_data_Zensus_hp = pd.read_csv("input/Bev_data_Zensus_land_all.csv")
gpd_bundesland_hp = gpd.read_file("input/georef-germany-postleitzahl.geojson")
data_zensus_hp = pd.read_csv("input/zensus_sum/data_hp.csv")

#%%
# Prepare data for optimization
Bev_data_Zensus_hp = (Bev_data_Zensus_hp.agg(agg_dict_all).reset_index().set_index('index').T).copy()
Bev_data_Zensus_hp.index = ['Germany']
zensus_buses_hp = pca_hp_zensus_merged.reset_index().copy()
zensus_buses_hp['Bund'] = 'Germany'


#%%
# Calculate Germany total HP
hp_germany = pd.DataFrame({'HP': [zensus_buses_hp['HP'].sum()], 'Bund': ['Germany']})
hp_germany = hp_germany.set_index('Bund')

#%% Filter dataframes to relevant columns only
columns_to_keep_hp = list(corr_filtered_hp.index) 
Bev_data_Zensus_hp = Bev_data_Zensus_hp.loc[:, columns_to_keep_hp]
columns_to_keep_hp.append('Bund')
columns_to_keep_hp.append('HP')
zensus_buses_hp = zensus_buses_hp.loc[:, columns_to_keep_hp]

# Initialize dataframe to store factors
F_hp = zensus_buses_hp[['HP', 'Bund']].copy()
zensus_buses_hp.drop(columns=['HP'], inplace=True)
F_hp['factor'] = None

# Calculate factors: (HP in plz / HP in Germany)
for idx, row in F_hp.iterrows():
    F_hp.at[idx, 'factor'] = (F_hp.at[idx, 'HP'] / hp_germany.loc[row['Bund']])


#%% Function for least_squares
def function_vectorized_ls(f, zensus_bus, Bev_data_Zensus, res):

    # Numerator per PLZ
    numerator = zensus_bus.drop(columns=['Bund']).values @ f  

    # Denominator per Bundesland
    denominator_map = dict(zip(Bev_data_Zensus.index, Bev_data_Zensus.values @ f))

    # Assign correct denominator per PLZ
    denominator = np.array([denominator_map[name] for name in zensus_bus['Bund']])
    
    # Calculate residuals
    x = numerator / denominator - res.to_numpy(dtype=float)

    return np.asarray(x, dtype=float)


#%% Perform minimization
zensus_buses_short_hp = zensus_buses_hp.copy()
faktoren_short_hp = F_hp['factor']

# Check for NaNs
nan_indices_hp = faktoren_short_hp[faktoren_short_hp.isna()].index

# Remove corresponding rows everywhere
if len(nan_indices_hp) > 0:
    faktoren_short_hp = faktoren_short_hp.drop(index=nan_indices_hp).reset_index(drop=True)
    zensus_buses_short_hp = zensus_buses_short_hp.drop(index=nan_indices_hp).reset_index(drop=True)

# Initial values for factors
num_f_hp = zensus_buses_short_hp.drop(columns=['Bund']).shape[1]
f0_hp = np.ones(num_f_hp)

# Perform minimization
results_hp = least_squares(function_vectorized_ls, f0_hp, args=(zensus_buses_short_hp, Bev_data_Zensus_hp, faktoren_short_hp))


# %%
# Store results in CSV
columns = zensus_buses_short_hp.drop(columns=['Bund']).columns
results = results_hp.x

# Load existing CSV
csv = pd.read_csv("input/Faktoren_opt.csv")
csv.set_index("Technik", inplace=True)

# Add columns with respective values from results
for col, res in zip(columns, results):
    csv.at['HP', col] = res

# Fill missing values with 0
csv.fillna(0, inplace=True)
csv.to_csv("input/Faktoren_opt.csv")
# %%
