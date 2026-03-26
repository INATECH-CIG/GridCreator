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
This is the scipt with which the census data was filtered and the factors for the downscaling of the electric car data were calculated. The factors were calculated by minimizing the squared differences between the actual electric car numbers per PLZ and the estimated electric car numbers per PLZ, which were calculated by multiplying the factors with the respective census features and normalizing them by the sum of the respective census features per federal state.

It can be recalculated by copying the file into the main folder and running it. The factors will be stored in the "input/Faktoren_opt.csv" file.
'''

#%%
# Filtering the car data to only represent the most current data
FZ_Pkw_mit_Elektroantrieb_Zulassungsbezirk_8414538009745447927 = gpd.read_file("input/FZ Pkw mit Elektroantrieb Zulassungsbezirk_-8414538009745447927.geojson")
FZ_Pkw_mit_Elektroantrieb_Zulassungsbezirk_8414538009745447927_aktuell = FZ_Pkw_mit_Elektroantrieb_Zulassungsbezirk_8414538009745447927[FZ_Pkw_mit_Elektroantrieb_Zulassungsbezirk_8414538009745447927["Berichtszeitpunkt"] == "2025.07"]
FZ_Pkw_mit_Elektroantrieb_Zulassungsbezirk_8414538009745447927_aktuell.to_file("input/FZ Pkw mit Elektroantrieb Zulassungsbezirk_-8414538009745447927_aktuell.geojson", driver="GeoJSON")


#%% Data loading
data_pkw = pd.read_csv("input/zensus_sum/data_kreis.csv", sep=",")
FZ_Pkw_mit_Elektroantrieb_Zulassungsbezirk_8414538009745447927 = gpd.read_file("input/FZ Pkw mit Elektroantrieb Zulassungsbezirk_-8414538009745447927_aktuell.geojson")
Bev_data_Zensus_land_all = pd.read_csv("input/Bev_data_Zensus_land_all.csv")
georef_germany_postleitzahl = gpd.read_file("input/georef-germany-postleitzahl.geojson")

#%%
'''
Determine E-Car correlation
'''
#%%
# Data preparation
pca_ecar_zensus = (data_pkw.rename(columns={"Unnamed: 0": "Schluessel_Zulbz"}).assign(Schluessel_Zulbz=lambda df: df["Schluessel_Zulbz"].astype(int).astype(str).str.zfill(5)).set_index("Schluessel_Zulbz")).copy()

ecar_kreis_gpd = FZ_Pkw_mit_Elektroantrieb_Zulassungsbezirk_8414538009745447927.copy()
ecar_kreis_reduced = ecar_kreis_gpd[["Schluessel_Zulbz", 'Pkw_insgesamt', 'Pkw_Elektro_Anteil', 'geometry']]

# Add absolute electric car number
ecar_kreis_reduced["Pkw_Elektro_Anzahl"] = (ecar_kreis_reduced["Pkw_insgesamt"] * ecar_kreis_reduced["Pkw_Elektro_Anteil"] / 100).round().astype(int)
# Delete from total cars and electric share
ecar_kreis_reduced.drop(columns=["Pkw_insgesamt", "Pkw_Elektro_Anteil"], inplace=True)
# Aggregate to districts
ecar_kreis_dissolved = ecar_kreis_reduced.dissolve(by="Schluessel_Zulbz",aggfunc="sum")
ecar_kreis_dissolved.drop(columns=["geometry"], inplace=True)

# Merge both DataFrames
pca_ecar_zensus_merged = ecar_kreis_dissolved.merge(pca_ecar_zensus, left_index=True, right_index=True, how="inner")

# Normalize
pca_ecar_zensus_merged_norm = pca_ecar_zensus_merged.copy()
# Normalize census data by population
# Filter columns that need to be normalized
sum_cols = [col for col, agg in agg_dict_all.items() if agg == "sum"]
pca_ecar_zensus_merged_norm[sum_cols] = pca_ecar_zensus_merged[sum_cols].div(pca_ecar_zensus_merged["Zensus_Einwohner"], axis=0)
# Normalize electric car number by population
pca_ecar_zensus_merged_norm["Pkw_Elektro_Anzahl"] = pca_ecar_zensus_merged["Pkw_Elektro_Anzahl"].div(pca_ecar_zensus_merged["Zensus_Einwohner"], axis=0)

# Determine correlations
correlations_ecar = pca_ecar_zensus_merged_norm.corr()
# Sort correlations
correlations_ecar_sorted = correlations_ecar.sort_values(by="Pkw_Elektro_Anzahl", ascending=False)

# Sort by absolute values
corr_ecar_sort_abs = correlations_ecar.reindex(correlations_ecar['Pkw_Elektro_Anzahl'].abs().sort_values(ascending=False).index)

# Select the top 20 percent
corr_ecar_sort_abs.drop(index=['Pkw_Elektro_Anzahl'], inplace=True)
corr_ecar_sort_abs_20 = corr_ecar_sort_abs.head(int(0.2*len(corr_ecar_sort_abs)))


# %%
'''
Calculation of Variance Inflation Factor
'''
#%% # Relevant variables for VIF
vif_data_ecar = pca_ecar_zensus_merged_norm[corr_ecar_sort_abs_20.index]

# Perform VIF until no VIF is above 10
while True:
    vif_ecar = pd.DataFrame()
    vif_ecar["VIF"] = [variance_inflation_factor(vif_data_ecar.values, i) for i in range(vif_data_ecar.shape[1])]
    vif_ecar["Feature"] = vif_data_ecar.columns
    vif_ecar = vif_ecar.sort_values("VIF", ascending=True)

    if vif_ecar["VIF"].iloc[-1] < 10:
        break

    # Entferne die Variable mit dem höchsten VIF
    vif_data_ecar = vif_data_ecar.drop(columns=[vif_ecar["Feature"].iloc[-1]])

# Correlation coefficients of filtered in extra DF speichern
corr_filtered_ecar = correlations_ecar[vif_ecar['Feature']].loc['Pkw_Elektro_Anzahl']

# %%
'''
Minimization with scipy.optimize.least_squares
'''

#%%
# Data preparation for optimization
Bev_data_Zensus_pkw = Bev_data_Zensus_land_all.copy()
gpd_bundesland_pkw = georef_germany_postleitzahl.copy()
data_zensus_plz_pkw = data_pkw.copy()
data_zensus_plz_pkw = (
    pca_ecar_zensus_merged.reset_index().rename(columns={"Schluessel_Zulbz": "krs_code"}).assign(krs_code=lambda df:df["krs_code"].astype(str).str.zfill(5))).copy()
gpd_bundesland_pkw['krs_code'] = gpd_bundesland_pkw['krs_code'].astype(str).str.zfill(5)

# Merge to get Bundesland info
zensus_buses_pkw = pd.merge(data_zensus_plz_pkw, gpd_bundesland_pkw[['krs_code', 'lan_name']], on='krs_code', how='left')

# E-Car per Bundesland
pkw_bundesland = zensus_buses_pkw.groupby('lan_name')['Pkw_Elektro_Anzahl'].sum()
Bev_data_Zensus_pkw = Bev_data_Zensus_pkw.set_index('GEN')
pkw_bundesland = pkw_bundesland.reindex(Bev_data_Zensus_pkw.index)

# Filter correct columns
columns_to_keep_pkw = list(corr_filtered_ecar.index) 
Bev_data_Zensus_pkw = Bev_data_Zensus_pkw.loc[:, columns_to_keep_pkw]
columns_to_keep_pkw.append('lan_name')
columns_to_keep_pkw.append('Pkw_Elektro_Anzahl')
zensus_buses_pkw = zensus_buses_pkw.loc[:, columns_to_keep_pkw]

# Initialize dataframe to store factors
F_pkw = zensus_buses_pkw[['Pkw_Elektro_Anzahl', 'lan_name']].copy()
zensus_buses_pkw.drop(columns=['Pkw_Elektro_Anzahl'], inplace=True)
F_pkw['factor'] = None

# Calculation of factors: (Pkw_Elektro_Anzahl in plz / Pkw_Elektro_Anzahl in respective Bundesland)
for idx, row in F_pkw.iterrows():
    F_pkw.at[idx, 'factor'] = (F_pkw.at[idx, 'Pkw_Elektro_Anzahl'] / pkw_bundesland.loc[row['lan_name']])

#%% Function for least_squares

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


#%% Perform minimization
zensus_buses_short_pkw = zensus_buses_pkw.copy()
faktoren_short_pkw = F_pkw['factor']

# Check for NaNs
nan_indices_pkw = faktoren_short_pkw[faktoren_short_pkw.isna()].index

# Remove corresponding rows everywhere
if len(nan_indices_pkw) > 0:
    faktoren_short_pkw = faktoren_short_pkw.drop(index=nan_indices_pkw).reset_index(drop=True)
    zensus_buses_short_pkw = zensus_buses_short_pkw.drop(index=nan_indices_pkw).reset_index(drop=True)

# Initial values for factors
num_f_pkw = zensus_buses_short_pkw.drop(columns=['lan_name']).shape[1]
f0_pkw = np.ones(num_f_pkw)

# Perform minimization
results_pkw = least_squares(function_vectorized_ls, f0_pkw, args=(zensus_buses_short_pkw, Bev_data_Zensus_pkw, faktoren_short_pkw))


# %%
# Store results in CSV
columns = zensus_buses_short_pkw.drop(columns=['lan_name']).columns
results = results_pkw.x

# Load existing CSV
csv = pd.read_csv("input/Faktoren_opt.csv")
csv.set_index("Technik", inplace=True)

# Add columns with respective values from results
for col, res in zip(columns, results):
    csv.at['E_car', col] = res

# Fill alle NaNs with 0
csv.fillna(0, inplace=True)
csv.to_csv("input/Faktoren_opt.csv")
# %%
