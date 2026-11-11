# Topology-independent household profile database

This folder generates a reusable household demand database with GridCreator's
stochastic household model. It intentionally does not create or require an
electrical network. Profile-to-bus assignment is a separate experiment step.

## Files

- `profile_config.json`: household counts, seeds, weather location and output.
- `generate_household_profiles.py`: database generator.
- `initialize_profile_database.py`: create schema-valid placeholder files
  without executing GridCreator.
- `README.md`: usage and methodological notes.

## Default population



## Run

Activate the GridCreator environment from the repository root:

```powershell
conda activate GridCreator
python use_case_topologies/generate_household_profiles.py --validate-only
python use_case_topologies/generate_household_profiles.py --limit 2
python use_case_topologies/generate_household_profiles.py
```

`--limit 2` is a quick integration test. Delete or redirect its output before
the full run if you do not want the full run to overwrite it.

To initialize the CSV and Parquet schemas before the GridCreator environment
is available, run:

```powershell
python use_case_topologies/initialize_profile_database.py
```

The initialized Parquet files contain null values, not synthetic data. The
manifest labels them `initialized_not_generated`. The full generator replaces
the demand and occupancy files with calculated profiles.

## Outputs

By default, outputs are written to
`output/household_profile_database_2013/`:

- `household_metadata.csv`: profile parameters and summary statistics.
- `technology_assignments_template.csv`: separate, initially unassigned
  technology parameters for every household profile.
- `household_timeseries.parquet`: demand in MW; timestamps are rows and
  `profile_id` values are columns.
- `occupancy_timeseries.parquet`: integer occupancy profiles in the same form.
- `profile_generation_manifest.json`: exact configuration and file inventory.

Parquet is used to keep the annual database compact. `pyarrow` is already
listed in GridCreator's requirements.

## Reproducibility rules

1. Do not regenerate profiles between topology or placement cases. Select the
   same profile IDs and change only their bus mapping.
2. Record the Git commit of GridCreator alongside the generated database.
3. Keep `profile_config.json` with the simulation results.
4. Use independent seeds only when defining an explicit profile realization.

## Scope

This first generator covers inflexible household electricity demand and
occupancy. PV, EV, heat-pump and stationary-battery ownership should be added
as separate technology tables. In particular, GridCreator's current heat-pump
function produces a temperature-driven electrical schedule rather than the
exogenous thermal demand required by the GrECCo heat-pump optimization.

## Weather limitation

The current GridCreator checkout reads `input/weather_2013` explicitly. The
generator therefore rejects other years instead of silently mislabeling them.
Parameterize `functions.env_weather` before changing `weather_year`.

## Prepare and optimize topology scenarios

Scenario building checks EV battery feasibility before exporting each scenario.
It calculates the minimum usable capacity from driving consumption and maximum
home charging at 95% efficiency, including short charging stops and the cyclic
horizon boundary. An undersized battery raises an error with the minimum kWh
required. If total driving energy exceeds possible home charging, the error
explains that no finite battery size can make the profile feasible. This check
allows SOC between zero and one; departure SOC requirements and network limits
can impose further restrictions during optimization.

`prepare_operation.py` reads one exported scenario or recursively finds scenarios
under a root directory. It prepares each network, runs operational optimization,
and writes separate PyPSA CSV networks with results; source files are preserved.
Source and output paths may be nested, and existing output directories are allowed.
Exported files overwrite files with the same names in the destination.
For a single scenario, network files are written directly into the output root.
For multiple scenarios, their paths relative to the source root are preserved.
The default solver is
HiGHS; use `--solver` to select another installed solver.

```powershell
python use_case_topologies/prepare_operation.py --scenario-root use_case_topologies/pypsa_scenarios --output-root use_case_topologies/prepared_operation --import-price 300
```

The import price is in EUR/MWh (300 is an example, not a prescribed tariff).
Exports are allowed at zero remuneration by default. `--export-price 80` sets
remuneration to 80 EUR/MWh. PV can be curtailed. Export remuneration must be below the
import price. All equipment capacities are fixed, and original line and
transformer electrical parameters are preserved. PV profiles become dispatch
availability limits with zero marginal cost. Heat pumps consume the exported
thermal demand divided by the scenario COP; heating is not flexible.

Stationary batteries can only supply their own household loads, heat pump, and
EV charger. One-way household import links prevent battery energy from reaching
the LV network, while separate PV paths allow solar energy to supply the home
or be exported. These lossless accounting links preserve the physical lines and
transformer. PV can export while a battery supplies local consumption, but the
battery itself has no export path. The rule is embedded in the exported network
and works with a plain `n.optimize()` call, without additional callbacks.

EV occupancy is read from `--profile-db` (default:
`household_profile_database_2013` beside the script). As in `create_e_car`, an EV
is home when occupancy equals the household's maximum rounded occupancy, and
driving consumes 1.1 kW while away. Each EV is represented by a battery `Store`,
a one-way charger `Link` limited by its assigned power (at most 11 kW), and a
driving `Load` on the battery bus. Charging efficiency comes from the exported
EV component. There is no V2G. Availability, driving demand, and minimum SOC
are embedded in native PyPSA component files, without duplicate EV CSV files.

`--departure-soc` sets a minimum SOC immediately before departure (default 1.0).
Energy carries between trips; legacy session StartSOC values and immediate
charging profiles are not used. An unreachable departure target or insufficient
trip energy makes optimization infeasible. EV and stationary battery boundaries
are cyclic, treating the entire horizon as a repeating period and preventing
free initial energy. Snapshot weights use the actual interval in hours.

The output includes complete time series for every component, including columns
that are zero throughout the horizon:

| File | Time series |
| --- | --- |
| `loads-p_set.csv`, `loads-p.csv` | Baseline, heat-pump, and EV driving consumption (MW) |
| `generators-p.csv` | Actual PV generation and grid imports/exports (MW) |
| `generators-p_max_pu.csv` | PV availability (per unit) |
| `links-p0.csv`, `links-p1.csv` | EV charging before/after losses and household/PV routing (MW) |
| `stores-p.csv`, `stores-e.csv` | EV battery net power (MW) and stored energy (MWh) |
| `stores-e_min_pu.csv` | EV minimum SOC requirements (per unit) |
| `storage_units-p.csv` | Stationary battery net power (MW) |
| `storage_units-p_store.csv`, `storage_units-p_dispatch.csv` | Stationary charging/discharging power (MW) |
| `storage_units-state_of_charge.csv` | Stationary stored energy (MWh) |

Store power is positive for discharge and negative for charging. EV grid-side
charging is the charger column in `links-p0.csv`; `stores-p.csv` records battery
net power, including driving consumption. Files for absent component types are
omitted. Native CSV time-series row numbers refer to dates in `snapshots.csv`.
Scenario provenance is stored in PyPSA's `meta.json`. Results are exported only
when the solver reports an optimal solution.

Load an optimized scenario, including its time series:

```python
import pypsa

n = pypsa.Network()
n.import_from_csv_folder("use_case_topologies/prepared_operation/TOPOLOGY/SCENARIO")
pv_and_grid_power = n.generators_t.p
ev_battery_energy = n.stores_t.e
ev_charger_power = n.links_t.p0.filter(like="_charger")
```

This optimizes costs for the entire network. It does not independently optimize
each household or perform an AC power-flow voltage assessment.
