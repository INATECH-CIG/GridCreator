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
