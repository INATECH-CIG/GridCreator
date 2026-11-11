"""Build the full SimBench LV penetration/placement experiment as PyPSA grids.

This script is intentionally not executed on import. Run it from the
GridCreator repository after generating non-null household Parquet profiles.

GridCreator functionality reused here:
* functions.env_weather for the common 2013 weather environment;
* creating_demand_and_load.create_pv for PV profiles;
* creating_demand_and_load.create_hp for the heat-demand proxy;
* creating_demand_and_load.create_e_car for EV availability and trip energy.

GridCreator.main_functions.loads_assignment is not used because it would draw
new stochastic households for each network. pypsa_preparation is not used
because it replaces the SimBench line impedances and ratings, which are the
topological quantities under study.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

import creating_demand_and_load as gc_profiles
import functions as gc_functions
from scenario_assignment import (
    AssignmentDesign,
    PLACEMENTS,
    QUOTAS,
    build_topology_assignments,
    load_topology,
)

DEFAULT_PROFILE_DB = (
    REPOSITORY / "use_case_topologies" / "household_profile_database_2013"
)
DEFAULT_OUTPUT = REPOSITORY / "use_case_topologies" / "pypsa_scenarios" / "EV load"


@dataclass(frozen=True)
class ProsumerSpec:
    pv_kw: float
    pv_orientation: str
    pv_tilt_deg: float
    ev_charger_kw: float
    ev_battery_kwh: float
    heat_pump_kw: float
    battery_power_kw: float
    battery_capacity_kwh: float
    heat_pump_cop: float = 3.0


class OccupancyAdapter:
    """Give stored occupancy arrays GridCreator's create_e_car interface."""

    def __init__(self, values: pd.Series):
        self._values = values.to_numpy(dtype=float)

    def get_occ_profile_in_curr_timestep(self) -> np.ndarray:
        return self._values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Topologies are supplied locally; require their path rather than a machine-specific default.
    parser.add_argument(
        "--topology-root",
        type=Path,
        required=True,
        help="Path to one PyPSA topology CSV folder or a directory containing topology folders.",
    )
    parser.add_argument("--profile-db", type=Path, default=DEFAULT_PROFILE_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gridcreator-input", type=Path, default=REPOSITORY / "input")
    parser.add_argument("--household-seed", type=int, default=21051991)
    parser.add_argument("--mapping-seed", type=int, default=17071993)
    parser.add_argument("--random-seed-1", type=int, default=101)
    parser.add_argument("--random-seed-2", type=int, default=202)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing scenario output directory.",
    )
    return parser.parse_args()


def read_profile_table(path: Path) -> pd.DataFrame:
    table = pd.read_parquet(path)
    if "timestamp" in table.columns:
        table = table.set_index("timestamp")
    table.index = pd.DatetimeIndex(table.index).tz_localize(None)
    if not table.index.is_monotonic_increasing or table.index.has_duplicates:
        raise ValueError(f"Invalid time index in {path}")
    return table


def require_profiles(table: pd.DataFrame, profile_ids: list[str], label: str) -> None:
    missing_columns = sorted(set(profile_ids) - set(table.columns))
    if missing_columns:
        raise ValueError(f"{label}: missing profile columns: {missing_columns}")
    null_profiles = table[profile_ids].columns[table[profile_ids].isna().any()].tolist()
    if null_profiles:
        raise ValueError(
            f"{label}: profiles still contain null placeholders; run household "
            f"generation first. First affected profiles: {null_profiles[:10]}"
        )


def _one_active_value(table: pd.DataFrame, flag: str, field: str) -> float:
    values = pd.to_numeric(
        table.loc[table[flag].astype(int) == 1, field], errors="raise"
    )
    values = sorted(values[values > 0].unique())
    if len(values) != 1:
        raise ValueError(f"Expected one positive {field} value, found {values}")
    return float(values[0])


def read_prosumer_spec(path: Path) -> ProsumerSpec:
    technology = pd.read_csv(path, sep=";")
    pv_rows = technology.loc[technology["has_pv"].astype(int) == 1]
    orientations = sorted(pv_rows["pv_orientation"].dropna().str.lower().unique())
    tilts = sorted(pd.to_numeric(pv_rows["pv_tilt_deg"], errors="raise").unique())
    if len(orientations) != 1 or len(tilts) != 1:
        raise ValueError("Full-prosumer PV orientation and tilt must each be constant")
    return ProsumerSpec(
        pv_kw=_one_active_value(technology, "has_pv", "pv_p_nom_kw"),
        pv_orientation=orientations[0],
        pv_tilt_deg=float(tilts[0]),
        ev_charger_kw=_one_active_value(technology, "has_ev", "ev_charger_kw"),
        ev_battery_kwh=_one_active_value(technology, "has_ev", "ev_battery_kwh"),
        heat_pump_kw=_one_active_value(
            technology, "has_heat_pump", "heat_pump_p_nom_kw"
        ),
        battery_power_kw=_one_active_value(
            technology, "has_stationary_battery", "battery_power_kw"
        ),
        battery_capacity_kwh=_one_active_value(
            technology, "has_stationary_battery", "battery_capacity_kwh"
        ),
    )


def gridcreator_orientation(label: str) -> str:
    mapping = {
        "south": "Süd",
        "süd": "Süd",
        "southeast": "Süd-Ost",
        "southwest": "Süd-West",
        "east": "Ost",
        "west": "West",
        "north": "Nord",
    }
    try:
        return mapping[label.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported PV orientation: {label}") from exc


def common_gridcreator_profiles(
    index: pd.DatetimeIndex,
    profile_config: dict,
    input_path: Path,
    spec: ProsumerSpec,
):
    step_seconds = int((index[1] - index[0]).total_seconds())
    environment = gc_functions.env_weather(
        bbox=list(profile_config["bbox"].values()),
        input_path=str(input_path),
        time_discretization=step_seconds,
        timesteps_horizon=len(index),
        timesteps_used_horizon=len(index),
        timesteps_total=len(index),
    )

    raw_pv = gc_profiles.create_pv(
        env=environment,
        peakpower=spec.pv_kw,
        index=index,
        beta=spec.pv_tilt_deg,
        gamma=gridcreator_orientation(spec.pv_orientation),
    ).clip(lower=0.0)
    if raw_pv.max() <= 0:
        raise ValueError("GridCreator returned an empty PV profile")
    # normalize raw pv (weather) with max and PV-plants nominal power
    pv_mw = raw_pv / raw_pv.max() * (spec.pv_kw / 1000.0)

    raw_hp_mw = gc_profiles.create_hp(index=index, env=environment).clip(lower=0.0)
    if raw_hp_mw.max() <= 0:
        raise ValueError("GridCreator returned an empty heat-pump profile")
    hp_electric_mw = raw_hp_mw / raw_hp_mw.max() * (spec.heat_pump_kw / 1000.0)
    heat_demand_kw = hp_electric_mw * 1000.0 * spec.heat_pump_cop
    return environment, pv_mw, heat_demand_kw


def add_carrier(network: pypsa.Network, name: str) -> None:
    if name not in network.carriers.index:
        network.add("Carrier", name)


def add_slack(network: pypsa.Network) -> None:
    if (network.generators.get("control", pd.Series(dtype=str)) == "Slack").any():
        return
    mv_bus = str(network.transformers.iloc[0]["bus0"])
    add_carrier(network, "grid_connection")
    network.add(
        "Generator",
        "upstream_grid",
        bus=mv_bus,
        carrier="grid_connection",
        control="Slack",
        p_nom=1e3,
    )


def minimum_ev_battery_kwh(
    trip_mw: pd.Series,
    at_home: pd.Series,
    charger_kw: float,
    efficiency: float = 0.95,
) -> float:
    """Minimum usable capacity for cyclic driving and home-only charging.

    This is an energy-feasibility check with freely scheduled charging and SOC
    between zero and one. Departure SOC targets and grid constraints can impose
    additional restrictions. Short charging stops and horizon wraparound are
    included; increasing capacity cannot fix a total charging-energy deficit.
    """
    index = pd.DatetimeIndex(trip_mw.index)
    if len(index) < 2 or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("EV profiles require at least two increasing timestamps")
    if not at_home.index.equals(trip_mw.index):
        raise ValueError("EV driving and availability timestamps differ")
    hours = (index[1:] - index[:-1]).total_seconds().to_numpy() / 3600.0
    if hours[0] <= 0 or not np.allclose(hours, hours[0]):
        raise ValueError("EV profiles require regular timestamps")
    if not np.isfinite(charger_kw) or not 0 < charger_kw <= 11:
        raise ValueError("EV charger power must be positive and at most 11 kW")
    if not np.isfinite(efficiency) or not 0 < efficiency <= 1:
        raise ValueError("EV charging efficiency must lie in (0, 1]")
    driving = trip_mw.to_numpy(dtype=float) * 1000.0 * hours[0]
    if not np.isfinite(driving).all() or (driving < 0).any():
        raise ValueError("EV driving demand must be finite and nonnegative")
    charging = at_home.astype(bool).to_numpy() * charger_kw * efficiency * hours[0]
    if driving.sum() > charging.sum() + 1e-8:
        raise ValueError(
            f"No finite battery size can make this EV profile feasible: driving "
            f"requires {driving.sum():.2f} kWh, but available home charging can "
            f"supply at most {charging.sum():.2f} kWh over the horizon "
            f"at {charger_kw:g} kW and {efficiency:.0%} efficiency"
        )
    # Maximum circular energy deficit: consumption accumulated between charging
    # opportunities, including sequences of trips with insufficient short stops.
    deficit = driving - charging
    cumulative = np.concatenate(([0.0], np.cumsum(deficit)))
    maximum = float((cumulative - np.minimum.accumulate(cumulative)).max())
    minimum = float((cumulative - np.maximum.accumulate(cumulative)).min())
    return max(0.0, maximum, float(deficit.sum()) - minimum)


def ev_requests(
    profile_id: str,
    bus: str,
    occupancy: pd.Series,
    spec: ProsumerSpec,
) -> pd.DataFrame:
    """Convert GridCreator availability into reproducible GrECCo requests."""
    trip_mw, _, at_home = gc_profiles.create_e_car(
        occ=OccupancyAdapter(occupancy), index=occupancy.index
    )
    at_home = at_home.astype(bool)
    try:
        required_kwh = minimum_ev_battery_kwh(trip_mw, at_home, spec.ev_charger_kw)
    except ValueError as exc:
        raise ValueError(f"EV {profile_id} at bus {bus}: {exc}") from exc
    if not np.isfinite(spec.ev_battery_kwh) or spec.ev_battery_kwh <= 0:
        raise ValueError(f"EV {profile_id}: battery capacity must be positive and finite")
    if spec.ev_battery_kwh + 1e-8 < required_kwh:
        raise ValueError(
            f"EV {profile_id} at bus {bus}: assigned battery {spec.ev_battery_kwh:g} kWh "
            f"is too small; at least {required_kwh:.2f} kWh of usable capacity is "
            "required for driving and available home charging. Departure SOC "
            "targets may require additional capacity."
        )
    departures = np.flatnonzero(at_home.shift(fill_value=True) & ~at_home)
    arrivals = np.flatnonzero(~at_home.shift(fill_value=True) & at_home)
    rows = []
    for departure in departures:
        later_arrivals = arrivals[arrivals > departure]
        if len(later_arrivals) == 0:
            continue
        arrival = int(later_arrivals[0])
        next_departures = departures[departures > arrival]
        deadline = int(next_departures[0]) if len(next_departures) else len(at_home) - 1
        step_hours = (occupancy.index[1] - occupancy.index[0]).total_seconds() / 3600
        trip_kwh = float(trip_mw.iloc[departure:arrival].sum() * step_hours * 1000)
        start_soc = max(0.0, 1.0 - trip_kwh / spec.ev_battery_kwh)
        rows.append(
            {
                "EVID": f"ev_{profile_id}",
                "Bus": bus,
                "ChargerID": f"ev_{profile_id}",
                "Start_Time": occupancy.index[arrival],
                "End_Time": occupancy.index[deadline],
                "StartSOC": start_soc,
                "TargetSOC": 1.0,
            }
        )
    return pd.DataFrame(rows)


def add_households_and_prosumers(
    network: pypsa.Network,
    assignment: pd.DataFrame,
    demand: pd.DataFrame,
    occupancy: pd.DataFrame,
    pv_mw: pd.Series,
    heat_demand_kw: pd.Series,
    spec: ProsumerSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    network.set_snapshots(demand.index)
    for carrier in ("baseload", "solar", "heat_pump", "battery", "ev"):
        add_carrier(network, carrier)
    add_slack(network)

    baseload_series = {}
    pv_series = {}
    heat_columns = {}
    request_frames = []

    for row in assignment.itertuples(index=False):
        bus = str(row.bus)
        profile_id = str(row.profile_id)
        load_name = f"baseload_at_{bus}"
        network.add("Load", load_name, bus=bus, carrier="baseload", p_set=0.0)
        baseload_series[load_name] = demand[profile_id]

        if not bool(row.full_prosumer):
            continue

        pv_name = f"pv_at_{bus}"
        network.add(
            "Generator",
            pv_name,
            bus=bus,
            carrier="solar",
            control="PQ",
            p_nom=spec.pv_kw / 1000.0,
        )
        pv_series[pv_name] = pv_mw

        network.add(
            "Load",
            f"heat_pump_at_{bus}",
            bus=bus,
            carrier="heat_pump",
            p_set=spec.heat_pump_kw / 1000.0,
        )
        heat_columns[bus] = heat_demand_kw

        network.add(
            "StorageUnit",
            f"battery_at_{bus}",
            bus=bus,
            type="h0_battery",
            carrier="battery",
            p_nom=spec.battery_power_kw / 1000.0,
            max_hours=spec.battery_capacity_kwh / spec.battery_power_kw,
            efficiency_store=0.95,
            efficiency_dispatch=0.95,
            cyclic_state_of_charge=True,
        )
        network.storage_units.loc[f"battery_at_{bus}", "capacity"] = (
            spec.battery_capacity_kwh
        )
        network.add(
            "StorageUnit",
            f"ev_{profile_id}",
            bus=bus,
            type="ev_charger",
            carrier="ev",
            p_nom=spec.ev_charger_kw / 1000.0,
            max_hours=spec.ev_battery_kwh / spec.ev_charger_kw,
            efficiency_store=0.95,
            efficiency_dispatch=0.95,
        )
        network.storage_units.loc[f"ev_{profile_id}", "capacity"] = spec.ev_battery_kwh
        request_frames.append(ev_requests(profile_id, bus, occupancy[profile_id], spec))

    network.loads_t.p_set = pd.DataFrame(baseload_series, index=demand.index)
    network.generators_t.p_set = pd.DataFrame(pv_series, index=demand.index)
    heat_demand = pd.DataFrame(heat_columns, index=demand.index)
    requests = (
        pd.concat(request_frames, ignore_index=True)
        if request_frames
        else pd.DataFrame(
            columns=[
                "EVID",
                "Bus",
                "ChargerID",
                "Start_Time",
                "End_Time",
                "StartSOC",
                "TargetSOC",
            ]
        )
    )
    return heat_demand, requests


def ev_charging_load(
    index: pd.DatetimeIndex,
    requests: pd.DataFrame,
    ev_ids: list[str],
    spec: ProsumerSpec,
    efficiency: float = 0.95,
) -> pd.DataFrame:
    """Grid-side MW for immediate charging, averaged over each snapshot interval.

    Sessions use [Start_Time, End_Time); charging stops at the target SOC or
    departure. The final charging interval may use less than rated power.
    """
    load = pd.DataFrame(0.0, index=index, columns=ev_ids)
    step_hours = (index[1] - index[0]).total_seconds() / 3600.0
    charger_mw = spec.ev_charger_kw / 1000.0
    for request in requests.itertuples(index=False):
        remaining_mwh = (
            max(0.0, request.TargetSOC - request.StartSOC)
            * spec.ev_battery_kwh
            / 1000.0
            / efficiency
        )
        start = index.searchsorted(request.Start_Time)
        end = index.searchsorted(request.End_Time)
        for position in range(start, end):
            if remaining_mwh <= 0.0:
                break
            power_mw = min(charger_mw, remaining_mwh / step_hours)
            load.iat[position, load.columns.get_loc(request.EVID)] = power_mw
            remaining_mwh = max(0.0, remaining_mwh - power_mw * step_hours)
    return load


def scenario_name(quota: float, placement: str) -> str:
    return f"q{int(round(quota * 100)):03d}_{placement}"


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output exists: {path}; pass --overwrite to replace it"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)


def find_topology_paths(topology_root: Path) -> list[Path]:
    if (topology_root / "buses.csv").is_file():
        return [topology_root]
    return sorted(path for path in topology_root.iterdir() if path.is_dir())


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.profile_db / "household_metadata.csv")
    profile_ids = metadata["profile_id"].astype(str).tolist()
    demand = read_profile_table(args.profile_db / "household_timeseries.parquet")
    occupancy = read_profile_table(args.profile_db / "occupancy_timeseries.parquet")
    require_profiles(demand, profile_ids, "household demand")
    require_profiles(occupancy, profile_ids, "occupancy")
    if not demand.index.equals(occupancy.index):
        raise ValueError("Demand and occupancy timestamps differ")

    with (args.profile_db / "profile_generation_manifest.json").open(
        encoding="utf-8"
    ) as handle:
        profile_manifest = json.load(handle)
    with (REPOSITORY / "use_case_topologies" / "profile_config.json").open(
        encoding="utf-8"
    ) as handle:
        profile_config = json.load(handle)
    spec = read_prosumer_spec(args.profile_db / "technology_assignments_template.csv")
    _, pv_mw, heat_demand_kw = common_gridcreator_profiles(
        demand.index, profile_config, args.gridcreator_input, spec
    )

    design = AssignmentDesign(
        household_seed=args.household_seed,
        mapping_seed=args.mapping_seed,
        random_seed_1=args.random_seed_1,
        random_seed_2=args.random_seed_2,
    )
    topology_paths = find_topology_paths(args.topology_root)
    if not topology_paths:
        raise ValueError(f"No SimBench topology folders found in {args.topology_root}")

    experiment_rows = []
    for topology_path in topology_paths:
        source_network = load_topology(topology_path)
        base, scenarios = build_topology_assignments(
            topology_path.name, source_network, profile_ids, design, QUOTAS
        )
        topology_output = args.output_root / topology_path.name
        topology_output.mkdir(parents=True, exist_ok=True)
        base.to_csv(topology_output / "base_household_assignment.csv", index=False)

        for quota in QUOTAS:
            for placement in PLACEMENTS:
                assignment = scenarios[(quota, placement)]
                output = topology_output / scenario_name(quota, placement)
                network = load_topology(topology_path)
                heat_demand, ev_data = add_households_and_prosumers(
                    network,
                    assignment,
                    demand,
                    occupancy,
                    pv_mw,
                    heat_demand_kw,
                    spec,
                )
                prepare_output(output, args.overwrite)
                network.export_to_csv_folder(output)
                assignment.to_csv(output / "scenario_assignment.csv", index=False)
                heat_demand.to_csv(output / "heat_demand.csv")
                ev_data.to_csv(output / "ev_charging_processes.csv", index=False)
                ev_ids = network.storage_units.index[
                    network.storage_units["carrier"] == "ev"
                ].tolist()
                ev_charging_load(demand.index, ev_data, ev_ids, spec).to_csv(
                    output / "ev_charging_load.csv"
                )

                scenario_metadata = {
                    "topology": topology_path.name,
                    "quota_requested": quota,
                    "quota_realized": float(assignment["quota_realized"].iloc[0]),
                    "placement": placement,
                    "endpoint_equivalent": bool(
                        assignment["endpoint_equivalent"].iloc[0]
                    ),
                    "household_population_fixed_within_topology": True,
                    "household_bus_mapping_fixed_within_topology": True,
                    "prosumer_package": asdict(spec),
                    "assignment_design": asdict(design),
                    "profile_manifest_status": profile_manifest.get("status"),
                    "ev_charging_load": {
                        "file": "ev_charging_load.csv",
                        "unit": "MW",
                        "strategy": "immediate_charging_on_arrival",
                        "efficiency": 0.95,
                        "columns": "EV StorageUnit names",
                    },
                }
                with (output / "scenario_metadata.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(scenario_metadata, handle, indent=2)
                experiment_rows.append(scenario_metadata)

    pd.DataFrame(experiment_rows).to_json(
        args.output_root / "experiment_manifest.json", orient="records", indent=2
    )


if __name__ == "__main__":
    main()
