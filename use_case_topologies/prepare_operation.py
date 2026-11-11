"""Prepare, optimize, and export topology scenarios as native PyPSA CSV data.

Example (prices in EUR/MWh):
    python use_case_topologies/prepare_operation.py --scenario-root PATH \
        --output-root PATH --import-price 300

EVs use continuous battery energy balances, home-only charging, fixed driving
demand (1.1 kW while away, matching create_e_car), and a minimum departure SOC.
EV and stationary batteries have cyclic horizon boundaries. The horizon thus
represents a repeating period; there is no free initial battery energy. Strict
departure targets can make the optimization infeasible. No V2G is enabled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


def read_timeseries(path: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    table = pd.read_csv(path, index_col=0, parse_dates=True)
    table.columns = table.columns.astype(str)
    if not table.index.equals(index):
        raise ValueError(f"Timestamps differ from network snapshots: {path}")
    if not np.isfinite(table.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite values in {path}")
    return table


def restrict_storage_to_household(n: pypsa.Network) -> None:
    """Give stationary batteries no path to export beyond their household.

    Separate lossless accounting buses route PV either to household consumption
    or to the physical LV bus. The household can import, but cannot send battery
    power back to the LV bus. These links are not additional physical equipment.
    """
    batteries = n.storage_units.index[n.storage_units.carrier == "battery"]
    for carrier in ("household", "pv_routing"):
        if carrier not in n.carriers.index:
            n.add("Carrier", carrier)
    for bus in n.storage_units.loc[batteries, "bus"].unique():
        local_batteries = batteries[n.storage_units.loc[batteries, "bus"] == bus]
        loads = n.loads.index[n.loads.bus == bus]
        chargers = n.links.index[
            (n.links.bus0 == bus) & (n.links.carrier == "ev_charger")
        ]
        solar = n.generators.index[
            (n.generators.bus == bus) & (n.generators.carrier == "solar")
        ]
        household_bus = f"household_at_{bus}"
        n.add("Bus", household_bus, carrier="household", v_nom=n.buses.at[bus, "v_nom"])
        # Bound imports by all local demand plus installed charging powers.
        import_capacity = float(n.storage_units.loc[local_batteries, "p_nom"].sum())
        import_capacity += float(n.links.loc[chargers, "p_nom"].sum())
        for load in loads:
            profile = n.loads_t.p_set.get(load)
            import_capacity += (
                float(profile.max())
                if profile is not None
                else float(n.loads.at[load, "p_set"])
            )
        n.add(
            "Link",
            f"household_import_at_{bus}",
            bus0=bus,
            bus1=household_bus,
            carrier="household",
            p_nom=max(import_capacity, 1e-9),
            p_min_pu=0.0,
            efficiency=1.0,
        )
        n.loads.loc[loads, "bus"] = household_bus
        n.storage_units.loc[local_batteries, "bus"] = household_bus
        n.links.loc[chargers, "bus0"] = household_bus
        for generator in solar:
            pv_bus = f"{generator}_routing_bus"
            nominal = float(n.generators.at[generator, "p_nom"])
            n.add("Bus", pv_bus, carrier="pv_routing", v_nom=n.buses.at[bus, "v_nom"])
            n.generators.at[generator, "bus"] = pv_bus
            n.add(
                "Link",
                f"{generator}_to_household",
                bus0=pv_bus,
                bus1=household_bus,
                carrier="pv_routing",
                p_nom=nominal,
                p_min_pu=0.0,
                efficiency=1.0,
            )
            n.add(
                "Link",
                f"{generator}_to_grid",
                bus0=pv_bus,
                bus1=bus,
                carrier="pv_routing",
                p_nom=nominal,
                p_min_pu=0.0,
                efficiency=1.0,
            )


def prepare_network(
    n: pypsa.Network,
    occupancy: pd.DataFrame,
    heat_kw: pd.DataFrame,
    *,
    import_price: float,
    cop: float,
    departure_soc: float = 1.0,
    export_price: float | None = 0.0,
) -> dict[str, pd.DataFrame]:
    """Modify a freshly loaded scenario, returning auditable EV input tables."""
    if not np.isfinite(import_price) or import_price <= 0:
        raise ValueError("import_price must be positive and finite")
    if export_price is not None and (
        not np.isfinite(export_price) or not 0 <= export_price < import_price
    ):
        raise ValueError("export_price must be nonnegative and below import_price")
    if not np.isfinite(cop) or cop <= 0 or not 0 <= departure_soc <= 1:
        raise ValueError("COP must be positive; departure SOC must lie in [0, 1]")
    index = pd.DatetimeIndex(n.snapshots)
    if len(index) < 2 or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("At least two increasing, unique snapshots are required")
    intervals = (index[1:] - index[:-1]).total_seconds().to_numpy() / 3600.0
    if intervals[0] <= 0 or not np.allclose(intervals, intervals[0]):
        raise ValueError("A regular snapshot interval is required")
    n.snapshot_weightings.loc[:, :] = float(intervals[0])
    if len(n.transformers) != 1:
        raise ValueError("Expected one MV/LV transformer")

    # Keep all existing network capacities and electrical parameters unchanged.
    for table in (
        n.generators,
        n.storage_units,
        n.stores,
        n.links,
        n.lines,
        n.transformers,
    ):
        for column in table.columns:
            if column.endswith("_extendable"):
                table[column] = False
        if "capital_cost" in table:
            table["capital_cost"] = 0.0

    solar = n.generators.index[n.generators.carrier == "solar"]
    upstream = n.generators.index[n.generators.control == "Slack"]
    if len(upstream) != 1:
        raise ValueError("Expected exactly one upstream Slack generator")
    grid = upstream[0]
    if n.generators.at[grid, "bus"] != n.transformers.iloc[0].bus0:
        raise ValueError("Upstream generator must be on the MV side")
    unexpected = n.generators.index.difference(solar.union(upstream))
    if len(unexpected):
        raise ValueError(f"Unexpected generators: {unexpected.tolist()}")
    n.generators.loc[grid, ["marginal_cost", "p_min_pu", "p_max_pu"]] = [
        import_price,
        0.0,
        1.0,
    ]
    for name in solar:
        nominal = float(n.generators.at[name, "p_nom"])
        if nominal <= 0 or name not in n.generators_t.p_set:
            raise ValueError(f"Missing PV capacity/profile for {name}")
        availability = n.generators_t.p_set[name] / nominal
        if (
            not np.isfinite(availability).all()
            or not availability.between(0, 1 + 1e-9).all()
        ):
            raise ValueError(f"Invalid PV availability for {name}")
        n.generators_t.p_max_pu[name] = availability.clip(0, 1)
        n.generators.loc[name, ["marginal_cost", "p_min_pu"]] = [0.0, 0.0]
    n.generators_t.p_set = n.generators_t.p_set.drop(columns=solar, errors="ignore")
    if export_price is not None:
        n.add(
            "Generator",
            "grid_export",
            bus=n.generators.at[grid, "bus"],
            carrier=n.generators.at[grid, "carrier"],
            p_nom=float(n.generators.at[grid, "p_nom"]),
            p_min_pu=-1.0,
            p_max_pu=0.0,
            marginal_cost=export_price,
        )

    for name in n.loads.index[n.loads.carrier == "heat_pump"]:
        bus = str(n.loads.at[name, "bus"])
        if not heat_kw.index.equals(index) or bus not in heat_kw:
            raise ValueError(f"Missing heat demand for {name}")
        power = heat_kw[bus] / cop / 1000.0
        if not np.isfinite(power).all() or (power < 0).any():
            raise ValueError(f"Invalid heat demand for {name}")
        if (power > float(n.loads.at[name, "p_set"]) + 1e-9).any():
            raise ValueError(f"Heat demand exceeds heat-pump rating for {name}")
        n.loads_t.p_set[name] = power

    stationary = n.storage_units.index[n.storage_units.carrier == "battery"]
    n.storage_units.loc[stationary, "cyclic_state_of_charge"] = True
    n.storage_units.loc[stationary, "marginal_cost"] = 0.0
    for carrier in ("ev_battery", "ev_driving", "ev_charger"):
        if carrier not in n.carriers.index:
            n.add("Carrier", carrier)
    availability_columns, driving_columns, soc_columns = {}, {}, {}
    evs = n.storage_units.loc[n.storage_units.carrier == "ev"].copy()
    if len(evs) and not occupancy.index.equals(index):
        raise ValueError("Occupancy timestamps differ from network snapshots")
    for name, ev in evs.iterrows():
        profile = str(name).removeprefix("ev_")
        if profile not in occupancy:
            raise ValueError(
                f"Missing occupancy profile {profile} for {name}; use the profile "
                "database that generated this scenario (--profile-db)"
            )
        values = occupancy[profile].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid occupancy for {name}")
        values = np.rint(values)
        home = pd.Series((values == values.max()).astype(float), index=index)
        charger_mw = float(ev.p_nom)
        capacity_mwh = charger_mw * float(ev.max_hours)
        if (
            not np.isfinite(capacity_mwh)
            or capacity_mwh <= 0
            or not 0 < charger_mw <= 0.011 + 1e-12
        ):
            raise ValueError(f"{name}: require positive capacity and charger <= 11 kW")
        # SOC is an end-of-interval value: constrain the interval BEFORE departure.
        previous_home = home.shift(1, fill_value=home.iloc[-1])
        departures = (home == 0) & (previous_home == 1)
        minimum_soc = pd.Series(0.0, index=index)
        for position in np.flatnonzero(departures.to_numpy()):
            minimum_soc.iloc[(position - 1) % len(index)] = departure_soc
        drive = (1.0 - home) * 0.0011
        battery_bus = f"{name}_battery_bus"
        n.remove("StorageUnit", name)
        n.add("Bus", battery_bus, carrier="ev_battery")
        n.add(
            "Store",
            name,
            bus=battery_bus,
            carrier="ev_battery",
            e_nom=capacity_mwh,
            e_cyclic=True,
            e_min_pu=minimum_soc,
        )
        n.add(
            "Link",
            f"{name}_charger",
            bus0=ev.bus,
            bus1=battery_bus,
            carrier="ev_charger",
            p_nom=charger_mw,
            p_min_pu=0.0,
            p_max_pu=home,
            efficiency=float(ev.efficiency_store),
        )
        n.add(
            "Load",
            f"{name}_driving",
            bus=battery_bus,
            carrier="ev_driving",
            p_set=drive,
        )
        availability_columns[name] = home
        driving_columns[name] = drive
        soc_columns[name] = minimum_soc
    restrict_storage_to_household(n)
    return {
        "ev_home_availability": pd.DataFrame(availability_columns, index=index),
        "ev_driving_mw": pd.DataFrame(driving_columns, index=index),
        "ev_minimum_soc": pd.DataFrame(soc_columns, index=index),
    }


def optimize_and_export(
    n: pypsa.Network, output: Path, *, solver: str = "highs"
) -> None:
    """Export complete native result series, retaining even all-zero columns.

    PyPSA normally omits columns equal to their defaults. Explicitly retaining
    these columns makes the operating profiles complete for every component.
    """
    status, condition = n.optimize(solver_name=solver)
    if status != "ok" or condition != "optimal":
        raise RuntimeError(
            f"Optimization failed ({status}, {condition}); no results exported to {output}"
        )
    series = [
        ("loads", n.loads, n.loads_t, ("p_set", "p")),
        ("generators", n.generators, n.generators_t, ("p",)),
        ("stores", n.stores, n.stores_t, ("p", "e")),
        (
            "storage_units",
            n.storage_units,
            n.storage_units_t,
            ("p", "p_store", "p_dispatch", "state_of_charge"),
        ),
        ("links", n.links, n.links_t, ("p0", "p1")),
    ]
    complete = {}
    for label, static, dynamic, attributes in series:
        if static.empty:
            continue
        for attribute in attributes:
            frame = (
                dynamic[attribute]
                .reindex(index=n.snapshots, columns=static.index)
                .copy()
            )
            if attribute == "p_set":
                frame = frame.fillna(static.p_set)
            if not np.isfinite(frame.to_numpy(dtype=float)).all():
                raise ValueError(
                    f"Incomplete optimization results: {label}-{attribute}"
                )
            dynamic[attribute] = frame
            complete[f"{label}-{attribute}.csv"] = frame
    output.mkdir(parents=True, exist_ok=True)
    n.export_to_csv_folder(output)
    for filename, frame in complete.items():
        # Match PyPSA CSV's positional snapshot index; snapshots.csv holds dates.
        frame.reset_index(drop=True).to_csv(output / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--profile-db",
        type=Path,
        default=Path(__file__).parent / "household_profile_database_2013",
    )
    parser.add_argument("--import-price", type=float, default=10, help="EUR/MWh")
    parser.add_argument(
        "--export-price",
        type=float,
        default=0.0,
        help="EUR/MWh; default: exports allowed without remuneration",
    )
    parser.add_argument("--departure-soc", type=float, default=0.9)
    parser.add_argument("--solver", default="highs")
    args = parser.parse_args()
    source, target = args.scenario_root.resolve(), args.output_root.resolve()
    paths = (
        [source / "scenario_metadata.json"]
        if (source / "scenario_metadata.json").is_file()
        else sorted(source.rglob("scenario_metadata.json"))
    )
    if not paths:
        raise FileNotFoundError(f"No scenario metadata under {source}")
    occupancy = pd.read_parquet(args.profile_db / "occupancy_timeseries.parquet")
    if "timestamp" in occupancy:
        occupancy = occupancy.set_index("timestamp")
    occupancy.index = pd.DatetimeIndex(occupancy.index).tz_localize(None)
    occupancy.columns = occupancy.columns.astype(str)
    for metadata_path in paths:
        folder = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        n = pypsa.Network()
        n.import_from_csv_folder(folder)
        heat = read_timeseries(folder / "heat_demand.csv", n.snapshots)
        prepare_network(
            n,
            occupancy,
            heat,
            import_price=args.import_price,
            export_price=args.export_price,
            departure_soc=args.departure_soc,
            cop=float(metadata["prosumer_package"]["heat_pump_cop"]),
        )
        output = target / folder.relative_to(source)
        metadata["operation_preparation"] = {
            "source": str(folder),
            "import_price_eur_per_mwh": args.import_price,
            "export_price_eur_per_mwh": args.export_price,
            "departure_soc": args.departure_soc,
            "capacities_fixed": True,
            "battery_boundary": "cyclic",
            "ev_v2g": False,
            "stationary_battery_export": False,
            "ev_driving_kw_while_away": 1.1,
            "heat_pump_operation": "fixed_demand_over_cop",
            "optimized": True,
            "solver": args.solver,
        }
        n.meta = metadata
        optimize_and_export(n, output, solver=args.solver)
        print(f"Optimized and exported {output}")


if __name__ == "__main__":
    main()
