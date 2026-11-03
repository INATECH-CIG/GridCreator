#!/usr/bin/env python3
"""Generate a topology-independent household profile database with GridCreator.

The script deliberately uses only GridCreator's stochastic household profile
function. It does not create a grid and does not assign profiles to buses.
Topology placement and technology ownership are separate experiment steps.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import creating_demand_and_load as profile_generator  # noqa: E402
import functions as gridcreator_functions  # noqa: E402


REQUIRED_WEATHER_VARIABLES = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "surface_pressure",
    "surface_solar_radiation_downwards",
    "total_sky_direct_solar_radiation_at_surface",
    "total_cloud_cover",
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "weather_year",
        "start_time",
        "time_discretization_seconds",
        "bbox",
        "profiles_per_household_size",
        "seed_start",
        "electrical_demand_method",
        "light_configuration",
        "include_occupancy",
        "output_directory",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")

    sizes = config["profiles_per_household_size"]
    if set(sizes) != {"1", "2", "3", "4", "5"}:
        raise ValueError("profiles_per_household_size must define sizes 1 to 5")
    if any(int(count) < 0 for count in sizes.values()):
        raise ValueError("Profile counts must be non-negative")

    resolution = int(config["time_discretization_seconds"])
    if resolution <= 0 or 3600 % resolution != 0:
        raise ValueError(
            "time_discretization_seconds must be a positive divisor of 3600"
        )

    bbox = config["bbox"]
    if not {"left", "bottom", "right", "top"}.issubset(bbox):
        raise ValueError("bbox must contain left, bottom, right and top")
    if float(bbox["left"]) >= float(bbox["right"]):
        raise ValueError("bbox.left must be smaller than bbox.right")
    if float(bbox["bottom"]) >= float(bbox["top"]):
        raise ValueError("bbox.bottom must be smaller than bbox.top")


def expected_timesteps(year: int, resolution_seconds: int) -> int:
    days = 366 if calendar.isleap(year) else 365
    return days * 24 * 3600 // resolution_seconds


def weather_files(input_directory: Path, weather_year: int) -> list[Path]:
    weather_directory = input_directory / f"weather_{weather_year}"
    return [
        weather_directory / f"GER_{variable}.nc"
        for variable in REQUIRED_WEATHER_VARIABLES
    ]


def validate_inputs(config: dict[str, Any], input_directory: Path) -> None:
    weather_year = int(config["weather_year"])
    missing = [path for path in weather_files(input_directory, weather_year) if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing GridCreator weather files:\n{formatted}")

    # GridCreator currently reads input/weather_2013 explicitly. Refuse to
    # produce mislabeled output until the upstream function is parameterized.
    if weather_year != 2013:
        raise NotImplementedError(
            "This GridCreator checkout hard-codes input/weather_2013. "
            "Parameterize functions.env_weather before using another year."
        )

    start_year = pd.Timestamp(config["start_time"]).year
    if start_year != weather_year:
        raise ValueError(
            f"start_time year ({start_year}) differs from weather_year ({weather_year})"
        )


def make_environment(config: dict[str, Any], input_directory: Path):
    bbox_config = config["bbox"]
    bbox = [
        float(bbox_config["left"]),
        float(bbox_config["bottom"]),
        float(bbox_config["right"]),
        float(bbox_config["top"]),
    ]
    resolution = int(config["time_discretization_seconds"])
    timesteps = expected_timesteps(int(config["weather_year"]), resolution)
    return gridcreator_functions.env_weather(
        bbox=bbox,
        input_path=str(input_directory),
        time_discretization=resolution,
        timesteps_horizon=timesteps,
        timesteps_used_horizon=timesteps,
        timesteps_total=timesteps,
    )


def profile_plan(config: dict[str, Any], limit: int | None = None):
    seed = int(config["seed_start"])
    produced = 0
    for occupants_text, count in config["profiles_per_household_size"].items():
        occupants = int(occupants_text)
        for sequence in range(1, int(count) + 1):
            if limit is not None and produced >= limit:
                return
            yield {
                "profile_id": f"hh_{occupants}p_{sequence:03d}",
                "occupants": occupants,
                "seed": seed,
            }
            seed += 1
            produced += 1


def write_technology_template(profile_ids: list[str], path: Path) -> None:
    """Create an unassigned, Excel-friendly CSV without replacing personal assignments."""
    fields = [
        "profile_id", "has_pv", "pv_p_nom_kw", "pv_orientation", "pv_tilt_deg",
        "has_ev", "ev_charger_kw", "ev_battery_kwh", "has_heat_pump",
        "heat_pump_p_nom_kw", "has_stationary_battery", "battery_power_kw",
        "battery_capacity_kwh", "assignment_status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = path.open("x", encoding="utf-8", newline="")
    except FileExistsError:
        print(f"Keeping existing technology assignments: {path}")
        return
    with stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=";")
        writer.writeheader()
        for profile_id in profile_ids:
            row = dict.fromkeys(fields, 0)
            row.update(
                profile_id=profile_id,
                pv_orientation="none",
                assignment_status="unassigned",
            )
            writer.writerow(row)


def occupancy_series(occupancy: Any, index: pd.DatetimeIndex) -> pd.Series:
    values = np.rint(occupancy.get_occ_profile_in_curr_timestep()).astype(int)
    if len(values) != len(index):
        raise ValueError(
            f"Occupancy length {len(values)} does not match index length {len(index)}"
        )
    return pd.Series(values, index=index, dtype="int16")


def generate_profiles(
    config: dict[str, Any],
    input_directory: Path,
    output_directory: Path,
    limit: int | None,
) -> None:
    environment = make_environment(config, input_directory)
    resolution = int(config["time_discretization_seconds"])
    timesteps = expected_timesteps(int(config["weather_year"]), resolution)
    snapshots = pd.date_range(
        start=pd.Timestamp(config["start_time"]),
        periods=timesteps,
        freq=pd.Timedelta(seconds=resolution),
    )

    load_profiles: dict[str, pd.Series] = {}
    occupancy_profiles: dict[str, pd.Series] = {}
    metadata: list[dict[str, Any]] = []
    step_hours = resolution / 3600

    for item in profile_plan(config, limit):
        profile_id = item["profile_id"]
        seed = item["seed"]
        random.seed(seed)
        np.random.seed(seed)

        demand, occupancy = profile_generator.create_appartment(
            env=environment,
            people=item["occupants"],
            index=snapshots,
            light_config=int(config["light_configuration"]),
            meth=int(config["electrical_demand_method"]),
        )
        if len(demand) != len(snapshots):
            raise ValueError(
                f"{profile_id}: demand length {len(demand)} does not match {len(snapshots)}"
            )

        demand = demand.astype(float).rename(profile_id)
        load_profiles[profile_id] = demand
        if bool(config["include_occupancy"]):
            occupancy_profiles[profile_id] = occupancy_series(occupancy, snapshots)

        metadata.append(
            {
                "profile_id": profile_id,
                "seed": seed,
                "occupants": item["occupants"],
                "weather_year": int(config["weather_year"]),
                "time_resolution_minutes": resolution // 60,
                "electrical_demand_method": int(config["electrical_demand_method"]),
                "light_configuration": int(config["light_configuration"]),
                "annual_demand_mwh": float(demand.sum() * step_hours),
                "mean_demand_kw": float(demand.mean() * 1000),
                "peak_demand_kw": float(demand.max() * 1000),
            }
        )

    if not metadata:
        raise ValueError("The configuration requests zero household profiles")

    output_directory.mkdir(parents=True, exist_ok=True)
    load_frame = pd.DataFrame(load_profiles, index=snapshots)
    load_frame.index.name = "timestamp"
    load_frame.to_parquet(output_directory / "household_timeseries.parquet")

    if occupancy_profiles:
        occupancy_frame = pd.DataFrame(occupancy_profiles, index=snapshots)
        occupancy_frame.index.name = "timestamp"
        occupancy_frame.to_parquet(output_directory / "occupancy_timeseries.parquet")

    metadata_frame = pd.DataFrame(metadata).sort_values("profile_id")
    metadata_frame.to_csv(output_directory / "household_metadata.csv", index=False)
    write_technology_template(
        metadata_frame["profile_id"].tolist(),
        output_directory / "technology_assignments_template.csv",
    )

    manifest = {
        "generator": "INATECH-CIG/GridCreator",
        "generator_repository": "https://github.com/INATECH-CIG/GridCreator",
        "topology_independent": True,
        "profile_count": len(metadata_frame),
        "snapshot_count": len(snapshots),
        "first_snapshot": snapshots[0].isoformat(),
        "last_snapshot": snapshots[-1].isoformat(),
        "configuration": config,
        "files": {
            "technology_assignments": "technology_assignments_template.csv",
            "metadata": "household_metadata.csv",
            "demand": "household_timeseries.parquet",
            "occupancy": (
                "occupancy_timeseries.parquet" if occupancy_profiles else None
            ),
        },
    }
    with (output_directory / "profile_generation_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    print(f"Generated {len(metadata_frame)} profiles in {output_directory}")
    print(f"Snapshots: {snapshots[0]} to {snapshots[-1]} ({len(snapshots)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "profile_config.json",
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Generate only the first N profiles for a quick integration test.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration and weather inputs without generating profiles.",
    )
    parser.add_argument(
        "--template-only",
        action="store_true",
        help="Create only an unassigned technology CSV; preserve an existing table.",
    )
    args = parser.parse_args()
    if args.template_only and args.validate_only:
        parser.error("--template-only and --validate-only cannot be combined")

    config_path = args.config.resolve()
    config = load_config(config_path)
    input_directory = REPO_ROOT / "input"

    configured_output = Path(config["output_directory"])
    output_directory = (
        configured_output
        if configured_output.is_absolute()
        else REPO_ROOT / configured_output
    )

    if args.template_only:
        write_technology_template(
            [item["profile_id"] for item in profile_plan(config, args.limit)],
            output_directory / "technology_assignments_template.csv",
        )
        return

    validate_inputs(config, input_directory)
    if args.validate_only:
        requested = sum(
            int(value) for value in config["profiles_per_household_size"].values()
        )
        print(f"Configuration valid: {requested} profiles requested")
        print(f"Weather input: {input_directory / 'weather_2013'}")
        print(f"Output directory: {output_directory}")
        return

    generate_profiles(
        config=config,
        input_directory=input_directory,
        output_directory=output_directory,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
