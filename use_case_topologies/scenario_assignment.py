"""Create reproducible full-prosumer placement assignments for LV grids.

The base household population and its bus mapping are fixed once per topology.
Scenario variants change only whether that household receives the complete
prosumer package. Random placements are nested prefixes of two fixed random
bus permutations, so increasing a quota never removes an already selected bus.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
import pypsa

QUOTAS = [0.1]  # (0.0, 0.10, 0.25, 0.50, 0.70, 0.90, 1.0)
PLACEMENTS = ["weak_first"]  # ("weak_first", "strong_first", "random_1", "random_2")


@dataclass(frozen=True)
class AssignmentDesign:
    household_seed: int = 21051991
    mapping_seed: int = 17071993
    random_seed_1: int = 101
    random_seed_2: int = 202
    lv_voltage_kv: float = 0.4


def _stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def load_topology(path: Path) -> pypsa.Network:
    network = pypsa.Network()
    network.import_from_csv_folder(path)
    if len(network.transformers) != 1:
        raise ValueError(f"{path.name}: expected exactly one MV/LV transformer")
    return network


def eligible_lv_buses(network: pypsa.Network, lv_voltage_kv: float) -> list[str]:
    """Return every LV bus; the MV transformer bus is excluded by voltage."""
    v_nom = pd.to_numeric(network.buses["v_nom"], errors="raise")
    # trafo connecting bus is not used
    trafo_bus = network.transformers["bus1"].values
    buses = network.buses.index[v_nom <= lv_voltage_kv + 1e-9].astype(str).tolist()
    buses.remove(trafo_bus)
    if not buses:
        raise ValueError("Topology contains no eligible LV buses")
    return sorted(buses)


def electrical_distances(network: pypsa.Network, buses: Iterable[str]) -> pd.Series:
    """Shortest-path impedance magnitude from the LV transformer bus."""
    root = str(network.transformers.iloc[0]["bus1"])
    graph = nx.Graph()
    graph.add_nodes_from(network.buses.index.astype(str))
    for _, line in network.lines.iterrows():
        resistance = float(line.get("r", 0.0) or 0.0)
        reactance = float(line.get("x", 0.0) or 0.0)
        impedance = math.hypot(resistance, reactance)
        graph.add_edge(
            str(line["bus0"]), str(line["bus1"]), weight=max(impedance, 1e-12)
        )

    distances = nx.single_source_dijkstra_path_length(graph, root, weight="weight")
    missing = sorted(set(buses) - set(distances))
    if missing:
        raise ValueError(
            f"LV buses disconnected from transformer root {root}: {missing}"
        )
    return pd.Series(
        {bus: float(distances[bus]) for bus in buses}, name="electrical_distance"
    )


def fixed_household_mapping(
    topology_name: str,
    buses: list[str],
    available_profile_ids: list[str],
    design: AssignmentDesign,
) -> pd.DataFrame:
    """Select one global cohort and map it to buses once for this topology."""
    if len(available_profile_ids) < len(buses):
        raise ValueError(
            f"{topology_name}: {len(buses)} LV buses but only "
            f"{len(available_profile_ids)} household profiles"
        )

    cohort_rng = np.random.default_rng(design.household_seed)
    cohort = np.asarray(available_profile_ids, dtype=object).copy()
    cohort_rng.shuffle(cohort)
    cohort = cohort[: len(buses)]

    bus_rng = np.random.default_rng(_stable_seed(design.mapping_seed, topology_name))
    mapped_buses = np.asarray(sorted(buses), dtype=object)
    bus_rng.shuffle(mapped_buses)

    assignment = pd.DataFrame({"bus": mapped_buses, "profile_id": cohort})
    return assignment.sort_values("bus").reset_index(drop=True)


def placement_orders(
    topology_name: str,
    distances: pd.Series,
    design: AssignmentDesign,
) -> dict[str, list[str]]:
    strong = sorted(distances.index, key=lambda b: (distances[b], b))
    weak = sorted(distances.index, key=lambda b: (-distances[b], b))

    orders = {"weak_first": weak, "strong_first": strong}
    for label, seed in (
        ("random_1", design.random_seed_1),
        ("random_2", design.random_seed_2),
    ):
        rng = np.random.default_rng(_stable_seed(seed, topology_name))
        order = np.asarray(sorted(distances.index), dtype=object)
        rng.shuffle(order)
        orders[label] = order.tolist()
    return orders


def quota_count(quota: float, population: int) -> int:
    """Round halves upward and keep the result inside [0, population]."""
    return min(population, max(0, int(math.floor(quota * population + 0.5))))


def build_topology_assignments(
    topology_name: str,
    network: pypsa.Network,
    available_profile_ids: list[str],
    design: AssignmentDesign = AssignmentDesign(),
    quotas: Iterable[float] = QUOTAS,
) -> tuple[pd.DataFrame, dict[tuple[float, str], pd.DataFrame]]:
    buses = eligible_lv_buses(network, design.lv_voltage_kv)
    distances = electrical_distances(network, buses)
    base = fixed_household_mapping(topology_name, buses, available_profile_ids, design)
    base["electrical_distance"] = base["bus"].map(distances)
    base["topology"] = topology_name

    orders = placement_orders(topology_name, distances, design)
    scenarios: dict[tuple[float, str], pd.DataFrame] = {}
    for quota in quotas:
        quota = float(quota)
        selected_count = quota_count(quota, len(base))
        for placement in PLACEMENTS:
            selected = set(orders[placement][:selected_count])
            scenario = base.copy()
            scenario["quota_requested"] = quota
            scenario["quota_realized"] = selected_count / len(base)
            scenario["placement"] = placement
            scenario["full_prosumer"] = scenario["bus"].isin(selected)
            scenario["endpoint_equivalent"] = quota in (0.0, 1.0)
            scenario["placement_rank"] = scenario["bus"].map(
                {bus: rank + 1 for rank, bus in enumerate(orders[placement])}
            )
            scenarios[(quota, placement)] = scenario
    return base, scenarios
