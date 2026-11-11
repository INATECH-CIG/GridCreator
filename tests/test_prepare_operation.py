"""Optimization integration checks on a small topology scenario."""

import sys
from pathlib import Path

import pandas as pd
import pypsa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "use_case_topologies"))
from prepare_operation import optimize_and_export, prepare_network


@pytest.fixture
def scenario():
    n = pypsa.Network()
    index = pd.date_range("2013-01-01", periods=8, freq="15min")
    n.set_snapshots(index)
    for carrier in ("AC", "solar", "grid_connection", "ev", "heat_pump", "baseload"):
        n.add("Carrier", carrier)
    n.add("Bus", "mv", v_nom=20)
    n.add("Bus", "lv", v_nom=0.4)
    n.add("Transformer", "transformer", bus0="mv", bus1="lv",
          s_nom=0.4, r=0.01, x=0.04, s_nom_extendable=True)
    n.add("Generator", "upstream_grid", bus="mv", carrier="grid_connection",
          control="Slack", p_nom=1, p_nom_extendable=True)
    n.add("Generator", "pv", bus="lv", carrier="solar", p_nom=0.01,
          p_set=pd.Series([0.01, 0.01, 0, 0, 0, 0, 0, 0], index=index))
    n.add("Load", "house", bus="lv", carrier="baseload", p_set=0.002)
    n.add("Load", "hp", bus="lv", carrier="heat_pump", p_set=0.005)
    n.add("StorageUnit", "ev_test", bus="lv", carrier="ev",
          p_nom=0.011, max_hours=50 / 11, efficiency_store=0.95)
    occupancy = pd.DataFrame({"test": [1, 1, 0, 0, 1, 1, 1, 1]}, index=index)
    heat = pd.DataFrame({"lv": [3.0] * len(index)}, index=index)
    return n, occupancy, heat


def test_preparation_and_optimization_survive_csv_roundtrip(scenario, tmp_path):
    n, occupancy, heat = scenario
    original_transformer = n.transformers.loc["transformer", ["r", "x", "s_nom"]].copy()
    tables = prepare_network(n, occupancy, heat, import_price=300, cop=3)
    pd.testing.assert_series_equal(
        n.transformers.loc["transformer", ["r", "x", "s_nom"]], original_transformer)
    assert not n.transformers.s_nom_extendable.any()
    assert not n.generators.p_nom_extendable.any()
    assert n.generators.at["upstream_grid", "marginal_cost"] == 300
    assert n.generators.at["pv", "marginal_cost"] == 0
    assert n.generators_t.p_max_pu["pv"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    assert (n.loads_t.p_set.hp == 0.001).all()
    assert "ev_test" not in n.storage_units.index
    assert tables["ev_minimum_soc"].ev_test.iloc[1] == 1
    assert (n.snapshot_weightings.stores == 0.25).all()

    n.export_to_csv_folder(tmp_path)
    restored = pypsa.Network()
    restored.import_from_csv_folder(tmp_path)
    status, condition = restored.optimize(solver_name="highs")
    assert (status, condition) == ("ok", "optimal")
    charger = restored.links_t.p0.ev_test_charger
    assert charger.max() <= 0.011 + 1e-7
    assert charger.iloc[2:4].abs().max() < 1e-7
    assert restored.stores_t.e.ev_test.iloc[1] == pytest.approx(0.05)
    # Cyclic energy balance: all trip consumption is replenished, including losses.
    charged_mwh = charger.sum() * 0.25 * 0.95
    driving_mwh = tables["ev_driving_mw"].ev_test.sum() * 0.25
    assert charged_mwh == pytest.approx(driving_mwh, abs=1e-8)


def test_unreachable_departure_soc_is_infeasible(scenario):
    n, occupancy, heat = scenario
    # Only one 15-minute home interval between two trips; tiny charger.
    occupancy["test"] = [1, 1, 0, 0, 1, 0, 0, 1]
    n.storage_units.at["ev_test", "p_nom"] = 0.0001
    n.storage_units.at["ev_test", "max_hours"] = 500
    prepare_network(n, occupancy, heat, import_price=300, cop=3)
    _, condition = n.optimize(solver_name="highs", compute_infeasibilities=False)
    assert condition == "infeasible"


def test_rejects_charger_above_11_kw(scenario):
    n, occupancy, heat = scenario
    n.storage_units.at["ev_test", "p_nom"] = 0.022
    with pytest.raises(ValueError, match="11 kW"):
        prepare_network(n, occupancy, heat, import_price=300, cop=3)


def test_export_tariff_sign_and_no_arbitrage(scenario):
    n, occupancy, heat = scenario
    prepare_network(n, occupancy, heat, import_price=300, export_price=80, cop=3)
    assert n.generators.at["grid_export", "p_min_pu"] == -1
    assert n.generators.at["grid_export", "p_max_pu"] == 0
    assert n.generators.at["grid_export", "marginal_cost"] == 80
    with pytest.raises(ValueError, match="below import_price"):
        prepare_network(n, occupancy, heat, import_price=300, export_price=300, cop=3)


def test_missing_occupancy_identifies_required_profile(scenario):
    n, occupancy, heat = scenario
    with pytest.raises(ValueError, match="profile test.*--profile-db"):
        prepare_network(n, occupancy.drop(columns="test"), heat, import_price=300, cop=3)


def test_scenario_without_prosumers(scenario):
    n, occupancy, heat = scenario
    n.remove("StorageUnit", "ev_test")
    n.remove("Generator", "pv")
    n.remove("Load", "hp")
    tables = prepare_network(
        n, occupancy.iloc[:, :0], heat.iloc[:, :0], import_price=300, cop=3)
    assert all(table.shape == (8, 0) for table in tables.values())
    assert len(n.stores) == 0


def test_heat_demand_above_capacity_is_rejected(scenario):
    n, occupancy, heat = scenario
    # 5 kW electrical at COP 3 can supply at most 15 kW thermal.
    heat.loc[heat.index[3], "lv"] = 16.0
    with pytest.raises(ValueError, match="Heat demand exceeds heat-pump rating"):
        prepare_network(n, occupancy, heat, import_price=300, cop=3)


def test_heat_demand_at_capacity_is_accepted(scenario):
    n, occupancy, heat = scenario
    heat["lv"] = 15.0
    prepare_network(n, occupancy, heat, import_price=300, cop=3)
    assert (n.loads_t.p_set.hp == 0.005).all()


def add_stationary_battery(n):
    n.add("Carrier", "battery")
    n.add("StorageUnit", "battery", bus="lv", carrier="battery",
          p_nom=0.005, max_hours=2, efficiency_store=0.95,
          efficiency_dispatch=0.95)


def test_pv_exports_allowed_and_battery_routing_survives_export(scenario, tmp_path):
    n, occupancy, heat = scenario
    add_stationary_battery(n)
    n.generators.at["pv", "p_nom"] = 0.1
    n.generators_t.p_set["pv"] *= 10
    prepare_network(n, occupancy, heat, import_price=300, export_price=80, cop=3)
    assert n.storage_units.at["battery", "bus"] == "household_at_lv"
    assert n.links.at["ev_test_charger", "bus0"] == "household_at_lv"
    assert n.links.at["pv_to_grid", "bus1"] == "lv"
    assert n.links.at["household_import_at_lv", "p_min_pu"] == 0
    n.export_to_csv_folder(tmp_path)
    restored = pypsa.Network()
    restored.import_from_csv_folder(tmp_path)
    assert restored.optimize(solver_name="highs") == ("ok", "optimal")
    assert restored.generators_t.p.grid_export.min() < -0.001
    assert (restored.links_t.p0.household_import_at_lv >= -1e-7).all()
    assert (restored.links_t.p0.pv_to_grid >= -1e-7).all()


def test_battery_cannot_discharge_more_than_household_consumption(scenario):
    n, occupancy, heat = scenario
    add_stationary_battery(n)
    prepare_network(n, occupancy, heat, import_price=300, cop=3)
    # At snapshot 2 the EV is away; household + heating consume 3 kW.
    # Forcing 4 kW of battery output would require exporting the spare 1 kW.
    def force_battery_export(network, snapshots):
        discharge = network.model["StorageUnit-p_dispatch"].sel(
            name="battery", snapshot=snapshots[2])
        charge = network.model["StorageUnit-p_store"].sel(
            name="battery", snapshot=snapshots[2])
        network.model.add_constraints(discharge - charge >= 0.004,
                                      name="force_battery_export")

    _, condition = n.optimize(
        solver_name="highs", extra_functionality=force_battery_export,
        compute_infeasibilities=False,
    )
    assert condition == "infeasible"


def test_exports_enabled_by_default(scenario):
    n, occupancy, heat = scenario
    prepare_network(n, occupancy, heat, import_price=300, cop=3)
    assert n.generators.at["grid_export", "marginal_cost"] == 0
    assert n.generators.at["grid_export", "p_min_pu"] == -1


def test_complete_native_operating_series_export(scenario, tmp_path):
    n, occupancy, heat = scenario
    add_stationary_battery(n)
    n.add("Load", "idle_load", bus="lv", carrier="baseload", p_set=0)
    prepare_network(n, occupancy, heat, import_price=300, cop=3)
    output = tmp_path / "operation"
    output.mkdir()
    source = output / "source_scenario"
    source.mkdir()
    source_metadata = source / "scenario_metadata.json"
    source_metadata.write_text('{"source": true}', encoding="utf-8")
    manifest = output / "experiment_manifest.json"
    manifest.write_text('{"experiment": true}', encoding="utf-8")
    optimize_and_export(n, output)
    assert source_metadata.read_text(encoding="utf-8") == '{"source": true}'
    assert manifest.read_text(encoding="utf-8") == '{"experiment": true}'
    expected = {
        "loads-p_set.csv": n.loads.index,
        "loads-p.csv": n.loads.index,
        "generators-p.csv": n.generators.index,
        "stores-p.csv": n.stores.index,
        "stores-e.csv": n.stores.index,
        "storage_units-p.csv": n.storage_units.index,
        "storage_units-p_store.csv": n.storage_units.index,
        "storage_units-p_dispatch.csv": n.storage_units.index,
        "storage_units-state_of_charge.csv": n.storage_units.index,
        "links-p0.csv": n.links.index,
        "links-p1.csv": n.links.index,
    }
    for filename, names in expected.items():
        table = pd.read_csv(output / filename, index_col=0)
        assert table.shape == (len(n.snapshots), len(names))
        assert table.columns.tolist() == names.tolist()
        assert table.notna().all().all()
    assert not list(output.glob("ev_*.csv"))
    restored = pypsa.Network()
    restored.import_from_csv_folder(output)
    pd.testing.assert_frame_equal(restored.loads_t.p, n.loads_t.p,
                                  check_dtype=False, check_freq=False)
    pd.testing.assert_frame_equal(restored.stores_t.e, n.stores_t.e,
                                  check_dtype=False, check_freq=False)
    pd.testing.assert_frame_equal(restored.links_t.p0, n.links_t.p0,
                                  check_dtype=False, check_freq=False)


def test_failed_optimization_does_not_export_results(scenario, tmp_path):
    n, occupancy, heat = scenario
    occupancy["test"] = [1, 1, 0, 0, 1, 0, 0, 1]
    n.storage_units.at["ev_test", "p_nom"] = 0.0001
    n.storage_units.at["ev_test", "max_hours"] = 500
    prepare_network(n, occupancy, heat, import_price=300, cop=3)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="infeasible"):
        optimize_and_export(n, output)
    assert not output.exists()
