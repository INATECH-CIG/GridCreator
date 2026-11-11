"""Run from the repository root: python -m pytest tests -v."""

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "use_case_topologies"))
import build_pypsa_scenarios as scenarios


@pytest.fixture
def index():
    return pd.date_range("2013-01-01", periods=25, freq="15min")


@pytest.fixture
def spec():
    return scenarios.ProsumerSpec(
        pv_kw=5.0,
        pv_orientation="south",
        pv_tilt_deg=30.0,
        ev_charger_kw=11.0,
        ev_battery_kwh=50.0,
        heat_pump_kw=5.0,
        battery_power_kw=5.0,
        battery_capacity_kwh=10.0,
    )


@pytest.fixture
def charging_load(index, spec):
    def calculate(*, start_soc, departure, charger_kw=11.0):
        requests = pd.DataFrame(
            [{
                "EVID": "ev_test",
                "Start_Time": index[1],
                "End_Time": index[departure],
                "StartSOC": start_soc,
                "TargetSOC": 1.0,
            }]
        )
        return scenarios.ev_charging_load(
            index, requests, ["ev_test"],
            replace(spec, ev_charger_kw=charger_kw),
        )["ev_test"]

    return calculate


def test_short_stay_cannot_reach_full_soc_at_11_kw(charging_load, spec):
    # One hour can supply 10.45 kWh to the battery; 25 kWh is needed.
    load = charging_load(start_soc=0.5, departure=5)
    assert load.max() == pytest.approx(0.011)
    assert (load.iloc[1:5] == 0.011).all()
    assert (load.iloc[5:] == 0.0).all()
    assert load.iloc[0] == 0.0
    charged_kwh = load.sum() * 0.25 * 1000.0 * 0.95
    final_soc = 0.5 + charged_kwh / spec.ev_battery_kwh
    assert final_soc == pytest.approx(0.709)
    assert final_soc < 1.0


def test_uses_configured_charger_power_below_11_kw(charging_load):
    load = charging_load(start_soc=0.5, departure=5, charger_kw=7.4)
    assert (load.iloc[1:5] == 0.0074).all()
    assert load.sum() * 0.25 * 1000.0 * 0.95 == pytest.approx(7.03)


def test_sufficient_stay_reaches_target_without_overcharging(charging_load):
    # A 5 kWh deficit needs one full interval and a partial interval.
    load = charging_load(start_soc=0.9, departure=10)
    assert load.sum() * 0.25 * 1000.0 * 0.95 == pytest.approx(5.0)
    assert load.iloc[1] == pytest.approx(0.011)
    assert 0.0 < load.iloc[2] < 0.011
    assert (load.iloc[3:] == 0.0).all()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="ev_requests assumes every departure is full instead of carrying SOC forward",
)
def test_requests_carry_unmet_soc_into_next_trip(index, spec, monkeypatch):
    # Remove xfail when SOC is carried between charging sessions.
    at_home = pd.Series([1, 0, 0, 1, 0, 0] + [1] * (len(index) - 6), index=index)
    trip_mw = (1 - at_home) * 0.0011
    occupancy = at_home.astype(float)
    spec = replace(spec, ev_charger_kw=0.3)
    monkeypatch.setattr(
        scenarios.gc_profiles, "create_e_car",
        lambda **kwargs: (trip_mw, 0.011, at_home),
    )
    requests = scenarios.ev_requests("test", "bus_1", occupancy, spec)

    assert len(requests) == 2
    first_trip_kwh = 1.1 * 0.5
    first_charge_kwh = spec.ev_charger_kw * 0.25 * 0.95
    second_trip_kwh = 1.1 * 0.5
    expected_soc = 1.0 - (
        first_trip_kwh - first_charge_kwh + second_trip_kwh
    ) / spec.ev_battery_kwh
    assert requests.iloc[1]["StartSOC"] == pytest.approx(expected_soc)


@pytest.mark.parametrize(
    "home, driving_kw, charger_kw, expected",
    [
        ([1, 0, 0, 1], [0, 1, 1, 0], 2, 2),
        # Away intervals on opposite ends of the horizon form one cyclic trip.
        ([0, 1, 0], [1, 0, 1], 2, 2),
        # A short home stop supplies 1 kWh between two 2 kWh trips.
        ([0, 1, 0, 1, 1, 1], [2, 0, 2, 0, 0, 0], 1, 3),
        ([1, 1], [0, 0], 11, 0),
    ],
)
def test_minimum_battery_includes_short_stops_and_cyclic_boundary(
    home, driving_kw, charger_kw, expected,
):
    timestamps = pd.date_range("2013-01-01", periods=len(home), freq="h")
    trip = pd.Series(driving_kw, index=timestamps) / 1000.0
    availability = pd.Series(home, index=timestamps)
    assert scenarios.minimum_ev_battery_kwh(
        trip, availability, charger_kw, efficiency=1,
    ) == pytest.approx(expected)


def test_no_battery_can_fix_insufficient_charging_energy():
    timestamps = pd.date_range("2013-01-01", periods=4, freq="h")
    trip = pd.Series([0, 0.001, 0.001, 0.001], index=timestamps)
    home = pd.Series([1, 0, 0, 0], index=timestamps)
    with pytest.raises(ValueError, match="No finite battery.*3.00 kWh.*0.95 kWh"):
        scenarios.minimum_ev_battery_kwh(trip, home, charger_kw=1)


def test_scenario_build_rejects_undersized_battery(spec, monkeypatch):
    timestamps = pd.date_range("2013-01-01", periods=4, freq="h")
    home = pd.Series([1, 0, 0, 1], index=timestamps)
    trip = pd.Series([0, 0.001, 0.001, 0], index=timestamps)
    monkeypatch.setattr(scenarios.gc_profiles, "create_e_car",
                        lambda **kwargs: (trip, 0.011, home))
    with pytest.raises(ValueError, match="EV test at bus bus_1.*at least 2.00 kWh"):
        scenarios.ev_requests(
            "test", "bus_1", home, replace(spec, ev_battery_kwh=1.5),
        )
    requests = scenarios.ev_requests(
        "test", "bus_1", home, replace(spec, ev_battery_kwh=2.0),
    )
    assert len(requests) == 1


def test_scenario_build_rejects_impossible_charging_even_with_large_battery(spec, monkeypatch):
    timestamps = pd.date_range("2013-01-01", periods=4, freq="h")
    home = pd.Series([1, 0, 0, 0], index=timestamps)
    trip = pd.Series([0, 0.001, 0.001, 0.001], index=timestamps)
    monkeypatch.setattr(scenarios.gc_profiles, "create_e_car",
                        lambda **kwargs: (trip, 0.011, home))
    with pytest.raises(ValueError, match="EV test.*No finite battery"):
        scenarios.ev_requests(
            "test", "bus_1", home,
            replace(spec, ev_battery_kwh=1000, ev_charger_kw=1),
        )
