"""
Apple Silicon thermal and power heuristics for experiment scheduling.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass


BATTERY_RE = re.compile(r"(?P<pct>\d+)%;\s*(?P<status>[^;]+);")


@dataclass
class RuntimeState:
    power_source: str
    battery_percent: int | None
    charging: bool
    thermal_warning: bool
    performance_warning: bool
    cpu_power_warning: bool


@dataclass
class RuntimeProfile:
    mode: str
    sample_duration_seconds: int
    allow_full_run: bool
    suggested_mps_autocast: str
    notes: str
    state: RuntimeState


def _run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def probe_runtime_state() -> RuntimeState:
    batt_text = _run(["pmset", "-g", "batt"])
    therm_text = _run(["pmset", "-g", "therm"])

    power_source = "Unknown"
    if "AC Power" in batt_text:
        power_source = "AC Power"
    elif "Battery Power" in batt_text:
        power_source = "Battery Power"

    battery_percent = None
    charging = False
    match = BATTERY_RE.search(batt_text)
    if match:
        battery_percent = int(match.group("pct"))
        charging = "charg" in match.group("status").lower()

    thermal_warning = "No thermal warning level has been recorded" not in therm_text
    performance_warning = "No performance warning level has been recorded" not in therm_text
    cpu_power_warning = "No CPU power status has been recorded" not in therm_text

    return RuntimeState(
        power_source=power_source,
        battery_percent=battery_percent,
        charging=charging,
        thermal_warning=thermal_warning,
        performance_warning=performance_warning,
        cpu_power_warning=cpu_power_warning,
    )


def recommend_runtime_profile(state: RuntimeState | None = None) -> RuntimeProfile:
    state = state or probe_runtime_state()

    if state.thermal_warning or state.performance_warning or state.cpu_power_warning:
        return RuntimeProfile(
            mode="cooldown",
            sample_duration_seconds=15,
            allow_full_run=False,
            suggested_mps_autocast="off",
            notes="Thermal or performance warnings detected; stay conservative.",
            state=state,
        )

    if state.power_source != "AC Power":
        return RuntimeProfile(
            mode="battery",
            sample_duration_seconds=20,
            allow_full_run=False,
            suggested_mps_autocast="off",
            notes="Battery power detected; prefer short samples and avoid long full runs.",
            state=state,
        )

    if state.battery_percent is not None and state.battery_percent < 40:
        return RuntimeProfile(
            mode="balanced",
            sample_duration_seconds=25,
            allow_full_run=True,
            suggested_mps_autocast="off",
            notes="On AC but battery is still low; take the balanced path.",
            state=state,
        )

    return RuntimeProfile(
        mode="aggressive",
        sample_duration_seconds=30,
        allow_full_run=True,
        suggested_mps_autocast="off",
        notes="AC power with no thermal warnings; full benchmarks are reasonable.",
        state=state,
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Probe Apple Silicon runtime conditions.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    profile = recommend_runtime_profile()
    if args.json:
        print(json.dumps(asdict(profile), indent=2))
        return 0

    print(f"mode: {profile.mode}")
    print(f"sample_duration_seconds: {profile.sample_duration_seconds}")
    print(f"allow_full_run: {profile.allow_full_run}")
    print(f"suggested_mps_autocast: {profile.suggested_mps_autocast}")
    print(f"notes: {profile.notes}")
    print(f"power_source: {profile.state.power_source}")
    print(f"battery_percent: {profile.state.battery_percent}")
    print(f"charging: {profile.state.charging}")
    print(f"thermal_warning: {profile.state.thermal_warning}")
    print(f"performance_warning: {profile.state.performance_warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
