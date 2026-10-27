"""Deployment-owned fidelity requirements; no optimizer or testing callbacks."""
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Fidelity:
    mode: str = "numerical"
    at_least: str | None = None


def resolve_fidelity(config: Mapping) -> Fidelity:
    value = config.get("fidelity", {"mode": "numerical"})
    if not isinstance(value, Mapping):
        raise ValueError("fidelity must be a mapping")
    mode = value.get("mode", "numerical")
    allowed = {"mode"} if mode == "numerical" else {"mode", "at_least"}
    if mode not in ("numerical", "quality") or set(value) - allowed:
        raise ValueError("unsupported fidelity mode or fields")
    baseline = value.get("at_least")
    if mode == "quality" and (not isinstance(baseline, str) or not baseline.strip()):
        raise ValueError("quality fidelity requires a named model baseline")
    return Fidelity(mode, baseline)
