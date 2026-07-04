"""Local workbench experiments for resonator thermal-population checks."""

__all__ = ["HahnEcho", "HahnEchoHandler"]


def __getattr__(name: str):
    if name in __all__:
        from measure_resonator_thermal_population.hahn_echo import HahnEcho, HahnEchoHandler

        return {"HahnEcho": HahnEcho, "HahnEchoHandler": HahnEchoHandler}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
