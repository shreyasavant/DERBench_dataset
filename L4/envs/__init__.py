"""Importing this package registers every L4 environment with DERBenchEnv."""
_CATEGORY_MODULES = [
    "grid_edge_control", "vpp_scheduling", "forecasting",
    "p2p_trading", "cybersecurity", "state_estimation", "ev_v2g",
    "fault_protection", "hosting_capacity",
]

def load_all() -> None:
    import importlib
    for m in _CATEGORY_MODULES:
        importlib.import_module(f"envs.{m}")

load_all()
