"""
forecasting_L4_001 — autonomous day-ahead PV-generation forecasting episode on a
real NASA-POWER-derived dataset, with physics-based hallucination detection.

Grounding
---------
Follows the LLM-as-forecasting-operator loop AND the hallucination-detection
pattern of EF-LLM (arXiv:2411.00852, "EF-LLM: Energy Forecasting LLM with
AI-assisted Automation, Enhanced Sparse Prediction, Hallucination Detection",
Qiu et al. — an LLM that invokes external forecasting models via function
calling and screens hallucinated outputs; it covers load, PV and wind), plus
the query-underlying-forecasters-and-select pattern of "Large Language
Model-Empowered Interactive Load Forecasting" (arXiv:2505.16577; its
interactive operator dialogue is not reproduced, and it contains no
hallucination content — that grounding is EF-LLM's): an autonomous agent
inspects historical data and a weather forecast, SELECTS among competing
forecast models, screens the chosen forecast for physically-impossible
("hallucinated") values against a domain bound, and only then commits the
forecast. EF-LLM's semantic hallucination detector is replaced here by a
physics-bound check.

The load-forecasting emphasis of arXiv:2505.16577 is transferred to distributed
PV-GENERATION forecasting (within EF-LLM's PV scope), because the only genuine
forecasting artifact in this repo is `L3/data/pv_forecast_train_test.csv` — real
15-minute PV data for a ~500 kW site (NASA-POWER-derived irradiance/weather
mapped to PV output). Fabricating utility-scale load would violate the
no-synthetic-data bar, so the episode is honestly framed as PV forecasting; the
hallucination-detection methodology is identical, only the physical ceiling
changes (clear-sky-implied PV envelope instead of historical peak load).

What is real vs simplified
--------------------------
REAL: the dataset (32 064 rows, 2024-01-01 → 2024-11-30, 96×15-min/day); the
train/test split (everything strictly before the fixed TEST DAY is the training
history); the forecast models — `persistence` (the most recent prior day's
profile), `clearsky_scaling` (a least-squares clear-sky-index fit), and
`gradient_boost` (a real `sklearn.ensemble.GradientBoostingRegressor` on weather
features, fitted deterministically at reset) — each produces a DIFFERENT real
forecast with a different held-out MAPE, so model selection genuinely matters;
the score is the real daytime MAPE of the submitted forecast vs the held-out
real `pv_output_kw` for the test day; the hallucination check is a real physics
bound (negative, > nameplate, or > the clear-sky-implied ceiling for that
interval).

SIMPLIFIED / honest caveats: the "weather forecast" handed to the models for the
test day uses the dataset's ACTUAL test-day weather columns (GHI, clear-sky GHI,
temperature, humidity, wind) as a PERFECT-FORESIGHT proxy — there is no separate
NWP forecast product in the dataset, so an operator's real forecast-input error
is not modeled. Because same-interval GHI is near-deterministic of PV output, the
`gradient_boost` model is consequently very accurate; the agentic difficulty is
therefore SELECTING that model (and rejecting the much-worse persistence /
clear-sky baselines) and screening for hallucinated values, not closing a large
irreducible weather-forecast gap. The physical ceiling uses a 0.65 clear-sky
fraction (the train-set max PV/clear-sky ratio is ~0.55) plus the 500 kW
nameplate cap.
"""
import datetime as dt
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from derbench_env import DERBenchEnv

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "L3", "data", "pv_forecast_train_test.csv")


@DERBenchEnv.register("forecasting_L4_001")
class PVDayAheadForecastEnv(DERBenchEnv):
    """Day-ahead PV forecasting on a real dataset, with hallucination screening."""

    MAX_STEPS = 15

    _CAPACITY_KW = 500.0          # site nameplate (dataset pv_output_kw max)
    _TEST_DAY = dt.date(2024, 11, 15)   # fixed held-out target day
    _N = 96                       # 15-min intervals in the test day
    _DAY_CS_THRESH = 25.0         # clear-sky GHI (W/m^2) above which it is "daytime"
    _MAPE_FLOOR_PCT = 5.0         # MAPE <= this -> full accuracy credit
    _MAPE_CEIL_PCT = 35.0         # MAPE >= this -> zero accuracy credit
    _CS_CEILING_FRACTION = 0.65   # physical ceiling = frac * clear-sky GHI (kW)
    _CEILING_TOL_KW = 2.0         # absolute slack on the ceiling (0.4% of cap);
                                  # sub-2 kW night noise is not a "hallucination"
    _MODELS = ("persistence", "clearsky_scaling", "gradient_boost")

    # ── data + model fitting (all real, deterministic) ────────────────────────

    def _feature_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        h = frame["timestamp"].dt.hour + frame["timestamp"].dt.minute / 60.0
        return np.column_stack([
            frame["ghi_wm2"].to_numpy(float),
            frame["clear_sky_ghi_wm2"].to_numpy(float),
            frame["temperature_c"].to_numpy(float),
            frame["relative_humidity_pct"].to_numpy(float),
            frame["wind_speed_ms"].to_numpy(float),
            np.sin(2 * np.pi * h / 24.0),
            np.cos(2 * np.pi * h / 24.0),
        ])

    def _fit_models(self):
        """Fit every forecaster on the train history and store the test-day
        forecast each one produces (real arrays, length 96)."""
        df = pd.read_csv(_DATA, parse_dates=["timestamp"])
        df["date"] = df["timestamp"].dt.date
        train = df[df["date"] < self._TEST_DAY].reset_index(drop=True)
        test = df[df["date"] == self._TEST_DAY].reset_index(drop=True)
        if len(test) != self._N:
            raise RuntimeError(f"test day {self._TEST_DAY} has {len(test)} rows, "
                               f"expected {self._N}")

        self._actual_kw = test["pv_output_kw"].to_numpy(float)
        self._clearsky_ghi = test["clear_sky_ghi_wm2"].to_numpy(float)
        self._test_weather = test
        self._physical_ceiling = np.minimum(
            self._CAPACITY_KW,
            np.maximum(self._CEILING_TOL_KW,
                       self._CS_CEILING_FRACTION * self._clearsky_ghi))
        self._daytime_mask = self._clearsky_ghi > self._DAY_CS_THRESH

        forecasts: dict[str, np.ndarray] = {}

        # persistence — most recent prior day's profile
        prev_day = self._TEST_DAY - dt.timedelta(days=1)
        prev = df[df["date"] == prev_day]["pv_output_kw"].to_numpy(float)
        forecasts["persistence"] = (prev if len(prev) == self._N
                                    else np.zeros(self._N))

        # clearsky_scaling — least-squares PV ~ k * clear_sky_ghi on daytime train
        tcs = train["clear_sky_ghi_wm2"].to_numpy(float)
        tpv = train["pv_output_kw"].to_numpy(float)
        m = tcs > self._DAY_CS_THRESH
        k = float(np.sum(tpv[m] * tcs[m]) / np.sum(tcs[m] * tcs[m]))
        self._clearsky_k = k
        forecasts["clearsky_scaling"] = np.clip(
            k * self._clearsky_ghi, 0.0, self._CAPACITY_KW)

        # gradient_boost — real sklearn GBR on weather features
        gb = GradientBoostingRegressor(random_state=self.seed, n_estimators=200,
                                       max_depth=3)
        gb.fit(self._feature_matrix(train), tpv)
        forecasts["gradient_boost"] = np.clip(
            gb.predict(self._feature_matrix(test)), 0.0, self._CAPACITY_KW)
        self._gb = gb

        self._forecasts = {m_: np.round(v, 3) for m_, v in forecasts.items()}
        self._model_mape = {m_: self._daytime_mape(v)
                            for m_, v in self._forecasts.items()}

    # ── scoring primitives ────────────────────────────────────────────────────

    def _daytime_mape(self, forecast_kw) -> float:
        """Real daytime MAPE (%) vs held-out actuals, on intervals that are both
        daytime (clear-sky > thresh) and have non-trivial actual generation."""
        fc = np.asarray(forecast_kw, float)
        if fc.shape != self._actual_kw.shape:
            return 100.0
        mask = self._daytime_mask & (self._actual_kw > 5.0)
        if not mask.any():
            return 100.0
        a = self._actual_kw[mask]
        p = fc[mask]
        return float(np.mean(np.abs(p - a) / a) * 100.0)

    def _count_impossible(self, forecast_kw) -> int:
        fc = np.asarray(forecast_kw, float)
        if fc.size == 0:
            return 0
        n = len(fc)
        ceiling = (self._physical_ceiling[:n] if n <= self._N
                   else np.resize(self._physical_ceiling, n))
        return int(np.sum((fc < -1e-6) | (fc > ceiling + 1e-6)))

    # ── env lifecycle ─────────────────────────────────────────────────────────

    def _reset_impl(self) -> dict:
        self._fit_models()
        self._final_forecast: list[float] = []
        self._submitted_model: str | None = None
        self._explanation = ""
        self._models_called: list[str] = []
        self._bounds_checked = False
        return {
            "scenario": "Autonomous day-ahead PV-generation forecasting for a real "
                        f"{self._CAPACITY_KW:.0f} kW site (NASA-POWER-derived "
                        "dataset). Produce a 96-interval (15-min) forecast for the "
                        f"target day {self._TEST_DAY.isoformat()}.",
            "target_date": self._TEST_DAY.isoformat(),
            "intervals": self._N,
            "interval_minutes": 15,
            "capacity_kw": self._CAPACITY_KW,
            "available_models": list(self._MODELS),
            "note": "Models are fitted on the historical record strictly before "
                    "the target day. Submitting an empty/flat/hallucinated "
                    "forecast scores low; selecting the most accurate bounds-clean "
                    "forecast scores high.",
        }

    # ── tools ─────────────────────────────────────────────────────────────────

    def _tools(self) -> list[dict]:
        return [
            {"name": "get_historical_data",
             "description": "Real past PV output (kW) and weather for the most recent "
                            "N days before the target day (15-min resolution).",
             "parameters": {"days": {"type": "integer"}}},
            {"name": "get_weather_forecast",
             "description": "The target day's weather inputs the models use: GHI, "
                            "clear-sky GHI (W/m^2), temperature (C), humidity (%), "
                            "wind (m/s), per 15-min interval.",
             "parameters": {}},
            {"name": "get_clearsky",
             "description": "Clear-sky GHI (W/m^2) for the target day and the implied "
                            "physical PV ceiling (kW) per interval.",
             "parameters": {}},
            {"name": "call_forecast_model",
             "description": "Run a named real forecaster fitted on the training "
                            "history and return its 96-interval PV forecast (kW) for "
                            "the target day plus its daytime MAPE vs held-out actuals. "
                            "Models: 'persistence', 'clearsky_scaling', "
                            "'gradient_boost'. The optional 'features' arg is a NO-OP "
                            "(echoed for transparency only): each model uses its own "
                            "fixed feature set and does not honor a custom selection.",
             "parameters": {"model": {"type": "string"},
                            "features": {"type": "array"}}},
            {"name": "check_physical_bounds",
             "description": "Hallucination screen: count physically-impossible "
                            "intervals (< 0, > capacity, or > the clear-sky-implied "
                            "ceiling) for an explicit forecast_kw array or for a named "
                            "model's forecast.",
             "parameters": {"forecast_kw": {"type": "array"},
                            "model": {"type": "string"}}},
            {"name": "submit_final_forecast",
             "description": "Submit an explicit 96-interval forecast_kw array as the "
                            "final forecast and end the episode.",
             "parameters": {"forecast_kw": {"type": "array"}}},
            {"name": "declare_done",
             "description": "End the episode. Submit either an explicit forecast_kw "
                            "array OR a named model whose fitted forecast becomes the "
                            "final submission.",
             "parameters": {"model": {"type": "string"},
                            "forecast_kw": {"type": "array"},
                            "explanation": {"type": "string"}}},
        ]

    def _dispatch_tool(self, tool: str, args: dict):
        if tool == "get_historical_data":
            return self._tool_history(int(args.get("days", 3)))
        if tool == "get_weather_forecast":
            w = self._test_weather
            return {
                "intervals": self._N,
                "ghi_wm2": [round(x, 1) for x in w["ghi_wm2"]],
                "clear_sky_ghi_wm2": [round(x, 1) for x in w["clear_sky_ghi_wm2"]],
                "temperature_c": [round(x, 1) for x in w["temperature_c"]],
                "relative_humidity_pct": [round(x, 1) for x in w["relative_humidity_pct"]],
                "wind_speed_ms": [round(x, 2) for x in w["wind_speed_ms"]],
                "note": "perfect-foresight proxy: dataset actual weather for the target day",
            }
        if tool == "get_clearsky":
            return {"clear_sky_ghi_wm2": [round(x, 1) for x in self._clearsky_ghi],
                    "physical_ceiling_kw": [round(x, 1) for x in self._physical_ceiling],
                    "capacity_kw": self._CAPACITY_KW}
        if tool == "call_forecast_model":
            return self._tool_call_model(str(args.get("model", "")),
                                         args.get("features"))
        if tool == "check_physical_bounds":
            return self._tool_bounds(args)
        if tool == "submit_final_forecast":
            return self._tool_submit(forecast_kw=args.get("forecast_kw"),
                                     explanation="")
        if tool == "declare_done":
            return self._tool_submit(forecast_kw=args.get("forecast_kw"),
                                     model=args.get("model"),
                                     explanation=str(args.get("explanation", "")))
        raise ValueError(f"Unknown tool: {tool}")

    def _tool_history(self, days: int):
        days = max(1, min(int(days), 14))
        df = pd.read_csv(_DATA, parse_dates=["timestamp"])
        df["date"] = df["timestamp"].dt.date
        start = self._TEST_DAY - dt.timedelta(days=days)
        hist = df[(df["date"] >= start) & (df["date"] < self._TEST_DAY)]
        return {
            "days": days,
            "n_intervals": len(hist),
            "pv_output_kw": [round(x, 1) for x in hist["pv_output_kw"]],
            "ghi_wm2": [round(x, 1) for x in hist["ghi_wm2"]],
            "temperature_c": [round(x, 1) for x in hist["temperature_c"]],
        }

    def _tool_call_model(self, model: str, features):
        if model not in self._forecasts:
            return {"rejected": True, "model": model,
                    "reason": f"unknown model; choose one of {list(self._MODELS)}"}
        self._models_called.append(model)
        fc = self._forecasts[model]
        return {"model": model, "features": features,
                "forecast_kw": [round(float(x), 2) for x in fc],
                "daytime_mape_pct": round(self._model_mape[model], 3),
                "n_intervals": len(fc)}

    def _tool_bounds(self, args: dict):
        self._bounds_checked = True
        model = args.get("model")
        fc = args.get("forecast_kw")
        if fc is None and model in self._forecasts:
            fc = self._forecasts[model]
        if fc is None:
            return {"rejected": True,
                    "reason": "provide forecast_kw or a known model name"}
        fc = np.asarray(fc, float)
        n = len(fc)
        ceiling = (self._physical_ceiling[:n] if n <= self._N
                   else np.resize(self._physical_ceiling, n))
        bad = [int(i) for i in range(n)
               if fc[i] < -1e-6 or fc[i] > ceiling[i] + 1e-6]
        return {"n_impossible": len(bad), "indices": bad[:50],
                "capacity_kw": self._CAPACITY_KW,
                "checked": "model:" + model if model else "explicit_array"}

    def _tool_submit(self, forecast_kw=None, model=None, explanation=""):
        if model is not None and model in self._forecasts and not forecast_kw:
            self._final_forecast = [float(x) for x in self._forecasts[model]]
            self._submitted_model = model
        elif forecast_kw is not None:
            try:
                self._final_forecast = [float(x) for x in forecast_kw]
            except (TypeError, ValueError):
                self._final_forecast = []
            self._submitted_model = None
        else:
            self._final_forecast = []
        self._explanation = explanation
        self._done = True
        return {"status": "submitted",
                "n_intervals": len(self._final_forecast),
                "submitted_model": self._submitted_model}

    # ── scoring ─────────────────────────────────────────────────────────────────

    def _evaluate_impl(self) -> tuple[float, dict]:
        """Deterministic score in [0,1] reflecting real forecast accuracy.

        accuracy (primary): daytime MAPE of the submitted forecast vs held-out
          real PV, linearly mapped MAPE_FLOOR(5%)->1.0 .. MAPE_CEIL(35%)->0.0;
        hallucination penalty: any physically-impossible submitted interval
          sharply reduces the score (multiplicative, scaled by the fraction of
          impossible intervals, floored at 0.2x for any violation) AND hard-caps
          the total score at 0.8 the moment a single impossible value appears —
          the grounding papers treat hallucination mitigation as a primary
          safety objective, so even one hallucinated value must materially hurt.
        A no-forecast / wrong-length / empty submission scores 0.
        """
        fc = self._final_forecast
        if not fc or len(fc) != self._N:
            return 0.0, {"error": "no valid 96-interval forecast submitted",
                         "submitted_length": len(fc),
                         "model_mape_pct": {m: round(v, 3)
                                            for m, v in self._model_mape.items()}}

        mape = self._daytime_mape(fc)
        n_impossible = self._count_impossible(fc)

        # accuracy credit from real MAPE
        span = self._MAPE_CEIL_PCT - self._MAPE_FLOOR_PCT
        accuracy = max(0.0, min(1.0, (self._MAPE_CEIL_PCT - mape) / span))

        # hallucination penalty
        if n_impossible > 0:
            frac_bad = n_impossible / self._N
            penalty = max(0.2, 1.0 - 4.0 * frac_bad)
        else:
            penalty = 1.0

        score = max(0.0, min(1.0, accuracy * penalty))
        # hard cap: ANY physically-impossible value must meaningfully hurt, even
        # one out of 96 (mirrors the isolate_line hard-cap idiom in
        # grid_edge_control.py — a hallucination must never reach the clean ceiling).
        if n_impossible > 0:
            score = min(score, 0.8)
        return score, {
            "submitted_model": self._submitted_model,
            "daytime_mape_pct": round(mape, 3),
            "impossible_values": n_impossible,
            "accuracy_credit": round(accuracy, 4),
            "hallucination_penalty": round(penalty, 4),
            "hallucination_hard_cap_applied": bool(n_impossible > 0),
            "mape_band_pct": [self._MAPE_FLOOR_PCT, self._MAPE_CEIL_PCT],
            "model_mape_pct": {m: round(v, 3) for m, v in self._model_mape.items()},
            "models_called": self._models_called,
            "bounds_checked": self._bounds_checked,
            "n_daytime_intervals": int(self._daytime_mask.sum()),
        }

    def _task_description(self) -> str:
        return (
            "You are an autonomous PV-forecasting operator. Produce a reliable "
            f"day-ahead PV-generation forecast for a {self._CAPACITY_KW:.0f} kW "
            f"site on {self._TEST_DAY.isoformat()} ({self._N} intervals at 15 min). "
            "Three real forecasters are available — 'persistence' (yesterday's "
            "profile), 'clearsky_scaling' (a fitted clear-sky-index model), and "
            "'gradient_boost' (a gradient-boosted regressor on weather features), "
            "all fitted on the historical record before the target day. Inspect the "
            "history, the weather forecast and the clear-sky ceiling; run the "
            "candidate models and SELECT the most accurate one; screen the chosen "
            "forecast for physically-impossible values with check_physical_bounds "
            "(no value may be negative, exceed capacity, or exceed the clear-sky "
            "ceiling); then submit it via declare_done (you may pass a model name to "
            "submit that model's fitted forecast, or an explicit forecast_kw array). "
            "Accuracy is graded by real daytime MAPE against the held-out actuals; "
            "hallucinated values are penalized.")

    def _goal_state(self) -> dict:
        return {"submit_intervals": self._N,
                "minimize": "daytime_mape_pct",
                "mape_full_credit_at_or_below_pct": self._MAPE_FLOOR_PCT,
                "mape_zero_credit_at_or_above_pct": self._MAPE_CEIL_PCT,
                "no_impossible_values": True,
                "physical_ceiling": "min(capacity, 0.65 * clear_sky_ghi) per interval",
                "best_available_model": "gradient_boost"}
