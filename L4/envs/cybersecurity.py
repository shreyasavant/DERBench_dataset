"""
cybersecurity_L4_001 — autonomous false-data-injection (FDI) detection on a real
DER inverter-telemetry stream.

Grounding
---------
Follows the agentic inspect -> select/tune -> commit -> verify loop shape of
Grid-Agent (arXiv:2508.05702, "Grid-Agent: An LLM-Powered Multi-Agent System
for Power Grid Control") — loop shape ONLY: Grid-Agent detects and remediates
grid violations (voltage/thermal) with power-flow tools and contains no FDI or
telemetry-anomaly-detection task. The FDI substance here is grounded in the
classical false-data-injection literature (Liu, Ning & Reiter 2011, ACM TISSEC
— see the task JSON's source_platforms) and standard detectors: an autonomous
agent inspects a telemetry stream, SELECTS and TUNES an anomaly detector,
screens the data, and commits a final set of flagged (attack) samples WITHOUT
excessive false positives — because a false positive here would trigger
unnecessary, harmful curtailment of a healthy inverter.

What is real vs simplified
--------------------------
REAL: the dataset — `L3/data/cybersecurity_inverter_telemetry_1000.csv`, 1000
5-minute samples of a DER inverter's `P_kw, Q_kvar, V_pu, f_hz` with a binary
`label` (50 stealthy FDI-attack samples, 950 normal; deterministic, seed=42).
The detectors are real and run on the actual data:
  - `zscore`: standard z-score on P/Q/V/f with an agent-tunable threshold; a
    sample is flagged if |z| exceeds the threshold on ANY channel. Lowering the
    threshold trades precision for recall; the F1-optimal threshold (~2.9) beats
    the naive |z|>3 reference.
  - `isolation_forest`: a real `sklearn.ensemble.IsolationForest`
    (contamination tunable, n_estimators=200, random_state=seed) on [P,Q,V,f].
  - `physical_bounds`: a real physics rule — flag samples whose frequency leaves
    [f_lo,f_hi] Hz or whose voltage leaves [v_lo,v_hi] pu.
The score is the real F1 of the agent's committed flagged set vs the real
`label` column. The reference benchmark in
`L3/data/cybersecurity_reference_metrics.json` is RECOMPUTED in-env from the data at
reset: the naive |z|>3 detector recomputes to F1≈0.535 (tp=19/fp=2/fn=31) on the
shipped data, while the published file value is F1=0.548 (tp=20/fp=3/fn=30, a
rounded snapshot). The two agree within the ±0.02 provenance tolerance, which
locks the dataset against drift.

SIMPLIFIED / honest caveats:
  - This is a SINGLE inverter telemetry stream, not a full network state-
    estimation / chi-squared bad-data problem (the setting of the classical FDI
    literature). We model the per-device anomaly-detection sub-problem because
    that is the genuine bespoke artifact in this repo.
  - The attacks are deliberately STEALTHY. Empirically, ~20 of the 50 attacks are
    cleanly separable (they sit in the tail of the Q_kvar distribution and a real
    detector flags them at precision ~1.0), but the other ~30 lie inside the
    normal operating envelope and CANNOT be recovered by any classical detector
    without incurring false positives. The achievable F1 CEILING on this data is
    therefore ~0.55–0.58 — the F1-optimal z-score reaches F1≈0.575, IsolationForest
    peaks ≈0.51 over the swept contamination grid (0.44 at the L3 task's c=0.05
    point). The scoring band is anchored to that real ceiling (see
    `_evaluate_impl`): the score measures "selected and tuned a competent detector
    to match-or-beat the reference," NOT "achieved perfect detection," which is
    impossible against stealthy FDI by design.
"""
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from derbench_env import DERBenchEnv

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "L3", "data", "cybersecurity_inverter_telemetry_1000.csv")
_REF = os.path.join(os.path.dirname(__file__), "..", "..",
                    "L3", "data", "cybersecurity_reference_metrics.json")

_FEATURES = ("P_kw", "Q_kvar", "V_pu", "f_hz")


@DERBenchEnv.register("cybersecurity_L4_001")
class FDITelemetryEnv(DERBenchEnv):
    """FDI-attack detection on a real DER inverter telemetry stream."""

    MAX_STEPS = 15

    # Scoring band, anchored to the REAL achievable ceiling on this dataset.
    # F1 at/below the floor -> 0; F1 at/above the ceiling -> 1.0. The ceiling is
    # DERIVED at reset by sweeping the real detectors (self._f1_ceil ~0.575, the
    # F1-optimal z-score), so it stays self-maintaining if the data changes and
    # never compresses the band by sitting below the true optimum. The naive
    # |z|>3 reference (~0.535) therefore maps meaningfully below 1.0, so tuning
    # is rewarded. flag-everything (F1~0.095) and flag-nothing (F1=0) map to 0.
    # _F1_FLOOR is a deliberate design choice: flag-everything must map to ~0.
    # See module docstring for why ~0.55-0.58 is the real ceiling against
    # stealthy FDI.
    _F1_FLOOR = 0.10

    # physical-bounds detector defaults (real DER operating envelope)
    _F_LO, _F_HI = 59.5, 60.5
    _V_LO, _V_HI = 0.90, 1.10

    # ── data + reference (all real, deterministic) ────────────────────────────

    def _load(self):
        df = pd.read_csv(_DATA)
        self._df = df
        self._X = df[list(_FEATURES)].to_numpy(float)
        self._y = df["label"].to_numpy(int)           # HIDDEN ground truth
        self._n = len(df)
        self._n_attack = int(self._y.sum())
        self._n_normal = int(self._n - self._n_attack)
        self._mu = self._X.mean(axis=0)
        self._sd = self._X.std(axis=0)
        self._sd[self._sd == 0] = 1.0
        self._Z = np.abs((self._X - self._mu) / self._sd)

        # Derive the F1 ceiling from the REAL achievable optimum on this data
        # rather than hardcoding it, so the scoring band's upper anchor is
        # self-maintaining and never sits below the true optimum (which would
        # compress the band). Sweep both real detectors; the z-score optimum
        # (~0.575 @ threshold ~2.9) dominates, IsolationForest peaks ~0.514.
        z_best = max(self._prf(self._zscore_flag(t))["f1"]
                     for t in np.arange(1.0, 5.0, 0.05))
        iso_best = max(self._prf(self._isoforest_flag(c))["f1"]
                       for c in (0.02, 0.03, 0.05, 0.08))
        self._f1_ceil = max(z_best, iso_best)   # ~0.575 on the shipped data

        # recompute the published reference and lock provenance. The dataset is
        # seed-fixed and the recomputed F1 is bit-stable (0.5352), only 0.013
        # below the published-file value (0.548); the published file rounds the
        # confusion counts (tp=20/fp=3/fn=30) whereas the exact recomputation is
        # tp=19/fp=2/fn=31. A ±0.02 tolerance is tight enough to be a real lock
        # (it forbids drift to ~0.44) yet absorbs that documented rounding gap;
        # the old ±0.10 was too loose to lock anything.
        ref = self._zscore_flag(3.0)
        self._ref_naive = self._prf(ref)
        with open(_REF) as fh:
            published = json.load(fh)
        tol = 0.02
        assert abs(self._ref_naive["f1"] - published["zscore"]["f1"]) <= tol, (
            f"recomputed naive z-score F1 {self._ref_naive['f1']:.4f} "
            f"deviates from reference {published['zscore']['f1']} by more than "
            f"the ±{tol} provenance tolerance")
        assert self._n_attack == published["n_attack"]
        assert self._n_normal == published["n_normal"]
        self._published = published

    # ── detectors (real; return a boolean flag mask) ──────────────────────────

    def _zscore_flag(self, threshold: float) -> np.ndarray:
        return (self._Z > float(threshold)).any(axis=1)

    def _isoforest_flag(self, contamination: float) -> np.ndarray:
        c = float(min(max(contamination, 0.001), 0.5))
        clf = IsolationForest(contamination=c, n_estimators=200,
                              random_state=self.seed)
        return clf.fit_predict(self._X) == -1

    def _physical_bounds_flag(self, f_lo, f_hi, v_lo, v_hi) -> np.ndarray:
        # NOTE: on the shipped data this flags ~0 samples — every sample stays
        # inside the normal frequency/voltage envelope (the FDI attacks are
        # stealthy and live in the Q_kvar tail, not as bound violations). That
        # is expected, not a bug: it is the honest result of a physics rule
        # against this stealthy dataset.
        f = self._df["f_hz"].to_numpy(float)
        v = self._df["V_pu"].to_numpy(float)
        return (f < f_lo) | (f > f_hi) | (v < v_lo) | (v > v_hi)

    def _detector_mask(self, method: str, params: dict) -> np.ndarray:
        params = params or {}
        if method == "zscore":
            return self._zscore_flag(params.get("threshold", 3.0))
        if method == "isolation_forest":
            return self._isoforest_flag(params.get("contamination", 0.05))
        if method == "physical_bounds":
            return self._physical_bounds_flag(
                params.get("f_lo", self._F_LO), params.get("f_hi", self._F_HI),
                params.get("v_lo", self._V_LO), params.get("v_hi", self._V_HI))
        raise ValueError(
            f"unknown method {method!r}; choose 'zscore', 'isolation_forest', "
            f"or 'physical_bounds'")

    # ── metrics ───────────────────────────────────────────────────────────────

    def _prf(self, mask: np.ndarray) -> dict:
        mask = np.asarray(mask, bool)
        tp = int(np.sum(mask & (self._y == 1)))
        fp = int(np.sum(mask & (self._y == 0)))
        fn = int(np.sum((~mask) & (self._y == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {"precision": round(prec, 4), "recall": round(rec, 4),
                "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn,
                "n_flagged": int(mask.sum())}

    def reference_metrics(self) -> dict:
        """The recomputed (real) detector reference, for tests/provenance."""
        return {"zscore_naive": self._ref_naive,
                "n_attack": self._n_attack, "n_normal": self._n_normal}

    # ── env lifecycle ─────────────────────────────────────────────────────────

    def _reset_impl(self) -> dict:
        self._load()
        self._flagged: set[int] = set()      # the agent's accumulating final set
        self._detectors_run: list[str] = []
        return {
            "scenario": (
                f"A DER inverter telemetry stream of {self._n} 5-minute samples "
                f"(P_kw, Q_kvar, V_pu, f_hz) contains stealthy false-data-injection "
                "(FDI) attacks. Flag the attack samples WITHOUT excessive false "
                "positives — a false positive would trigger harmful, unnecessary "
                "curtailment of a healthy inverter."),
            "n_samples": self._n,
            "features": list(_FEATURES),
            "available_methods": ["zscore", "isolation_forest", "physical_bounds"],
            "note": (
                "Select and tune a real anomaly detector, then COMMIT a final set "
                "of flagged sample indices (via submit_detector or "
                "flag_attack_samples) and declare_done. The ground-truth labels "
                "are hidden; you are scored on F1 against them. Flagging nothing, "
                "flagging everything, or an immediate declare_done scores low; "
                "matching-or-beating the classical reference scores high. The "
                "attacks are stealthy, so even the best classical detector caps "
                "around F1 ~0.55–0.58 — you are scored relative to that ceiling."),
        }

    # ── tools ─────────────────────────────────────────────────────────────────

    def _tools(self) -> list[dict]:
        return [
            {"name": "get_statistics",
             "description": "Real summary statistics (count/mean/std/min/max and "
                            "quartiles) of P_kw, Q_kvar, V_pu, f_hz across the "
                            "whole stream. Does NOT reveal labels.",
             "parameters": {}},
            {"name": "get_telemetry_window",
             "description": "Return the raw telemetry rows (P_kw, Q_kvar, V_pu, "
                            "f_hz, timestamp) for n samples starting at index "
                            "'start'. Labels are NOT included.",
             "parameters": {"start": {"type": "integer"}, "n": {"type": "integer"}}},
            {"name": "inspect_sample",
             "description": "Raw telemetry for a single sample index, with its "
                            "per-channel z-scores. No label.",
             "parameters": {"index": {"type": "integer"}}},
            {"name": "run_detector",
             "description": "Run a real detector on the full stream and return its "
                            "flagged COUNT and flagged indices (NOT the true "
                            "labels). Methods + params: zscore{threshold:float}, "
                            "isolation_forest{contamination:float}, "
                            "physical_bounds{f_lo,f_hi,v_lo,v_hi}. Tuning matters: "
                            "different methods/thresholds flag different sets.",
             "parameters": {"method": {"type": "string"}, "params": {"type": "object"}}},
            {"name": "flag_attack_samples",
             "description": "Add explicit sample indices to your accumulating final "
                            "flagged (attack) set. Returns the current set size.",
             "parameters": {"indices": {"type": "array"}}},
            {"name": "submit_detector",
             "description": "Run the named detector and SET your final flagged set "
                            "to exactly its flagged indices (replaces any prior "
                            "accumulation). Same method/params as run_detector.",
             "parameters": {"method": {"type": "string"}, "params": {"type": "object"}}},
            {"name": "declare_done",
             "description": "End the investigation. Your current flagged set is "
                            "scored by F1 against the hidden ground-truth labels.",
             "parameters": {}},
        ]

    def _dispatch_tool(self, tool: str, args: dict):
        if tool == "get_statistics":
            desc = self._df[list(_FEATURES)].describe()
            return {feat: {k: round(float(desc.loc[k, feat]), 4)
                           for k in desc.index} for feat in _FEATURES}
        if tool == "get_telemetry_window":
            return self._tool_window(int(args.get("start", 0)),
                                     int(args.get("n", 10)))
        if tool == "inspect_sample":
            return self._tool_inspect(int(args.get("index", 0)))
        if tool == "run_detector":
            return self._tool_run(str(args.get("method", "")),
                                  args.get("params") or {}, commit=False)
        if tool == "submit_detector":
            return self._tool_run(str(args.get("method", "")),
                                  args.get("params") or {}, commit=True)
        if tool == "flag_attack_samples":
            return self._tool_flag(args.get("indices") or [])
        if tool == "declare_done":
            self._done = True
            return {"done": True, "n_flagged": len(self._flagged)}
        raise ValueError(f"Unknown tool: {tool}")

    def _tool_window(self, start: int, n: int):
        start = max(0, min(start, self._n - 1))
        n = max(1, min(int(n), 100))
        rows = self._df.iloc[start:start + n]
        return {"start": start, "n": len(rows),
                "samples": [{"index": int(i),
                             "timestamp": str(r["timestamp"]),
                             "P_kw": float(r["P_kw"]), "Q_kvar": float(r["Q_kvar"]),
                             "V_pu": float(r["V_pu"]), "f_hz": float(r["f_hz"])}
                            for i, r in rows.iterrows()]}

    def _tool_inspect(self, index: int):
        if not (0 <= index < self._n):
            return {"rejected": True, "reason": f"index out of range [0,{self._n})"}
        r = self._df.iloc[index]
        return {"index": index, "timestamp": str(r["timestamp"]),
                "P_kw": float(r["P_kw"]), "Q_kvar": float(r["Q_kvar"]),
                "V_pu": float(r["V_pu"]), "f_hz": float(r["f_hz"]),
                "abs_zscore": {f: round(float(self._Z[index, j]), 3)
                               for j, f in enumerate(_FEATURES)}}

    def _tool_run(self, method: str, params: dict, *, commit: bool):
        try:
            mask = self._detector_mask(method, params)
        except ValueError as e:
            return {"rejected": True, "reason": str(e)}
        self._detectors_run.append(method)
        idx = [int(i) for i in np.where(mask)[0]]
        result = {"method": method, "params": params,
                  "n_flagged": len(idx), "flagged_indices": idx}
        if commit:
            self._flagged = set(idx)
            result["committed"] = True
            result["final_set_size"] = len(self._flagged)
        return result

    def _tool_flag(self, indices):
        added = 0
        for i in indices:
            try:
                ii = int(i)
            except (TypeError, ValueError):
                continue
            if 0 <= ii < self._n and ii not in self._flagged:
                self._flagged.add(ii)
                added += 1
        return {"added": added, "final_set_size": len(self._flagged)}

    # ── scoring ─────────────────────────────────────────────────────────────────

    def _evaluate_impl(self) -> tuple[float, dict]:
        """Deterministic score in [0,1] = F1 of the committed flagged set vs the
        real labels, mapped through a band anchored to the REAL achievable
        ceiling (F1_FLOOR -> 0, self._f1_ceil derived at reset -> 1.0). Because
        stealthy FDI caps a classical detector at F1 ~0.575, the band rewards
        "selected and TUNED a competent detector" (the F1-optimal z-score hits
        the ceiling; the naive |z|>3 reference lands meaningfully below it), not
        unattainable perfect detection."""
        mask = np.zeros(self._n, bool)
        for i in self._flagged:
            mask[i] = True
        m = self._prf(mask)

        span = self._f1_ceil - self._F1_FLOOR
        score = (m["f1"] - self._F1_FLOOR) / span
        score = max(0.0, min(1.0, score))

        return score, {
            "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            "n_flagged": m["n_flagged"], "n_attack": self._n_attack,
            "f1_band": [self._F1_FLOOR, round(self._f1_ceil, 4)],
            "reference_naive_zscore_f1": self._ref_naive["f1"],
            "achievable_ceiling_note": (
                "stealthy FDI: ~20/50 attacks cleanly separable; classical "
                "ceiling F1 ~0.55-0.58 (F1-opt z-score ~0.575)"),
            "detectors_run": self._detectors_run,
        }

    def _task_description(self) -> str:
        return (
            "You are an autonomous grid-security agent. A DER inverter telemetry "
            f"stream of {self._n if hasattr(self, '_n') else 1000} 5-minute samples "
            "(P_kw, Q_kvar, V_pu, f_hz) contains stealthy false-data-injection "
            "(FDI) attacks. Your job is to flag the attack samples WITHOUT "
            "excessive false positives — a false positive would trigger harmful, "
            "unnecessary curtailment of a healthy inverter. Inspect the stream "
            "(get_statistics, get_telemetry_window, inspect_sample), then SELECT "
            "and TUNE a real anomaly detector: 'zscore' (tunable threshold on "
            "P/Q/V/f), 'isolation_forest' (tunable contamination), or "
            "'physical_bounds' (frequency/voltage envelope). Use run_detector to "
            "compare candidates, then COMMIT your final flagged set with "
            "submit_detector (sets the flagged set to a detector's output) or "
            "flag_attack_samples (explicit indices); finally declare_done. The "
            "ground-truth labels are hidden; you are scored on F1 against them, "
            "relative to the best achievable classical detector (the attacks are "
            "stealthy, so even the optimal detector caps around F1 ~0.55–0.58 — "
            "match or beat the |z|>3 reference to score high).")

    def _goal_state(self) -> dict:
        return {"objective": "maximize F1 of flagged set vs hidden FDI labels",
                "n_attack_samples": self._n_attack if hasattr(self, "_n_attack") else 50,
                "f1_full_credit_at_or_above": round(self._f1_ceil, 4)
                    if hasattr(self, "_f1_ceil") else 0.575,
                "f1_zero_credit_at_or_below": self._F1_FLOOR,
                "best_classical_detector": "F1-optimal z-score (threshold ~2.9), F1 ~0.575",
                "stealthy_attack_ceiling": "F1 ~0.55-0.58 (classical detector)"}
