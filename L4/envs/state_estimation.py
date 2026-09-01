"""
state_estimation_L4_001 — autonomous home-energy NILM appliance-health agent on a
deterministically-constructed 7-day 1-minute aggregate smart-meter stream.

Grounding
---------
Reproduces the autonomous NILM-plus-maintenance loop described in the agentic
maintenance / NILM literature:
  - CrespoMarquez2026 ("Agentic Maintenance") — an autonomous agent that
    diagnoses appliance/asset health from sensor data and ROUTES the right
    maintenance action (schedule a hardware visit vs. raise an alert vs. log and
    monitor a usage change), penalising disproportionate / harmful interventions.
  - DaSilva2026 ("A Systematic Review of NILM") and LLM4NILM_2025 ("LLMs for
    Non-Intrusive Load Monitoring") — disaggregating a single aggregate meter
    into per-appliance power, then reasoning over per-appliance deviations from a
    historical baseline to detect faults.

The agent receives 7 days of 1-minute aggregate power (10 080 timesteps), must
disaggregate the appliances, detect which appliances behave anomalously in the
Day 6–7 window versus the Day 1–5 baseline, diagnose the cause, and take the
appropriate maintenance / demand-response action — without false-alarming on the
appliances that are still healthy.

What is real vs simplified
--------------------------
REAL: the disaggregation algorithm. `disaggregate()` runs an honest
combinatorial-optimisation NILM on whatever aggregate array it is handed: per
timestep it enumerates the cartesian product of appliance state levels (24
combinations) and picks the combination whose summed power best matches the
observed aggregate — the Hart-1992 "Combinatorial Optimisation" baseline as in
NILMTK (with the state levels given a priori rather than learned from training
data, a disclosed shortcut on this constructed dataset). It does NOT return the
stored truth —
it recovers per-appliance power from the aggregate signal, and the test
`test_disaggregate_recovers_appliance_levels` checks the recovered levels against
the constructed truth. The baselines, anomaly F1 and action scoring are computed
deterministically against the exact per-appliance ground truth.

SIMPLIFIED / honest caveats: real sub-metered ground-truth NILM data (REDD /
UK-DALE / Pecan Street) is licence-encumbered and not shipped in this repo, so
the 7-day aggregate is DETERMINISTICALLY CONSTRUCTED by summing explicit physical
appliance models (the exact signatures specified in the task prompt) — the same
"build the physics, get exact labels" philosophy used across the DER-Bench
constructed datasets.
This yields exact per-appliance labels, which is the standard NILM evaluation
substitute when real sub-metering is unavailable. A tiny seeded Gaussian
measurement noise (sigma = 2 W) is added to the aggregate only; it is far below
the smallest appliance level (150 W) so it never confuses the disaggregator. The
appliance signatures are distinct enough (150/220, 300, 1800, 3500, 7200 W) that
a real combinatorial NILM separates them cleanly — the genuine agentic difficulty
is the diagnosis + action routing, not the signal separation.
"""
import re
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from derbench_env import DERBenchEnv

# minutes
_MIN_PER_DAY = 1440
_N_DAYS = 7
_N = _MIN_PER_DAY * _N_DAYS  # 10080

# the appliances + their nameplate power levels (W) used by the NILM matcher
_BASE_W = 300.0
_FRIDGE_NORMAL_W = 150.0
_FRIDGE_DEGRADED_W = 220.0
_HVAC_W = 3500.0
_EV_W = 7200.0
_DISHWASHER_W = 1800.0

# appliances we score / report on
_APPLIANCES = ("refrigerator", "hvac", "ev_charger", "dishwasher", "base_load")
# the exact ground-truth anomaly partition (Day 6-7 vs Day 1-5)
_TRUE_ANOMALOUS = {
    "refrigerator": "degrading_hardware",   # higher draw, faster cycling -> fault
    "hvac": "stuck_on_extended_runtime",    # continuous runtime -> fault/alert
    "ev_charger": "absent_usage_change",    # not charging -> usage, NOT a fault
}
_TRUE_NORMAL = {"dishwasher", "base_load"}


def _minute_of_day(t: np.ndarray) -> np.ndarray:
    return t % _MIN_PER_DAY


@DERBenchEnv.register("state_estimation_L4_001")
class NILMApplianceHealthEnv(DERBenchEnv):
    """7-day NILM appliance-health monitoring on a constructed 1-min aggregate."""

    MAX_STEPS = 25

    # ── deterministic signal construction (exact ground truth) ────────────────

    @staticmethod
    def _fridge_signal(day_idx: np.ndarray, mod: np.ndarray, anomalous: bool):
        """Refrigerator: duty-cycle square wave. Normal 150 W (35 on / 25 off);
        Day 6-7 degraded 220 W (40 on / 15 off)."""
        out = np.zeros_like(mod, float)
        normal = ~anomalous
        # normal portion
        on_n, off_n, p_n = 35, 25, _FRIDGE_NORMAL_W
        period_n = on_n + off_n
        phase_n = mod % period_n
        out[normal & (phase_n < on_n)] = p_n
        # degraded portion
        on_a, off_a, p_a = 40, 15, _FRIDGE_DEGRADED_W
        period_a = on_a + off_a
        phase_a = mod % period_a
        out[anomalous & (phase_a < on_a)] = p_a
        return out

    @staticmethod
    def _hvac_signal(mod: np.ndarray, anomalous: bool):
        """HVAC 3500 W. Normal: 06:00-09:00 and 17:00-22:00. Day 6-7: continuous
        06:00-23:00 (stuck on / extended runtime)."""
        out = np.zeros_like(mod, float)
        h = mod / 60.0
        normal_on = ((h >= 6) & (h < 9)) | ((h >= 17) & (h < 22))
        ext_on = (h >= 6) & (h < 23)
        on = np.where(anomalous, ext_on, normal_on)
        out[on] = _HVAC_W
        return out

    @staticmethod
    def _ev_signal(mod: np.ndarray, absent: bool):
        """EV charger 7200 W on 22:00-02:00 (wraps midnight). Day 7: absent."""
        out = np.zeros_like(mod, float)
        h = mod / 60.0
        on = (h >= 22) | (h < 2)
        if not absent:
            out[on] = _EV_W
        return out

    @staticmethod
    def _dishwasher_signal(mod: np.ndarray):
        """Dishwasher 1800 W, ~19:30 for 90 min daily (normal on all days)."""
        out = np.zeros_like(mod, float)
        start = 19 * 60 + 30
        on = (mod >= start) & (mod < start + 90)
        out[on] = _DISHWASHER_W
        return out

    def _build_signals(self):
        """Construct exact per-appliance W signals over the full 7-day horizon."""
        t = np.arange(_N)
        day = t // _MIN_PER_DAY            # 0..6
        mod = _minute_of_day(t)            # 0..1439
        anomaly_day = day >= 5             # Day 6 (idx5) & Day 7 (idx6)

        # base load: constant 300 W everywhere
        base = np.full(_N, _BASE_W)

        # fridge: anomalous on Day 6-7
        fridge = self._fridge_signal(day, mod, anomaly_day)

        # hvac: per-day select normal vs extended
        hvac = np.zeros(_N, float)
        for d in range(_N_DAYS):
            sel = day == d
            hvac[sel] = self._hvac_signal(mod[sel], anomalous=(d >= 5))

        # EV: absent specifically on Day 7 (the prompt says "no charging on Day 7").
        ev = np.zeros(_N, float)
        for d in range(_N_DAYS):
            sel = day == d
            ev[sel] = self._ev_signal(mod[sel], absent=(d == 6))

        # dishwasher: normal every day
        dish = self._dishwasher_signal(mod)

        self._signals = {
            "base_load": base,
            "refrigerator": fridge,
            "hvac": hvac,
            "ev_charger": ev,
            "dishwasher": dish,
        }
        clean_total = sum(self._signals.values())
        # tiny seeded measurement noise on the AGGREGATE only (<< 150 W); never
        # large enough to confuse the level-matching disaggregator.
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0.0, 2.0, _N)
        self._aggregate = clean_total + noise
        self._day_index = day
        self._minute_index = mod

    # ── real disaggregation (combinatorial / power-level matching) ────────────

    # Combinatorial-Optimisation (CO) state space: every appliance is modelled as
    # a small set of admissible power levels; the disaggregator enumerates the
    # cartesian product of states and, per timestep, picks the combination whose
    # summed power is closest to the observed aggregate. This is the classic
    # NILMTK "Combinatorial Optimisation" baseline (Hart 1992 / Batra 2014). The
    # fridge carries two ON levels (150 normal / 220 degraded) so the matcher
    # recovers the degraded level when the signal supports it.
    _STATE_LEVELS = {
        "base_load": (_BASE_W,),                      # always on
        "refrigerator": (0.0, _FRIDGE_NORMAL_W, _FRIDGE_DEGRADED_W),
        "hvac": (0.0, _HVAC_W),
        "ev_charger": (0.0, _EV_W),
        "dishwasher": (0.0, _DISHWASHER_W),
    }

    @classmethod
    def _build_state_table(cls):
        """Precompute the (combination -> total power) table for CO matching."""
        import itertools
        names = list(cls._STATE_LEVELS)
        combos = list(itertools.product(*[cls._STATE_LEVELS[n] for n in names]))
        totals = np.array([sum(c) for c in combos], float)
        return names, np.array(combos, float), totals

    def _disaggregate(self, aggregate) -> dict[str, np.ndarray]:
        """REAL Combinatorial-Optimisation NILM: enumerate the appliance on/off
        (and fridge-level) state combinations and, for each timestep, assign the
        combination whose total power best matches the observed aggregate
        (minimum |sum - aggregate|). Runs on the PASSED aggregate array and
        recovers per-appliance power from the signal — it does NOT read back the
        stored truth. The full 5-appliance product is 1*3*2*2*2 = 24 states, so
        the brute force is trivially tractable (24 combinations per timestep)."""
        agg = np.asarray(aggregate, float)
        names, combos, totals = self._build_state_table()
        # per timestep, index of the closest combination total
        # |agg[:,None] - totals[None,:]| argmin
        diff = np.abs(agg[:, None] - totals[None, :])
        best = np.argmin(diff, axis=1)            # (n,)
        chosen = combos[best]                     # (n, n_appliances)
        est = {nm: chosen[:, j].astype(float)
               for j, nm in enumerate(names)}
        # reorder to canonical appliance order
        return {a: est[a] for a in _APPLIANCES}

    # ── baselines (real Day 1-5 statistics) ───────────────────────────────────

    def _baseline_stats(self, appliance: str) -> dict:
        """Real Day 1-5 statistics for an appliance, computed from the TRUE
        constructed signal (mean daily kWh, std, typical on-hours/day)."""
        sig = self._signals[appliance]
        day = self._day_index
        kwh_per_day, on_min_per_day = [], []
        for d in range(5):                       # Day 1-5 baseline window
            sel = day == d
            w = sig[sel]
            kwh_per_day.append(float(w.sum()) / 60.0 / 1000.0)   # W*min -> kWh
            on_min_per_day.append(float((w > 1.0).sum()))
        kwh = np.array(kwh_per_day)
        return {
            "mean_daily_kwh": round(float(kwh.mean()), 3),
            "std_daily_kwh": round(float(kwh.std()), 4),
            "typical_on_hours": round(float(np.mean(on_min_per_day)) / 60.0, 2),
        }

    def _day67_window(self):
        return self._aggregate[self._day_index >= 5]

    # ── env lifecycle ──────────────────────────────────────────────────────────

    def _reset_impl(self) -> dict:
        self._build_signals()
        # accumulating agent record
        self._flags: dict[str, set[str]] = {}   # appliance -> set of action kinds
        self._maintenance: list[dict] = []
        self._alerts: list[dict] = []
        self._dr_signals: list[dict] = []
        self._logs: list[dict] = []
        self._disaggregated = False
        self._baselines_read: set[str] = set()
        return {
            "scenario": (
                "Autonomous home energy management agent. You have 7 days of "
                f"aggregate smart-meter power at 1-minute resolution ({_N} "
                "timesteps). Day 1-5 is the normal baseline; Day 6-7 is the "
                "monitoring window. Disaggregate the appliances, detect which "
                "appliances are anomalous in Day 6-7 vs the Day 1-5 baseline, "
                "diagnose the cause, and take the appropriate maintenance / "
                "demand-response action WITHOUT false-alarming on healthy "
                "appliances."),
            "n_timesteps": _N,
            "resolution_minutes": 1,
            "appliances": list(_APPLIANCES),
            "baseline_days": [1, 2, 3, 4, 5],
            "monitoring_days": [6, 7],
            "tariff_usd_per_kwh": 0.15,
            "note": (
                "Call disaggregate on the Day 6-7 window, compare to get_baseline, "
                "then register findings via schedule_maintenance / send_alert / "
                "log_decision (each naming the appliance). A no-op or immediate "
                "declare_done scores low; correctly diagnosing the truly anomalous "
                "appliances with appropriate actions and NO false alarms scores "
                "high. Curtailing or scheduling maintenance for a healthy "
                "appliance is harmful and penalised."),
        }

    # ── tools ─────────────────────────────────────────────────────────────────

    def _tools(self) -> list[dict]:
        return [
            {"name": "disaggregate",
             "description": "Run the NILM disaggregator on an aggregate-power "
                            "array and return per-appliance power estimates (W, "
                            "1-min). Pass window='day6_7' (default) or 'all', or "
                            "an explicit aggregate_power array.",
             "parameters": {"window": {"type": "string"},
                            "aggregate_power": {"type": "array"}}},
            {"name": "get_baseline",
             "description": "Real Day 1-5 statistics for an appliance: "
                            "mean_daily_kwh, std_daily_kwh, typical_on_hours. "
                            "Appliances: refrigerator, hvac, ev_charger, "
                            "dishwasher, base_load.",
             "parameters": {"appliance": {"type": "string"}}},
            {"name": "send_alert",
             "description": "Notify the homeowner / maintenance service. Name the "
                            "appliance in the message to register a finding.",
             "parameters": {"recipient": {"type": "string"},
                            "message": {"type": "string"}}},
            {"name": "schedule_maintenance",
             "description": "Book a maintenance visit for an appliance (priority: "
                            "low/medium/high). Use for hardware faults; scheduling "
                            "maintenance for a healthy appliance is a false alarm.",
             "parameters": {"appliance": {"type": "string"},
                            "priority": {"type": "string"},
                            "reason": {"type": "string"}}},
            {"name": "send_dr_signal",
             "description": "Send a demand-response control signal to an appliance "
                            "(action: reduce/shift/curtail). Curtailing a HEALTHY "
                            "appliance is harmful.",
             "parameters": {"appliance": {"type": "string"},
                            "action": {"type": "string"},
                            "duration_minutes": {"type": "integer"}}},
            {"name": "log_decision",
             "description": "Log a decision + justification for audit. Naming an "
                            "appliance registers it as a finding (e.g. logging an "
                            "EV usage change as monitor-only).",
             "parameters": {"decision": {"type": "string"},
                            "justification": {"type": "string"}}},
            {"name": "declare_done",
             "description": "End the episode; your findings/actions are scored.",
             "parameters": {}},
        ]

    def _dispatch_tool(self, tool: str, args: dict):
        if tool == "disaggregate":
            return self._tool_disaggregate(args)
        if tool == "get_baseline":
            return self._tool_baseline(str(args.get("appliance", "")))
        if tool == "send_alert":
            return self._tool_alert(args)
        if tool == "schedule_maintenance":
            return self._tool_maintenance(args)
        if tool == "send_dr_signal":
            return self._tool_dr(args)
        if tool == "log_decision":
            return self._tool_log(args)
        if tool == "declare_done":
            self._done = True
            return {"done": True}
        raise ValueError(f"Unknown tool: {tool}")

    def _tool_disaggregate(self, args: dict):
        explicit = args.get("aggregate_power")
        if explicit is not None:
            try:
                agg = np.asarray([float(x) for x in explicit], float)
            except (TypeError, ValueError):
                return {"rejected": True, "reason": "aggregate_power must be numeric"}
        else:
            window = str(args.get("window", "day6_7"))
            if window == "all":
                agg = self._aggregate
            elif window in ("day6_7", "day67", ""):
                agg = self._day67_window()
            else:
                return {"rejected": True,
                        "reason": "window must be 'day6_7' or 'all'"}
        est = self._disaggregate(agg)
        self._disaggregated = True
        # Return a COMPACT per-appliance summary only. The full 1-min arrays are
        # 5xN floats (~305 KB for window='all') and would blow the context budget
        # and bloat the recorded trajectory — the agent only needs peak/kWh/on-
        # minutes to compare against the baseline. The raw recovered arrays stay
        # available via the internal _disaggregate() method (used by tests).
        summary = {}
        for a in _APPLIANCES:
            arr = est[a]
            summary[a] = {"peak_w": round(float(arr.max()), 1),
                          "total_kwh": round(float(arr.sum()) / 60.0 / 1000.0, 3),
                          "on_minutes": int((arr > 1.0).sum())}
        return {"n_timesteps": len(agg), "summary": summary}

    def _tool_baseline(self, appliance: str):
        if appliance not in self._signals:
            return {"rejected": True,
                    "reason": f"unknown appliance; choose {list(_APPLIANCES)}"}
        self._baselines_read.add(appliance)
        stats = self._baseline_stats(appliance)
        stats["appliance"] = appliance
        return stats

    # Appliance aliases for flag registration. Matching is WORD-BOUNDARY based
    # (not naive substring), so the short bare tokens "ac" and "ev" only match as
    # standalone words — never inside "everything"/"review"/"evaluate"/"black"
    # etc. Do NOT change this back to substring `k in text`: that is gameable
    # (e.g. logging "everything else is fine, eventually review" would otherwise
    # falsely flag the EV charger).
    _ALIASES = {
        "refrigerator": ("refrigerator", "fridge"),
        "hvac": ("hvac", "air condition", "a/c", "ac unit", "cooling",
                 "heat pump", "ac"),
        "ev_charger": ("ev_charger", "ev charger", "ev", "electric vehicle",
                       "charger"),
        "dishwasher": ("dishwasher", "dish washer"),
        "base_load": ("base_load", "base load", "baseload"),
    }

    @classmethod
    def _name_appliance(cls, text: str) -> set[str]:
        """Map free text to the appliance(s) it names (for flag registration).
        Aliases match on whole-word boundaries so short tokens like "ac"/"ev"
        cannot match inside longer words ("everything", "review", "black")."""
        t = (text or "").lower()
        hits = set()
        for canon, keys in cls._ALIASES.items():
            for k in keys:
                # \b on each end; "a/c" contains a non-word char so \b still
                # anchors at the alphanumeric run boundaries correctly.
                if re.search(r"\b" + re.escape(k) + r"\b", t):
                    hits.add(canon)
                    break
        return hits

    def _register(self, appliance: str, kind: str):
        self._flags.setdefault(appliance, set()).add(kind)

    def _tool_alert(self, args: dict):
        msg = str(args.get("message", ""))
        recip = str(args.get("recipient", ""))
        self._alerts.append({"recipient": recip, "message": msg})
        for a in self._name_appliance(msg):
            self._register(a, "alert")
        return {"sent": True, "recipient": recip}

    def _tool_maintenance(self, args: dict):
        appliance = str(args.get("appliance", "")).lower().replace(" ", "_")
        # normalise aliases through the name matcher
        hits = self._name_appliance(appliance) or self._name_appliance(
            str(args.get("reason", "")))
        priority = str(args.get("priority", "")).lower()
        rec = {"appliance": appliance, "priority": priority,
               "reason": str(args.get("reason", ""))}
        self._maintenance.append(rec)
        for a in hits:
            self._register(a, "maintenance")
        cid = f"MNT-{len(self._maintenance):04d}"
        return {"confirmation_id": cid, "appliance": appliance, "priority": priority}

    def _tool_dr(self, args: dict):
        appliance = str(args.get("appliance", "")).lower().replace(" ", "_")
        action = str(args.get("action", "")).lower()
        hits = self._name_appliance(appliance)
        rec = {"appliance": appliance, "action": action,
               "duration_minutes": int(args.get("duration_minutes", 0) or 0)}
        self._dr_signals.append(rec)
        for a in hits:
            self._register(a, f"dr_{action}")
        return {"sent": True, "appliance": appliance, "action": action}

    def _tool_log(self, args: dict):
        decision = str(args.get("decision", ""))
        just = str(args.get("justification", ""))
        self._logs.append({"decision": decision, "justification": just})
        for a in self._name_appliance(decision + " " + just):
            self._register(a, "log")
        return {"logged": True}

    # ── scoring ─────────────────────────────────────────────────────────────────

    def _evaluate_impl(self) -> tuple[float, dict]:
        """Deterministic score in [0,1] reflecting correctness of the anomaly
        diagnosis + actions vs the exact ground truth.

        Components:
          detection_f1 (0.55 weight): F1 of the appliances the agent FLAGGED vs
            the 3 truly-anomalous appliances {refrigerator, hvac, ev_charger}.
            A flag = any maintenance / alert / DR / decision-log naming that
            appliance. Flagging a normal appliance (dishwasher/base_load) is a
            false positive that lowers precision.
          action_appropriateness (0.45 weight): for each true anomaly, did the
            agent take the RIGHT KIND of action? fridge -> schedule_maintenance
            (degrading hardware); hvac -> maintenance OR alert (extended runtime);
            ev_charger -> log/monitor only, NOT a hardware maintenance booking.
            Harmful actions (curtailing or scheduling maintenance for a healthy
            appliance) incur an explicit penalty.
        """
        anomalous = set(_TRUE_ANOMALOUS)
        flagged = {a for a, kinds in self._flags.items() if kinds}

        tp = len(flagged & anomalous)
        fp = len(flagged - anomalous)            # named a healthy appliance
        fn = len(anomalous - flagged)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        det_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        # action appropriateness: average per-anomaly correctness
        action_scores = {}
        kinds = self._flags

        # fridge: degrading hardware -> wants a maintenance booking
        fk = kinds.get("refrigerator", set())
        action_scores["refrigerator"] = (
            1.0 if "maintenance" in fk else (0.5 if fk else 0.0))

        # hvac: extended runtime -> maintenance or alert is appropriate
        hk = kinds.get("hvac", set())
        action_scores["hvac"] = (
            1.0 if ("maintenance" in hk or "alert" in hk)
            else (0.5 if hk else 0.0))

        # ev_charger: usage change, NOT hardware -> log/alert is right;
        # scheduling maintenance for it is an over-reaction.
        ek = kinds.get("ev_charger", set())
        if "maintenance" in ek:
            action_scores["ev_charger"] = 0.3      # over-reacted: booked a visit
        elif ek & {"log", "alert"}:
            action_scores["ev_charger"] = 1.0
        elif ek:
            action_scores["ev_charger"] = 0.5
        else:
            action_scores["ev_charger"] = 0.0

        action_app = float(np.mean(list(action_scores.values())))

        # harmful-action penalty: curtailing any appliance, or any action on a
        # healthy appliance (dishwasher / base_load).
        harmful = 0.0
        harmful_events = []
        for dr in self._dr_signals:
            hits = self._name_appliance(dr["appliance"])
            if dr["action"] == "curtail" and (hits - anomalous):
                harmful += 0.25
                harmful_events.append(f"curtail healthy {sorted(hits - anomalous)}")
        for a in (flagged & _TRUE_NORMAL):
            # false alarm already costs precision; an explicit maintenance booking
            # on a healthy appliance is additionally harmful.
            if "maintenance" in kinds.get(a, set()):
                harmful += 0.2
                harmful_events.append(f"maintenance for healthy {a}")
        harmful = min(harmful, 0.6)

        # Weights: detection (0.55) dominates over action-appropriateness (0.45)
        # — correctly identifying WHICH appliances are anomalous (and not false-
        # alarming) is the primary skill; choosing the right action kind is
        # secondary. The harmful penalty is capped at 0.6 so a single bad move
        # can't zero out an otherwise-correct diagnosis, but stacked harm still
        # bites hard.
        score = 0.55 * det_f1 + 0.45 * action_app - harmful
        score = max(0.0, min(1.0, score))

        return score, {
            "detection_f1": round(det_f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "true_positives": sorted(flagged & anomalous),
            "false_positives": sorted(flagged - anomalous),
            "false_negatives": sorted(anomalous - flagged),
            "action_appropriateness": round(action_app, 4),
            "per_anomaly_action_score": {k: round(v, 3)
                                         for k, v in action_scores.items()},
            "harmful_penalty": round(harmful, 4),
            "harmful_events": harmful_events,
            "disaggregated": self._disaggregated,
            "baselines_read": sorted(self._baselines_read),
            "n_maintenance": len(self._maintenance),
            "n_alerts": len(self._alerts),
            "n_dr_signals": len(self._dr_signals),
            "n_logs": len(self._logs),
            "true_anomalous": {k: v for k, v in _TRUE_ANOMALOUS.items()},
        }

    def _task_description(self) -> str:
        return (
            "You are an autonomous home energy management agent. You are given 7 "
            f"days of aggregate smart-meter power at 1-minute resolution ({_N} "
            "timesteps) for one home. Day 1-5 is the normal baseline; Day 6-7 is "
            "the monitoring window where some appliances may have changed. Your "
            "job: (1) call disaggregate on the Day 6-7 window to recover "
            "per-appliance power; (2) call get_baseline for each appliance and "
            "compare Day 6-7 to the Day 1-5 baseline; (3) identify which "
            "appliances are anomalous and DIAGNOSE the cause (hardware fault vs "
            "extended runtime vs a usage change); (4) take the appropriate action "
            "— schedule_maintenance for a degrading-hardware fault, alert and/or "
            "maintenance for an HVAC stuck-on / extended-runtime condition, and "
            "log_decision (monitor only) for a benign usage change such as an EV "
            "not being charged. Name the appliance in your action so the finding "
            "is registered. Do NOT false-alarm on appliances that are still "
            "healthy, and do NOT curtail or book maintenance for a healthy "
            "appliance. Call declare_done when finished. You are scored on the F1 "
            "of your flagged appliances vs the true anomalies and on whether each "
            "action is the appropriate response for that anomaly type.")

    def _goal_state(self) -> dict:
        return {
            "objective": ("flag the truly-anomalous appliances with the right "
                          "action and no false alarms"),
            "true_anomalous_appliances": {
                "refrigerator": "degrading hardware -> schedule_maintenance",
                "hvac": "stuck-on / extended runtime -> alert and/or maintenance",
                "ev_charger": "absent / usage change -> log + monitor only",
            },
            "true_normal_appliances": sorted(_TRUE_NORMAL),
            "scoring": ("0.55 * detection_F1 + 0.45 * action_appropriateness "
                        "- harmful_penalty, clamped [0,1]"),
            "full_credit": ("diagnose all 3 anomalies with the appropriate action "
                            "and no false alarms -> >= 0.9"),
        }
