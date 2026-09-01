"""
ev_v2g_L4_001 — autonomous EV-fleet depot charging-management episode.

Grounding
---------
Reproduces the managed-charging / smart-charging agent loop for a commercial EV
depot: an autonomous agent perceives each vehicle's state of charge, the
time-of-use tariff and a utility demand-response (DR) signal, then plans a
staggered charging schedule that guarantees every vehicle reaches its departure
SoC without breaching the site's electrical service limit, while minimising
energy cost and honouring the DR commitment.

The depot physics and the smart-charging formulation follow the Caltech
Adaptive Charging Network (ACN-Data, Lee et al. 2019, "ACN-Data: Analysis and
Applications of an Open EV Charging Dataset", ACM e-Energy, doi:10.1145/3307772.
3328313) and the V2G/managed-charging fundamentals of Kempton & Tomic 2005
("Vehicle-to-grid power fundamentals", J. Power Sources,
doi:10.1016/j.jpowsour.2004.12.025): per-vehicle energy = (target − arrival)·
capacity, charging bounded by the EVSE power rating, the aggregate depot draw
bounded by the site transformer limit, and a DR event modelled as a temporary
reduction of that limit.

The agentic single-shot planning loop (perceive state → reason about
constraints → emit a verified schedule of tool actions → declare done) is the
methodology of AgenticHEMS (arXiv:2510.26603, "Agentic AI Home Energy
Management System: A Large Language Model Framework for Residential Load
Scheduling", El Makroum et al.), transferred from the home to the depot setting
— the same loop the sibling HEMS env (`vpp_scheduling_L4_002`) uses.

Grounded reference (real solver, no magic constants)
----------------------------------------------------
At reset the minimum-cost feasible charging schedule is computed by a real
`scipy.optimize.linprog` (HiGHS) solve and stored as the grading reference.
Decision variables are the per-van per-hour charging power x[v,h] over the
13-hour window 17:00..05:00 (06:00 is the departure boundary). Constraints:

  * 0 <= x[v,h] <= 7.2 kW (EVSE power rating);
  * sum_h x[v,h] >= need_v = (0.80 − arrival_soc_v)·60 kWh (reach 80% by 06:00);
  * sum_h x[v,h] <= (1 − arrival_soc_v)·60 kWh (never exceed the battery);
  * sum_v x[v,h] <= 120 kW for every hour, reduced to 120 − 60 = 60 kW during
    the DR window (hours 19:00, 20:00).
  * Objective: minimise sum_{v,h} price_h · x[v,h].

With 20 vans needing 450 kWh total and a 7-hour off-peak window (23:00 plus
00:00-05:00) offering 7·120 = 840 kWh of capacity at the $0.09 rate, the LP
places every kWh off-peak: the optimum is 450·0.09 = $40.50, peak concurrent
power ~82 kW (< 120), and zero charging in the DR window (so the DR commitment
is free). There is ample headroom — the binding fact the agent must discover is
that 20·7.2 = 144 kW > 120 kW, so charging MUST be staggered.

What is real vs simplified
--------------------------
REAL: the per-van energy bookkeeping (need, capacity room, delivered energy and
final SoC), the per-hour aggregate-power / site-limit accounting, the DR-window
limit reduction, the rejection of over-rating schedules, and the cost-optimal
feasible reference (a true HiGHS LP optimum over the full decision space, never
a hand-picked number).

SIMPLIFIED: hourly time resolution (no intra-hour ramping or charging-curve
taper); the continuous-power LP relaxation (per-van per-hour power is continuous
in [0, 7.2], not a discrete on/off EVSE state); deterministic arrival SoCs (a
fixed 25%..60% spread, mirrored in the task JSON context for reproducibility);
charging efficiency assumed unity; cost is energy·price with no demand charge.
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.optimize import linprog

from derbench_env import DERBenchEnv


@DERBenchEnv.register("ev_v2g_L4_001")
class EVFleetDepotChargingEnv(DERBenchEnv):
    """Schedule a 20-van depot to all-80%-by-06:00 within 120 kW, least-cost."""

    MAX_STEPS = 60

    # ── fleet + site (from the published task) ───────────────────────────────
    _FLEET_SIZE = 20
    _CAP_KWH = 60.0               # per-van battery capacity
    _CHARGER_KW = 7.2             # per-van EVSE power rating
    _TARGET_SOC = 0.80            # required departure SoC (fraction)
    _SITE_LIMIT_KW = 120.0        # depot service limit
    _DEPARTURE_HOUR = 6           # 06:00 departure (exclusive end of window)

    # operating window 17:00 -> 05:00 (06:00 is the departure boundary)
    _WINDOW_HOURS = list(range(17, 24)) + list(range(0, 6))   # 13 hourly slots

    # time-of-use tariff ($/kWh)
    _OFF_PEAK_HOURS = {0, 1, 2, 3, 4, 5, 23}
    _MID_PEAK_HOURS = {17, 18, 21, 22}
    _PEAK_HOURS = {19, 20}
    _OFF_PEAK_RATE = 0.09
    _MID_PEAK_RATE = 0.16
    _PEAK_RATE = 0.30

    # demand-response event
    _DR_EVENT_ID = "DR-EV-2026-0182"
    _DR_WINDOW = (19, 20)
    _DR_REDUCTION_KW = 60.0

    # ── deterministic arrival SoCs (mirrored in the task JSON context) ───────
    @classmethod
    def _arrival_socs(cls):
        """Fixed 25%..60% linear spread across the 20 vans (van_id 1..20)."""
        n = cls._FLEET_SIZE
        return [round(0.25 + (0.60 - 0.25) * i / (n - 1), 3) for i in range(n)]

    # ── helpers ──────────────────────────────────────────────────────────────

    def _price_of(self, h: int) -> float:
        if h in self._OFF_PEAK_HOURS:
            return self._OFF_PEAK_RATE
        if h in self._PEAK_HOURS:
            return self._PEAK_RATE
        return self._MID_PEAK_RATE

    def _hour_limit(self, h: int) -> float:
        """Effective site limit for hour h (reduced during the DR window)."""
        if h in self._DR_WINDOW:
            return self._SITE_LIMIT_KW - self._DR_REDUCTION_KW
        return self._SITE_LIMIT_KW

    def _need_kwh(self, soc: float) -> float:
        return max(0.0, (self._TARGET_SOC - soc) * self._CAP_KWH)

    def _room_kwh(self, soc: float) -> float:
        return max(0.0, (1.0 - soc) * self._CAP_KWH)

    def _window_slice(self, start_hour: int, end_hour: int):
        """Window-order list of hours in [start_hour, end_hour) along the
        17:00->06:00 timeline. `end_hour` may be 6 (the 06:00 departure)."""
        order = self._WINDOW_HOURS
        pos = {h: i for i, h in enumerate(order)}
        pos[self._DEPARTURE_HOUR] = len(order)   # 06:00 == one past the last slot
        if start_hour not in pos or end_hour not in pos:
            return None
        i, j = pos[start_hour], pos[end_hour]
        if j <= i:
            return None
        return [order[k] for k in range(i, j)]

    # ── env lifecycle ────────────────────────────────────────────────────────

    def _reset_impl(self) -> dict:
        socs = self._arrival_socs()
        self._van_soc0 = {v + 1: socs[v] for v in range(self._FLEET_SIZE)}
        self._prices = {h: self._price_of(h) for h in self._WINDOW_HOURS}

        # agent schedule state: van_id -> {hour: power_kw}
        self._schedule = {v: {} for v in self._van_soc0}
        self._dr_committed_kw = 0.0
        self._log: list[dict] = []
        self._declared = False

        # cost-optimal feasible reference (real LP solve, deterministic)
        (self._ref_cost, self._ref_peak_kw,
         self._ref_perhour) = self._solve_reference()

        return {
            "scenario": f"EV-fleet depot charging — get all {self._FLEET_SIZE} vans to "
                        f"{self._TARGET_SOC*100:.0f}% SoC by 06:00 within the "
                        f"{self._SITE_LIMIT_KW:.0f} kW site limit, least-cost, honouring "
                        f"DR event {self._DR_EVENT_ID}.",
            "current_time": "17:00",
            "fleet_size": self._FLEET_SIZE,
            "site_power_limit_kw": self._SITE_LIMIT_KW,
            "departure_hour": self._DEPARTURE_HOUR,
            "required_departure_soc_pct": self._TARGET_SOC * 100.0,
            "van_profile": {"battery_kwh": self._CAP_KWH, "charger_kw": self._CHARGER_KW},
            "dr_event": {"event_id": self._DR_EVENT_ID, "window_hours": list(self._DR_WINDOW),
                         "requested_reduction_kw": self._DR_REDUCTION_KW},
            "note": f"{self._FLEET_SIZE} x {self._CHARGER_KW} = "
                    f"{self._FLEET_SIZE * self._CHARGER_KW:.0f} kW > "
                    f"{self._SITE_LIMIT_KW:.0f} kW limit, so charging MUST be staggered.",
        }

    def _solve_reference(self):
        """Least-cost feasible per-van per-hour charging schedule via linprog.

        Returns (reference_cost_usd, reference_peak_power_kw, per-hour total kW)."""
        socs = self._arrival_socs()
        H = self._WINDOW_HOURS
        nH = len(H)
        N = self._FLEET_SIZE
        nvar = N * nH

        def idx(v, hi):
            return v * nH + hi

        c = np.zeros(nvar)
        for v in range(N):
            for hi, h in enumerate(H):
                c[idx(v, hi)] = self._prices[h]
        bounds = [(0.0, self._CHARGER_KW)] * nvar

        A_ub, b_ub = [], []
        # deadline: sum_h x[v,h] >= need_v  ->  -sum <= -need
        for v in range(N):
            row = np.zeros(nvar)
            for hi in range(nH):
                row[idx(v, hi)] = -1.0
            A_ub.append(row)
            b_ub.append(-self._need_kwh(socs[v]))
        # capacity room: sum_h x[v,h] <= room_v
        for v in range(N):
            row = np.zeros(nvar)
            for hi in range(nH):
                row[idx(v, hi)] = 1.0
            A_ub.append(row)
            b_ub.append(self._room_kwh(socs[v]))
        # per-hour site limit (DR-reduced during the DR window)
        for hi, h in enumerate(H):
            row = np.zeros(nvar)
            for v in range(N):
                row[idx(v, hi)] = 1.0
            A_ub.append(row)
            b_ub.append(self._hour_limit(h))

        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"EV-fleet reference LP failed to converge: {res.message}")
        x = res.x.reshape(N, nH)
        perhour = {h: round(float(x[:, hi].sum()), 4) for hi, h in enumerate(H)}
        peak = round(float(max(perhour.values())), 4)
        return round(float(res.fun), 4), peak, perhour

    # ── realized-schedule accounting (the grading model) ─────────────────────

    def _per_hour_power(self) -> dict:
        """Aggregate depot power (kW) each window hour from the agent schedule."""
        ph = {h: 0.0 for h in self._WINDOW_HOURS}
        for v, blocks in self._schedule.items():
            for h, p in blocks.items():
                ph[h] += p
        return ph

    def _delivered_kwh(self, van_id: int) -> float:
        """Energy delivered to a van, capped at filling it to 100% capacity."""
        raw = sum(self._schedule[van_id].values())   # 1 h slots -> kWh == kW
        return min(raw, self._room_kwh(self._van_soc0[van_id]))

    def _final_soc(self, van_id: int) -> float:
        return self._van_soc0[van_id] + self._delivered_kwh(van_id) / self._CAP_KWH

    def _realized_cost(self) -> float:
        """Energy cost of the realized schedule. Energy beyond the battery's
        capacity room is physically not absorbed, so it carries no cost: bill the
        delivered (capped) energy at each hour's price, scaling a van's hourly
        draw uniformly if its block over-fills the battery."""
        cost = 0.0
        for v, blocks in self._schedule.items():
            raw = sum(blocks.values())
            if raw <= 1e-12:
                continue
            scale = min(1.0, self._room_kwh(self._van_soc0[v]) / raw)
            for h, p in blocks.items():
                cost += p * scale * self._prices[h]
        return cost

    # ── tools ────────────────────────────────────────────────────────────────

    def _tools(self) -> list[dict]:
        return [
            {"name": "get_fleet_state",
             "description": "Each van's id, current SoC (%), battery capacity (kWh), "
                            "charger power (kW) and the energy (kWh) it needs to reach "
                            "80% by 06:00.",
             "parameters": {}},
            {"name": "get_price_forecast_24h",
             "description": "Hourly electricity prices ($/kWh) across the 17:00->06:00 "
                            "charging window.",
             "parameters": {}},
            {"name": "get_site_limit",
             "description": "The depot's site power limit (kW) and its DR-reduced value "
                            "during the demand-response window.",
             "parameters": {}},
            {"name": "set_charge_schedule",
             "description": "Schedule a van (van_id 1-20) to charge at power_kw (<= 7.2 kW) "
                            "for every hour in [start_hour, end_hour); hours must lie in the "
                            "17:00->06:00 window (end_hour may be 6 for the 06:00 deadline). "
                            "Replaces any prior schedule for that van. Rejected if power_kw "
                            "exceeds the charger rating or the hours fall outside the window.",
             "parameters": {"van_id": {"type": "integer"},
                            "start_hour": {"type": "integer"},
                            "end_hour": {"type": "integer"},
                            "power_kw": {"type": "number"}}},
            {"name": "get_dr_event",
             "description": "The active demand-response event: id, window hours and "
                            "requested depot power reduction (kW).",
             "parameters": {}},
            {"name": "respond_dr",
             "description": "Commit a depot-level power reduction (kW) during the DR event "
                            "window. The commitment is honoured only if the realized depot "
                            "power during the window is at least the committed amount below "
                            "the site limit.",
             "parameters": {"event_id": {"type": "string"},
                            "reduction_kw": {"type": "number"}}},
            {"name": "log_decision",
             "description": "Record a scheduling decision with its cost/constraint rationale.",
             "parameters": {"action": {"type": "string"},
                            "justification": {"type": "string"}}},
            {"name": "declare_done",
             "description": "End the episode; the realized charging schedule is then graded.",
             "parameters": {}},
        ]

    def _dispatch_tool(self, tool: str, args: dict):
        if tool == "get_fleet_state":
            return {"vans": [
                {"van_id": v, "soc_pct": round(self._van_soc0[v] * 100.0, 2),
                 "battery_kwh": self._CAP_KWH, "charger_kw": self._CHARGER_KW,
                 "energy_need_kwh": round(self._need_kwh(self._van_soc0[v]), 3)}
                for v in sorted(self._van_soc0)]}
        if tool == "get_price_forecast_24h":
            return {"prices_per_kwh": {str(h): round(self._prices[h], 4)
                                       for h in self._WINDOW_HOURS},
                    "window_hours": list(self._WINDOW_HOURS)}
        if tool == "get_site_limit":
            return {"site_power_limit_kw": self._SITE_LIMIT_KW,
                    "dr_window_hours": list(self._DR_WINDOW),
                    "dr_reduced_limit_kw": self._SITE_LIMIT_KW - self._DR_REDUCTION_KW}
        if tool == "set_charge_schedule":
            return self._tool_set_charge_schedule(
                int(args["van_id"]), int(args["start_hour"]),
                int(args["end_hour"]), float(args["power_kw"]))
        if tool == "get_dr_event":
            return {"event_id": self._DR_EVENT_ID, "window_hours": list(self._DR_WINDOW),
                    "requested_reduction_kw": self._DR_REDUCTION_KW,
                    "effective_limit_during_window_kw":
                        self._SITE_LIMIT_KW - self._DR_REDUCTION_KW}
        if tool == "respond_dr":
            return self._tool_respond_dr(str(args.get("event_id", "")),
                                         float(args["reduction_kw"]))
        if tool == "log_decision":
            self._log.append({"action": args.get("action", ""),
                              "justification": args.get("justification", "")})
            return {"logged": True, "log_entries": len(self._log)}
        if tool == "declare_done":
            self._declared = True
            self._done = True
            ph = self._per_hour_power()
            return {"status": "episode_complete",
                    "all_vans_meet_soc": all(
                        self._final_soc(v) >= self._TARGET_SOC - 1e-6 for v in self._van_soc0),
                    "peak_power_kw": round(max(ph.values()), 3),
                    "estimated_cost_usd": round(self._realized_cost(), 4)}
        raise ValueError(f"Unknown tool: {tool}")

    def _tool_set_charge_schedule(self, van_id: int, start_hour: int,
                                  end_hour: int, power_kw: float):
        if van_id not in self._van_soc0:
            return {"rejected": True, "van_id": van_id,
                    "reason": f"unknown van_id {van_id} (valid 1-{self._FLEET_SIZE})"}
        if power_kw < 0:
            return {"rejected": True, "van_id": van_id,
                    "reason": "power_kw must be non-negative"}
        if power_kw > self._CHARGER_KW + 1e-9:
            return {"rejected": True, "van_id": van_id,
                    "reason": f"power_kw={power_kw} exceeds the {self._CHARGER_KW} kW "
                              f"charger rating",
                    "charger_kw": self._CHARGER_KW}
        block = self._window_slice(start_hour, end_hour)
        if block is None:
            return {"rejected": True, "van_id": van_id,
                    "reason": f"[{start_hour}, {end_hour}) is not a valid sub-range of the "
                              f"17:00->06:00 window {self._WINDOW_HOURS} (end may be 6)"}
        # replace this van's schedule
        self._schedule[van_id] = {h: power_kw for h in block}
        delivered = self._delivered_kwh(van_id)
        final_soc = self._final_soc(van_id)
        return {"accepted": True, "van_id": van_id, "hours": block, "power_kw": power_kw,
                "scheduled_energy_kwh": round(power_kw * len(block), 3),
                "delivered_energy_kwh": round(delivered, 3),
                "final_soc_pct": round(final_soc * 100.0, 2),
                "meets_target": bool(final_soc >= self._TARGET_SOC - 1e-6)}

    def _tool_respond_dr(self, event_id: str, reduction_kw: float):
        if event_id and event_id != self._DR_EVENT_ID:
            return {"rejected": True, "reason": f"unknown event {event_id!r} "
                    f"(active event: {self._DR_EVENT_ID})"}
        self._dr_committed_kw = max(0.0, reduction_kw)
        return {"accepted": True, "event_id": self._DR_EVENT_ID,
                "committed_reduction_kw": self._dr_committed_kw,
                "note": "honoured only if realized depot power during the window stays at "
                        "least this far below the site limit"}

    # ── scoring ──────────────────────────────────────────────────────────────

    def _hard_constraints(self) -> dict:
        ph = self._per_hour_power()
        final_socs = {v: self._final_soc(v) for v in self._van_soc0}
        n_vans_ok = sum(1 for v in self._van_soc0
                        if final_socs[v] >= self._TARGET_SOC - 1e-6)
        all_soc = n_vans_ok == self._FLEET_SIZE
        peak = max(ph.values()) if ph else 0.0
        site_ok = all(ph[h] <= self._SITE_LIMIT_KW + 1e-6 for h in self._WINDOW_HOURS)
        # DR honoured: committed > 0 AND realized power in the DR window stays at
        # least the committed amount below the site limit (i.e. effective headroom).
        dr_window_peak = max((ph[h] for h in self._DR_WINDOW), default=0.0)
        dr_headroom_ok = dr_window_peak <= self._SITE_LIMIT_KW - self._dr_committed_kw + 1e-6
        dr_honored = (self._dr_committed_kw >= self._DR_REDUCTION_KW - 1e-6) and dr_headroom_ok
        return {
            "n_vans_meet_soc": n_vans_ok,
            "all_vans_meet_soc": bool(all_soc),
            "peak_power_kw": round(peak, 3),
            "site_limit_respected": bool(site_ok),
            "dr_committed_kw": round(self._dr_committed_kw, 3),
            "dr_window_peak_kw": round(dr_window_peak, 3),
            "dr_honored": bool(dr_honored),
            "per_hour_power_kw": {str(h): round(ph[h], 3) for h in self._WINDOW_HOURS},
            "final_soc_pct": {str(v): round(final_socs[v] * 100.0, 2) for v in self._van_soc0},
        }

    def _evaluate_impl(self) -> tuple[float, dict]:
        """Deterministic score in [0,1]:
          HARD CONSTRAINTS (gating, 0.60): (a) all 20 vans >= 80% by 06:00,
            (b) per-hour depot power <= 120 kW every hour, (c) the 60 kW DR
            commitment honoured during hours 19-20 — equal thirds. The cost
            credit below is heavily discounted while any hard constraint is
            violated, so a cheap-but-infeasible schedule cannot score high.
          COST-OPTIMALITY (0.40): closeness of the realized energy cost to the LP
            reference optimum (full credit at or below the optimum), awarded only
            when all hard constraints are satisfied.
        """
        hard = self._hard_constraints()
        flags = [hard["all_vans_meet_soc"], hard["site_limit_respected"], hard["dr_honored"]]
        n_hard = len(flags)
        n_met = sum(flags)
        all_hard_ok = n_met == n_hard

        # Hard block (0.60): proportional credit, full only when all three met.
        # The SoC component is itself graded on the fraction of vans that make it,
        # so a near-complete schedule is distinguished from a no-op.
        soc_frac = hard["n_vans_meet_soc"] / self._FLEET_SIZE
        hard_score = (0.60 / n_hard) * (soc_frac
                                        + (1.0 if hard["site_limit_respected"] else 0.0)
                                        + (1.0 if hard["dr_honored"] else 0.0))

        # Cost-optimality (0.40): closeness of realized cost to the LP optimum.
        # Below-optimum cost is only reachable by under-delivering energy (which
        # fails the SoC hard constraint), so it is clamped to full credit and the
        # infeasibility is already punished by the gate.
        cost = self._realized_cost()
        ref = self._ref_cost
        if ref > 1e-9:
            ratio = cost / ref                       # >= 1 at/above the optimum
            cost_credit = max(0.0, min(1.0, 1.0 - (ratio - 1.0)))
        else:
            cost_credit = 1.0 if cost <= 1e-9 else 0.0
        if not all_hard_ok:
            cost_credit *= 0.2                       # cannot bank cost on infeasibility
        cost_score = 0.40 * cost_credit

        score = max(0.0, min(1.0, hard_score + cost_score))

        details = dict(hard)
        details.update({
            "all_hard_constraints_satisfied": bool(all_hard_ok),
            "hard_constraints_met": f"{n_met}/{n_hard}",
            "realized_cost_usd": round(cost, 4),
            "reference_cost_usd": self._ref_cost,
            "reference_peak_power_kw": self._ref_peak_kw,
            "reference_per_hour_kw": self._ref_perhour,
            "log_entries": len(self._log),
            "hard_constraint_score": round(hard_score, 4),
            "cost_optimality_score": round(cost_score, 4),
        })
        return score, details

    def _task_description(self) -> str:
        return (
            "You are an autonomous EV-fleet charging-management agent for a depot of "
            f"{self._FLEET_SIZE} electric delivery vans. It is 17:00; every van has just "
            "plugged in and departs at 06:00. Objectives, in priority order: (1) every van "
            f"must reach at least {self._TARGET_SOC*100:.0f}% SoC by 06:00 (HARD); (2) total "
            f"depot power must stay within the {self._SITE_LIMIT_KW:.0f} kW site limit at all "
            f"times (HARD); (3) respond to demand-response event {self._DR_EVENT_ID} by "
            f"reducing depot power by {self._DR_REDUCTION_KW:.0f} kW during hours "
            f"{self._DR_WINDOW[0]}:00-{self._DR_WINDOW[1]}:00 (HARD once committed); and (4) "
            "minimise total energy cost.\n"
            f"Each van has a {self._CAP_KWH:.0f} kWh battery and a {self._CHARGER_KW} kW "
            f"charger. Note {self._FLEET_SIZE} x {self._CHARGER_KW} = "
            f"{self._FLEET_SIZE * self._CHARGER_KW:.0f} kW exceeds the "
            f"{self._SITE_LIMIT_KW:.0f} kW limit, so charging MUST be staggered; the "
            "13-hour 17:00->06:00 window has ample time for the ~4-5 h each van needs.\n"
            "Inspect the fleet, prices and DR event first; compute each van's energy need; "
            "build a staggered schedule that keeps concurrent power <= 120 kW and shifts the "
            "bulk of charging into the cheap off-peak hours (23:00 and 00:00-05:00 at "
            "$0.09); avoid charging during the $0.30 peak / DR window 19:00-20:00 and commit "
            "the 60 kW DR reduction; log your decisions, then call declare_done.")

    def _goal_state(self) -> dict:
        return {"fleet_size": self._FLEET_SIZE,
                "required_departure_soc_pct": self._TARGET_SOC * 100.0,
                "departure_hour": self._DEPARTURE_HOUR,
                "site_power_limit_kw": self._SITE_LIMIT_KW,
                "dr_event": {"event_id": self._DR_EVENT_ID, "window_hours": list(self._DR_WINDOW),
                             "reduction_kw": self._DR_REDUCTION_KW},
                "reference_cost_usd": self._ref_cost}
