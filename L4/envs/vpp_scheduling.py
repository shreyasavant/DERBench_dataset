"""
vpp_scheduling_L4_001 — autonomous VPP curtailment-response dispatch episode.

Grounding
---------
Inspired by AgenticPolicySearch (arXiv:2501.19340, "Adaptive Self-Improvement
for Smarter Energy Systems using Agentic Policy Search", Sommer et al.) — the
shared ground is the domain (an LLM agent dispatching battery/solar assets,
graded against an LP optimum, as APS also benchmarks against LP optimality),
NOT the paper's loop: APS iteratively generates and refines complete control
policies as executable code against simulator feedback, whereas this episode
is a single perceive -> tool-call dispatch -> verify episode. Here an LLM agent
perceives the live state of a distributed-energy portfolio, reasons about a
grid signal under price information, and emits a sequence of dispatch actions
that it verifies against the system constraints before committing.
LLM-ADN-Dispatch (arXiv:2507.21162, an NL-to-optimization-model agent pipeline
for active-distribution-network dispatch) loosely informs the framing of the
grid-operator curtailment signal as a net-power-exchange target the aggregator
must meet with its own resources.

The published L4 task ("reduce net grid INJECTION by 200 kW" while the ground
truth "prioritizes battery DISCHARGE") is physically incoherent as literally
stated — discharging a battery normally increases export. We resolve it with a
single coherent model:

    The grid operator requests a 200 kW reduction in the VPP's NET GRID POWER
    EXCHANGE over a 1-hour dispatch block. The aggregator meets it with its own
    dispatchable reduction capacity. Two levers reduce net exchange:
      * battery discharge — serves local load that would otherwise be imported,
        backing down grid draw. Revenue-neutral (stored energy already paid for),
        so it is the CHEAP, preferred lever.
      * solar curtailment — also reduces net exchange, but at the $0.22/kWh
        spot price every curtailed kW is lost revenue, so it is the COSTLY
        fallback used only for the residual the batteries cannot cover.

    Each battery's dispatchable power over the block is energy-limited:
        usable_kW = min(max_discharge_kw, (SoC - SoC_min) * capacity_kWh / hours)
    A battery already at the 20% SoC floor (B3) has zero usable headroom and is
    therefore ineligible — matching the published ground truth's B3 exclusion.

"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint

from derbench_env import DERBenchEnv


@DERBenchEnv.register("vpp_scheduling_L4_001")
class VPPCurtailmentDispatchEnv(DERBenchEnv):
    """Meet a 200 kW net-exchange reduction with a DER portfolio, least-cost."""

    MAX_STEPS = 25

    # ── portfolio + signal (from the published task) ────────────────────────
    _SIGNAL_KW = 200.0            # requested net-exchange reduction
    _DUR_H = 1.0                  # dispatch block length (hours)
    _MIN_SOC_PCT = 20.0           # SoC floor any battery must respect
    _PRICE_NOW = 0.22             # spot price during the spike ($/kWh)
    _PRICE_OFFPEAK = 0.08         # pre-spike price ($/kWh)
    _ACTION_BUDGET = 3            # max dispatch actions (battery + solar)

    # batteries: id -> (capacity_kWh, initial SoC %, max discharge kW)
    _BATTERIES = {
        "B1": {"capacity_kwh": 50.0, "soc_pct": 40.0, "max_discharge_kw": 25.0},
        "B2": {"capacity_kwh": 100.0, "soc_pct": 75.0, "max_discharge_kw": 50.0},
        "B3": {"capacity_kwh": 30.0, "soc_pct": 20.0, "max_discharge_kw": 15.0},
    }
    # solar farms: id -> nameplate current output kW
    _SOLAR = {"S1": 500.0, "S2": 300.0}

    # ── env lifecycle ───────────────────────────────────────────────────────

    def _usable_kw(self, bid: str, soc_pct: float) -> float:
        """Energy-limited dispatchable discharge power for `bid` at `soc_pct`."""
        b = self._BATTERIES[bid]
        usable_kwh = max(0.0, (soc_pct - self._MIN_SOC_PCT) / 100.0 * b["capacity_kwh"])
        return min(b["max_discharge_kw"], usable_kwh / self._DUR_H)

    def _initial_usable_kw(self, bid: str) -> float:
        """Energy-limited headroom at the battery's *initial* SoC (the LP bound)."""
        return self._usable_kw(bid, self._BATTERIES[bid]["soc_pct"])

    def _reset_impl(self) -> dict:
        # mutable portfolio state
        self._soc = {bid: b["soc_pct"] for bid, b in self._BATTERIES.items()}
        self._batt_dispatch_kw = {bid: 0.0 for bid in self._BATTERIES}
        self._solar_curtail_kw = {sid: 0.0 for sid in self._SOLAR}
        self._solar_curtail_pct = {sid: 0.0 for sid in self._SOLAR}
        self._dispatch_actions = 0     # battery + solar dispatch calls
        self._b3_dispatch_attempts = 0
        self._log: list[dict] = []
        self._declared = False

        # price trajectory across the 4-hour window: spike at hour 2 onward.
        self._prices = [self._PRICE_OFFPEAK, self._PRICE_NOW,
                        self._PRICE_NOW, self._PRICE_NOW]

        # cost-optimal feasible reference (real LP solve, deterministic)
        self._ref_cost, self._ref_dispatch = self._solve_reference()

        avail = {bid: round(self._usable_kw(bid, self._soc[bid]), 3)
                 for bid in self._BATTERIES}
        return {
            "scenario": "VPP curtailment response — reduce net grid exchange by "
                        f"{self._SIGNAL_KW:.0f} kW over a {self._DUR_H:.0f}-hour block.",
            "curtailment_signal_kw": self._SIGNAL_KW,
            "price_now_per_kwh": self._PRICE_NOW,
            "min_soc_pct": self._MIN_SOC_PCT,
            "dispatch_action_limit": self._ACTION_BUDGET,
            "batteries": [
                {"asset_id": bid, "capacity_kwh": b["capacity_kwh"],
                 "soc_pct": self._soc[bid], "max_discharge_kw": b["max_discharge_kw"],
                 "usable_discharge_kw": avail[bid]}
                for bid, b in self._BATTERIES.items()
            ],
            "solar_farms": [
                {"asset_id": sid, "current_output_kw": out} for sid, out in self._SOLAR.items()
            ],
        }

    def _solve_reference(self):
        """Least-cost feasible dispatch meeting the signal, via scipy.linprog.

        Variables: per-battery discharge kW (free) + per-solar curtail kW
        (cost = price x kW x hours). Minimise total cost s.t. the summed
        reduction >= signal and per-asset feasibility bounds.
        Returns (reference_cost_usd, {asset_id: dispatch_kw})."""
        batt_ids = list(self._BATTERIES)
        solar_ids = list(self._SOLAR)
        # objective: batteries cost 0 (revenue-neutral), solar costs lost revenue
        c = [0.0] * len(batt_ids) + [self._PRICE_NOW * self._DUR_H] * len(solar_ids)
        bounds = [(0.0, self._initial_usable_kw(b)) for b in batt_ids]
        bounds += [(0.0, self._SOLAR[s]) for s in solar_ids]
        # total reduction >= signal  ->  -sum(vars) <= -signal
        A_ub = [[-1.0] * (len(batt_ids) + len(solar_ids))]
        b_ub = [-self._SIGNAL_KW]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"VPP reference LP failed to converge: {res.message}")
        x = res.x
        dispatch = {}
        for i, b in enumerate(batt_ids):
            dispatch[b] = round(float(x[i]), 4)
        for j, s in enumerate(solar_ids):
            dispatch[s] = round(float(x[len(batt_ids) + j]), 4)
        return round(float(res.fun), 4), dispatch

    # ── current-state helpers ────────────────────────────────────────────────

    def _total_reduction_kw(self) -> float:
        return (sum(self._batt_dispatch_kw.values())
                + sum(self._solar_curtail_kw.values()))

    def _current_cost_usd(self) -> float:
        # only solar curtailment carries a cost (lost revenue at spot price)
        return self._PRICE_NOW * self._DUR_H * sum(self._solar_curtail_kw.values())

    def _active_violations(self) -> list[dict]:
        v = []
        for bid in self._BATTERIES:
            if self._soc[bid] < self._MIN_SOC_PCT - 1e-9:
                v.append({"asset_id": bid, "type": "soc_floor_breach",
                          "soc_pct": round(self._soc[bid], 3),
                          "min_soc_pct": self._MIN_SOC_PCT})
            if self._batt_dispatch_kw[bid] > self._BATTERIES[bid]["max_discharge_kw"] + 1e-9:
                v.append({"asset_id": bid, "type": "power_limit_breach",
                          "dispatch_kw": round(self._batt_dispatch_kw[bid], 3),
                          "max_discharge_kw": self._BATTERIES[bid]["max_discharge_kw"]})
        if self._dispatch_actions > self._ACTION_BUDGET:
            v.append({"type": "action_budget_breach",
                      "dispatch_actions": self._dispatch_actions,
                      "limit": self._ACTION_BUDGET})
        return v

    # ── tools ─────────────────────────────────────────────────────────────────

    def _tools(self) -> list[dict]:
        return [
            {"name": "get_asset_status",
             "description": "Live SoC (%), usable discharge headroom (kW), and status "
                            "for a battery (B1/B2/B3), or current output (kW) for a "
                            "solar farm (S1/S2).",
             "parameters": {"asset_id": {"type": "string"}}},
            {"name": "get_price_forecast",
             "description": "Forecast electricity prices ($/kWh) for the next N hours.",
             "parameters": {"hours": {"type": "integer"}}},
            {"name": "dispatch_battery",
             "description": "Discharge a battery at power_kw (positive) over the 1-hour "
                            "operating block (duration_min is recorded but the dispatch is "
                            "settled over the full block). Counts as a dispatch action. "
                            "Rejected if power_kw exceeds the energy-limited usable "
                            "discharge headroom (min of max discharge power and the energy "
                            "above the 20% SoC floor divided by the block length).",
             "parameters": {"asset_id": {"type": "string"},
                            "power_kw": {"type": "number"},
                            "duration_min": {"type": "number"}}},
            {"name": "curtail_solar",
             "description": "Curtail a solar farm's output by pct (0-100). Reduces net "
                            "grid exchange but loses revenue at the current price. Counts "
                            "as a dispatch action.",
             "parameters": {"asset_id": {"type": "string"}, "pct": {"type": "number"}}},
            {"name": "check_constraint_violation",
             "description": "Report any active constraint violations (SoC floors, power "
                            "limits, dispatch-action budget) from the current state.",
             "parameters": {}},
            {"name": "log_decision",
             "description": "Record an operational decision (including a decision NOT to "
                            "dispatch an asset) with its justification.",
             "parameters": {"action": {"type": "string"},
                            "justification": {"type": "string"}}},
            {"name": "declare_done",
             "description": "End the episode; the final dispatch is then graded.",
             "parameters": {}},
        ]

    def _dispatch_tool(self, tool: str, args: dict):
        if tool == "get_asset_status":
            return self._tool_status(str(args["asset_id"]))
        if tool == "get_price_forecast":
            h = max(0, min(int(args.get("hours", 4)), len(self._prices)))
            return {"hours": h, "prices_per_kwh": self._prices[:h],
                    "note": f"spike to ${self._PRICE_NOW}/kWh from hour 2 onward"}
        if tool == "dispatch_battery":
            return self._tool_dispatch_battery(
                str(args["asset_id"]), float(args["power_kw"]),
                float(args.get("duration_min", self._DUR_H * 60)))
        if tool == "curtail_solar":
            return self._tool_curtail_solar(str(args["asset_id"]), float(args["pct"]))
        if tool == "check_constraint_violation":
            v = self._active_violations()
            return {"violations": v, "n_violations": len(v),
                    "total_reduction_kw": round(self._total_reduction_kw(), 3),
                    "signal_kw": self._SIGNAL_KW}
        if tool == "log_decision":
            self._log.append({"action": args.get("action", ""),
                              "justification": args.get("justification", "")})
            return {"logged": True, "log_entries": len(self._log)}
        if tool == "declare_done":
            self._declared = True
            self._done = True
            return {"status": "episode_complete",
                    "total_reduction_kw": round(self._total_reduction_kw(), 3),
                    "signal_met": self._total_reduction_kw() >= self._SIGNAL_KW - 1e-6,
                    "estimated_cost_usd": round(self._current_cost_usd(), 4)}
        raise ValueError(f"Unknown tool: {tool}")

    def _tool_status(self, asset_id: str):
        if asset_id in self._BATTERIES:
            b = self._BATTERIES[asset_id]
            usable = self._usable_kw(asset_id, self._soc[asset_id])
            return {"asset_id": asset_id, "type": "battery",
                    "soc_pct": round(self._soc[asset_id], 3),
                    "capacity_kwh": b["capacity_kwh"],
                    "max_discharge_kw": b["max_discharge_kw"],
                    "usable_discharge_kw": round(usable, 3),
                    "currently_dispatched_kw": round(self._batt_dispatch_kw[asset_id], 3),
                    "eligible": usable > 1e-9,
                    "status": "at_soc_floor" if usable <= 1e-9 else "available"}
        if asset_id in self._SOLAR:
            out = self._SOLAR[asset_id]
            return {"asset_id": asset_id, "type": "solar",
                    "nameplate_output_kw": out,
                    "curtailed_kw": round(self._solar_curtail_kw[asset_id], 3),
                    "curtailed_pct": round(self._solar_curtail_pct[asset_id], 3),
                    "current_output_kw": round(out - self._solar_curtail_kw[asset_id], 3)}
        raise ValueError(f"Unknown asset_id: {asset_id} "
                         f"(valid: {list(self._BATTERIES) + list(self._SOLAR)})")

    def _tool_dispatch_battery(self, asset_id: str, power_kw: float, duration_min: float):
        if asset_id not in self._BATTERIES:
            raise ValueError(f"Unknown battery: {asset_id}")
        if asset_id == "B3":
            self._b3_dispatch_attempts += 1
        self._dispatch_actions += 1
        b = self._BATTERIES[asset_id]
        if power_kw < 0:
            return {"rejected": True, "asset_id": asset_id,
                    "reason": "charging is not a valid curtailment-response action here"}
        # Single-block energy model: the dispatch is settled over the fixed 1-hour
        # operating block (`_DUR_H`) regardless of the agent-supplied duration_min,
        # so the energy-limited headroom binds. Reject any power above the SAME
        # bound the LP reference uses — usable_kW = min(max_discharge_kw,
        # energy_headroom / _DUR_H) — so the env's feasible set equals the LP's.
        usable = self._usable_kw(asset_id, self._soc[asset_id])
        if power_kw > usable + 1e-9:
            limiter = ("max_discharge_kw" if usable >= b["max_discharge_kw"] - 1e-9
                       else "energy headroom above the SoC floor")
            return {"rejected": True, "asset_id": asset_id,
                    "reason": f"power_kw={power_kw} exceeds usable_discharge_kw="
                              f"{round(usable, 3)} (limited by {limiter} over the "
                              f"{self._DUR_H:.0f}-hour block)",
                    "usable_discharge_kw": round(usable, 3),
                    "max_discharge_kw": b["max_discharge_kw"]}
        # accept: settle energy over the fixed block, drawing down SoC accordingly.
        delta_soc_pct = power_kw * self._DUR_H / b["capacity_kwh"] * 100.0
        new_soc = self._soc[asset_id] - delta_soc_pct
        self._soc[asset_id] = new_soc
        self._batt_dispatch_kw[asset_id] += power_kw
        return {"accepted": True, "asset_id": asset_id, "dispatched_kw": power_kw,
                "duration_min": duration_min,
                "settled_over_block_h": self._DUR_H, "new_soc_pct": round(new_soc, 3),
                "dispatch_actions_used": self._dispatch_actions,
                "total_reduction_kw": round(self._total_reduction_kw(), 3)}

    def _tool_curtail_solar(self, asset_id: str, pct: float):
        if asset_id not in self._SOLAR:
            raise ValueError(f"Unknown solar farm: {asset_id}")
        self._dispatch_actions += 1
        pct = max(0.0, min(100.0, pct))
        out = self._SOLAR[asset_id]
        # `pct` sets the absolute curtailment level for this farm
        curtail_kw = out * pct / 100.0
        self._solar_curtail_pct[asset_id] = pct
        self._solar_curtail_kw[asset_id] = curtail_kw
        return {"accepted": True, "asset_id": asset_id, "curtail_pct": pct,
                "curtailed_kw": round(curtail_kw, 3),
                "lost_revenue_usd": round(self._PRICE_NOW * self._DUR_H * curtail_kw, 4),
                "dispatch_actions_used": self._dispatch_actions,
                "total_reduction_kw": round(self._total_reduction_kw(), 3)}

    # ── scoring ─────────────────────────────────────────────────────────────

    def _evaluate_impl(self) -> tuple[float, dict]:
        """Deterministic score in [0,1]:
          primary (0.55): signal met AND all constraints respected;
          cost-optimality (0.30): closeness of final cost to the LP reference;
          appropriateness (0.15): batteries before solar, no ineligible-B3 use.
        """
        total_red = self._total_reduction_kw()
        violations = self._active_violations()
        signal_met = total_red >= self._SIGNAL_KW - 1e-6
        constraints_ok = len(violations) == 0
        within_budget = self._dispatch_actions <= self._ACTION_BUDGET

        # Primary (0.55): full only if the signal is met with no violations.
        if signal_met and constraints_ok:
            primary = 0.55
        else:
            # partial credit toward the signal, gated by feasibility
            frac = max(0.0, min(1.0, total_red / self._SIGNAL_KW))
            primary = 0.55 * frac
            if not constraints_ok:
                primary *= 0.4  # heavy discount for an infeasible dispatch

        # Cost-optimality (0.30): only awarded if the signal is met. Closer to
        # (or below) the LP reference cost scores higher; runaway over-curtail
        # erodes it. Reference may be 0 only if batteries alone could cover the
        # signal — here they cannot, so _ref_cost > 0.
        cost = self._current_cost_usd()
        if signal_met and self._ref_cost > 1e-9:
            ratio = cost / self._ref_cost            # >= 1 when at/above optimum
            cost_credit = max(0.0, min(1.0, 1.0 - (ratio - 1.0)))
            cost_opt = 0.30 * cost_credit
        elif signal_met:
            cost_opt = 0.30 if cost <= 1e-9 else 0.0
        else:
            cost_opt = 0.0

        # Appropriateness (0.15):
        appropriateness = 0.0
        batt_kw = sum(self._batt_dispatch_kw.values())
        eligible_batt_kw = (self._initial_usable_kw("B1")
                            + self._initial_usable_kw("B2"))
        # +0.10 leaned on (most of) the free battery headroom before/with solar
        if eligible_batt_kw > 0 and batt_kw >= 0.9 * eligible_batt_kw:
            appropriateness += 0.10
        elif batt_kw > 0:
            appropriateness += 0.05
        # +0.05 did NOT attempt to dispatch ineligible B3
        if self._b3_dispatch_attempts == 0:
            appropriateness += 0.05

        score = primary + cost_opt + appropriateness
        if not within_budget:
            score = min(score, 0.5)   # over-budget can never reach a strong score
        score = max(0.0, min(1.0, score))

        return score, {
            "signal_kw": self._SIGNAL_KW,
            "total_reduction_kw": round(total_red, 3),
            "signal_met": bool(signal_met),
            "constraints_satisfied": bool(constraints_ok),
            "active_violations": violations,
            "dispatch_action_count": self._dispatch_actions,
            "within_action_budget": bool(within_budget),
            "battery_dispatch_kw": {k: round(v, 3) for k, v in self._batt_dispatch_kw.items()},
            "solar_curtail_kw": {k: round(v, 3) for k, v in self._solar_curtail_kw.items()},
            "final_cost_usd": round(cost, 4),
            "reference_cost_usd": self._ref_cost,
            "reference_dispatch_kw": self._ref_dispatch,
            "b3_dispatch_attempts": self._b3_dispatch_attempts,
            "log_entries": len(self._log),
            "primary_score": round(primary, 4),
            "cost_optimality_score": round(cost_opt, 4),
            "appropriateness_score": round(appropriateness, 4),
        }

    def _task_description(self) -> str:
        return (
            "You are an autonomous VPP operator managing a distributed-energy "
            "portfolio over a 4-hour window. A grid-operator curtailment signal "
            f"requests a {self._SIGNAL_KW:.0f} kW reduction in your net grid power "
            "exchange over the next 1-hour dispatch block, and the real-time price "
            f"has spiked to ${self._PRICE_NOW}/kWh. Meet the signal with your own "
            "resources in NO MORE THAN 3 dispatch actions.\n"
            "Battery discharge backs down grid draw and is revenue-neutral (the "
            "cheap, preferred lever); curtailing solar also reduces net exchange "
            "but loses revenue at the spot price (costly fallback). Each battery's "
            "usable discharge is energy-limited by its headroom above the 20% SoC "
            "floor: a battery at the floor (B3) has zero usable headroom and is "
            "ineligible. Inspect assets and prices first, prioritise free battery "
            "discharge, cover the remainder with the least-cost solar curtailment, "
            "verify with check_constraint_violation, log your decisions (including "
            "assets you chose NOT to dispatch), then call declare_done.")

    def _goal_state(self) -> dict:
        return {"net_exchange_reduction_kw_at_least": self._SIGNAL_KW,
                "max_dispatch_actions": self._ACTION_BUDGET,
                "soc_floor_pct": self._MIN_SOC_PCT,
                "preferred_order": ["battery_discharge", "solar_curtailment"],
                "ineligible_assets": ["B3"],
                "reference_cost_usd": self._ref_cost}


# ════════════════════════════════════════════════════════════════════════════
# vpp_scheduling_L4_002 — autonomous 24-hour Home Energy Management System
# ════════════════════════════════════════════════════════════════════════════

@DERBenchEnv.register("vpp_scheduling_L4_002")
class HEMS24hScheduleEnv(DERBenchEnv):
    """Build a cost-optimal, comfort-feasible 24-hour home energy schedule.

    Grounding
    ---------
    Reproduces the autonomous Home Energy Management System (HEMS) agent loop of
    AgenticHEMS (arXiv:2510.26603, "Agentic AI Home Energy Management System: A
    Large Language Model Framework for Residential Load Scheduling", El Makroum
    et al.; official code github.com/RedaElMakroum/agentic-ai-hems): an LLM
    agent perceives the home state, prices and the solar forecast, then plans a
    full-day schedule across the controllable loads (EV charging, home battery,
    deferrable appliances, HVAC) that minimises electricity cost while honouring
    hard comfort/deadline constraints and responding to a utility
    demand-response (DR) signal, graded against a MILP cost optimum as in the
    paper. Disclosed simplifications vs the paper: synthetic 3-tier TOU tariff
    (not real ENTSO-E prices), hourly (not 15-min) resolution, single agent (not
    orchestrator + specialists); the home battery, the 2R2C HVAC comfort band
    and the DR event are DER-Bench extensions beyond the paper's scope, with the
    comfort/occupancy framing informed by ContextLLM-BEMS (arXiv:2512.25055) and
    BehavioralGenerativeAgents (arXiv:2506.12664). AgenticHEMS is the
    directly-reproduced planning loop.

    Grounded reference (real solver, no magic constants)
    ----------------------------------------------------
    At reset the cost-optimal feasible 24-hour schedule is computed by a real
    `scipy.optimize.milp` (HiGHS) solve and stored as the grading reference. The
    formulation over the 24 hourly steps h = 0..23:

      Continuous vars (per hour): HVAC electrical power p_hvac_h ∈ [0, P_HVAC_MAX]
        (cooling thermal = COP·p), home-battery charge pch_h and discharge pdis_h
        ∈ [0, 3 kW], battery energy soc_h ∈ [0, 10] kWh, EV charge ev_h ∈
        [0, 7.2] kW, grid import gimp_h ≥ 0; plus DR reduction dr ∈ [0, 2] kW.
      Binary vars: dishwasher start-hour selection (one of the hours whose
        1.5 h cycle finishes < 08:00) and washing-machine start-hour selection
        (1.0 h cycle finishes < 18:00); each places a fixed power profile.
      Constraints:
        * battery SoC recursion soc_h = soc_{h-1} + η·pch_h − pdis_h/η, soc_0 from
          the 60 % initial SoC, bounds [0, 10] kWh;
        * EV energy: Σ_{j≤6} η·ev_j ≥ (80%−30%)·60 kWh = 30 kWh (reach 80 % by
          07:00), cumulative EV energy ≤ usable capacity;
        * indoor air temperature t_in_h ∈ [20, 24] °C for all h, where t_in is an
          AFFINE function of the HVAC-power vector built from the 2R2C RC model
          (see below) — the thermal state is linear in the HVAC energy, so the RC
          recursion is encoded exactly as linear equality/inequality constraints;
        * per-hour energy balance gimp_h = base_h + p_hvac_h + ev_h + appliance_h
          + pch_h − pdis_h − solar_h, with gimp_h ≥ 0 (no export credit);
        * DR credit dr ≤ pdis_17 and dr ≤ pdis_18 (the committed reduction must be
          backed by battery discharge during the DR window).
      Objective: minimise Σ_h gimp_h·price_h − incentive·|DR_HOURS|·dr (the DR
        credit is paid per kWh over each hour of the DR window, matching grading).

    Indoor-temperature model (2R2C, linear)
    ----------------------------------------
    The realized indoor-temperature trajectory — used BOTH to build the MILP
    thermal constraints and to grade the agent's HVAC setpoints — comes from a
    standard 2R2C lumped-capacitance building model
    (ISO 13790; Bacher & Madsen 2011, doi:10.1016/j.enbuild.2011.02.005):

        C_air dT_in/dt = (T_out−T_in)/R_ia + (T_m−T_in)/R_im + f·Q_sol + Q_int − COP·p
        C_m   dT_m/dt  = (T_in−T_m)/R_im + (1−f)·Q_sol

    integrated at 1/6-hour sub-steps. With the HVAC electrical power p held
    constant over each hour the state x_h = [T_in, T_m] is an AFFINE map of x_{h-1}
    and p_h, so the whole-day T_in trajectory is affine in the HVAC-power vector:
    t_in = M·p + d (M, d precomputed at reset). The free-running term d (p ≡ 0)
    peaks ≈ 26.3 °C on the hot summer day, so cooling is genuinely required and
    the [20, 24] band binds.

    What is real vs simplified
    --------------------------
    REAL: the 2R2C thermal dynamics and the affine T_in map; the SoC/EV energy
    recursions; the per-hour grid-energy balance with solar offset; the MILP
    reference cost (a true HiGHS optimum over all the above, never hand-picked);
    the realized-schedule re-simulation that grades the agent (a schedule that
    lets the home drift out of [20, 24] is caught by the RC trajectory).

    SIMPLIFIED: hourly resolution (no intra-hour ramping); HVAC is cooling-only on
    this summer day; the agent's HVAC setpoints are tracked by the env computing
    the per-hour cooling power needed to reach the setpoint (clamped to the cap);
    solar is consumed locally (no export settlement); appliance cycles are mapped
    onto the hourly grid (1.5 h dishwasher = 1.0 h + 0.5 h across two hours).
    """

    MAX_STEPS = 60

    # ── horizon + exogenous data (from the published task) ────────────────────
    _H = 24
    _OFF_PEAK = 0.08
    _MID_PEAK = 0.12
    _PEAK = 0.24
    _SOLAR_PEAK_KW = 4.2
    _SOLAR_PEAK_HOUR = 12

    # home battery
    _BATT_KWH = 10.0
    _BATT_SOC0_PCT = 60.0
    _BATT_MAX_KW = 3.0
    _BATT_ETA = 0.95

    # EV
    _EV_KWH = 60.0
    _EV_SOC0_PCT = 30.0
    _EV_TARGET_PCT = 80.0
    _EV_DEADLINE_H = 7           # must reach target by 07:00
    _EV_MAX_KW = 7.2
    _EV_ETA = 0.95

    # HVAC / comfort
    _T_MIN, _T_MAX = 20.0, 24.0
    _T_IN0, _T_MASS0 = 22.0, 22.5
    _HVAC_CAP_KW = 3.85          # electrical cap (cap_tons*3.517/cop)

    # appliances: name -> (duration_h, must_finish_before_hour, load_kw)
    _APPLIANCES = {
        "dishwasher": {"duration_h": 1.5, "before": 8, "load_kw": 1.2},
        "washing_machine": {"duration_h": 1.0, "before": 18, "load_kw": 0.8},
    }

    # DR event
    _DR_EVENT_ID = "DR-2024-0417"
    _DR_HOURS = (17, 18)
    _DR_REDUCTION_KW = 2.0
    _DR_INCENTIVE = 0.10         # $/kWh of committed reduction

    # 2R2C building parameters (ISO 13790-style defaults; authoritative values)
    _C_AIR, _C_M = 0.9, 18.0
    _R_IA, _R_IM = 4.5, 0.7
    _COP = 3.2
    _WIN_SPLIT = 0.4
    _F_WIN = 3.5
    _BASELOAD_KW = 0.55

    # ── exogenous-profile builders ────────────────────────────────────────────

    def _price_of(self, h: int) -> float:
        if 1 <= h <= 6:
            return self._OFF_PEAK
        if h in (17, 18, 19, 20):
            return self._PEAK
        return self._MID_PEAK

    def _solar_of(self, h: int) -> float:
        if h <= 6 or h >= 20:
            return 0.0
        # smooth bell curve peaking at _SOLAR_PEAK_KW at noon
        return round(self._SOLAR_PEAK_KW
                     * float(np.exp(-((h - self._SOLAR_PEAK_HOUR) ** 2) / (2 * 3.0 ** 2))), 4)

    def _build_exogenous(self):
        h = np.arange(self._H)
        self._prices = np.array([self._price_of(i) for i in range(self._H)])
        self._solar = np.array([self._solar_of(i) for i in range(self._H)])
        # warm summer day: outdoor temp peaks mid-afternoon (drives cooling need)
        self._t_out = 28.0 + 5.0 * np.cos((h - 15) / 24 * 2 * np.pi)
        # internal gains + solar thermal gain + non-HVAC baseload (deterministic)
        self._q_int = (0.3 + 0.5 * np.exp(-((h - 8) ** 2) / 6)
                       + 0.8 * np.exp(-((h - 19) ** 2) / 8))
        ghi = (self._solar / self._solar.max() * 800.0) if self._solar.max() > 0 else self._solar * 0.0
        self._q_sol = ghi / 1000.0 * self._F_WIN
        self._base = self._BASELOAD_KW * (0.6 + 0.5 * np.exp(-((h - 19) ** 2) / 10)
                                          + 0.3 * np.exp(-((h - 7) ** 2) / 6))

    # ── 2R2C affine thermal map: t_in = M @ p_hvac + d ───────────────────────

    def _step_maps(self, k: int):
        """Per-hour affine map x_k = A x_{k-1} + Bp·p_k + c (sub-stepped 2R2C)."""
        sub, dt = 6, 1.0 / 6
        A = np.eye(2)
        Bp = np.zeros(2)
        c = np.zeros(2)
        for _ in range(sub):
            J = np.array([
                [1 - dt / self._C_AIR * (1 / self._R_IA + 1 / self._R_IM),
                 dt / self._C_AIR * (1 / self._R_IM)],
                [dt / self._C_M * (1 / self._R_IM),
                 1 - dt / self._C_M * (1 / self._R_IM)],
            ])
            bp = np.array([-dt / self._C_AIR * self._COP, 0.0])
            cc = np.array([
                dt / self._C_AIR * (self._t_out[k] / self._R_IA
                                    + self._WIN_SPLIT * self._q_sol[k] + self._q_int[k]),
                dt / self._C_M * ((1 - self._WIN_SPLIT) * self._q_sol[k]),
            ])
            A = J @ A
            Bp = J @ Bp + bp
            c = J @ c + cc
        return A, Bp, c

    def _build_thermal_map(self):
        """Precompute t_in[k] = _t_in_M[k] @ p + _t_in_d[k] (p = HVAC-power vector).

        Convention (IDENTICAL to `_realized_hvac_power`): the state vector
        [T_IN0, T_MASS0] is the home state at the START of hour 0, and hour k
        applies exactly ONE 2R2C step `_step_maps(k)`, so t_in[k] is the
        end-of-hour-k indoor-air temperature. Feeding the SAME HVAC-power vector
        through this affine map and through the realized simulator therefore
        yields the identical T_in trajectory (verified to < 1e-6)."""
        x0 = np.array([self._T_IN0, self._T_MASS0])
        M = []      # M[k]: 2 x H sensitivity of x_k to the HVAC-power vector
        d = []      # d[k]: 2-vector free-running (p ≡ 0) state at hour k
        prevM = np.zeros((2, self._H))
        prevd = x0
        for k in range(self._H):
            A, Bp, c = self._step_maps(k)
            Mk = A @ prevM
            Mk[:, k] += Bp
            dk = A @ prevd + c
            M.append(Mk)
            d.append(dk)
            prevM, prevd = Mk, dk
        self._t_in_M = np.array([M[k][0] for k in range(self._H)])   # H x H
        self._t_in_d = np.array([d[k][0] for k in range(self._H)])   # H

    def _simulate_temp(self, p_hvac: np.ndarray) -> np.ndarray:
        """Realized indoor-air trajectory for an HVAC-power vector (affine map)."""
        return self._t_in_M @ p_hvac + self._t_in_d

    # ── reference MILP ────────────────────────────────────────────────────────

    def _solve_reference(self):
        """Cost-optimal feasible 24h schedule via scipy.optimize.milp (HiGHS).

        Returns (reference_cost_usd, reference_summary_dict)."""
        H = self._H
        dish = self._APPLIANCES["dishwasher"]
        wash = self._APPLIANCES["washing_machine"]
        # feasible integer start hours (cycle finishes strictly before deadline)
        dish_starts = [s for s in range(H) if s + dish["duration_h"] <= dish["before"]]
        wash_starts = [s for s in range(H) if s + wash["duration_h"] <= wash["before"]]
        nd, nw = len(dish_starts), len(wash_starts)

        idx = 0

        def alloc(n):
            nonlocal idx
            s = slice(idx, idx + n)
            idx += n
            return s

        P = alloc(H); PCH = alloc(H); PDIS = alloc(H); SOC = alloc(H)
        EV = alloc(H); GIMP = alloc(H); DR = alloc(1); DISH = alloc(nd); WASH = alloc(nw)
        N = idx

        c = np.zeros(N)
        c[GIMP] = self._prices
        # DR is a single committed-reduction power (kW) credited over the whole
        # DR window; value it IDENTICALLY to the grader, which pays the incentive
        # per kWh over each of the len(_DR_HOURS) hours (here 2 h).
        c[DR] = -self._DR_INCENTIVE * len(self._DR_HOURS)
        integrality = np.zeros(N)
        integrality[DISH] = 1
        integrality[WASH] = 1

        lb = np.zeros(N)
        ub = np.full(N, np.inf)
        ub[P] = self._HVAC_CAP_KW
        ub[PCH] = self._BATT_MAX_KW
        ub[PDIS] = self._BATT_MAX_KW
        ub[SOC] = self._BATT_KWH
        ub[EV] = self._EV_MAX_KW
        ub[GIMP] = 50.0
        ub[DR] = self._DR_REDUCTION_KW
        ub[DISH] = 1
        ub[WASH] = 1

        cons = []

        def add(A, lo, hi):
            cons.append(LinearConstraint(A, lo, hi))

        soc0 = self._BATT_SOC0_PCT / 100.0 * self._BATT_KWH
        eta = self._BATT_ETA
        for k in range(H):
            A = np.zeros(N)
            A[SOC.start + k] = 1
            if k > 0:
                A[SOC.start + k - 1] = -1
            A[PCH.start + k] = -eta
            A[PDIS.start + k] = 1 / eta
            rhs = soc0 if k == 0 else 0.0
            add(A, rhs, rhs)

        # EV: reach >= target by the deadline hour; never exceed capacity.
        ev_need = (self._EV_TARGET_PCT - self._EV_SOC0_PCT) / 100.0 * self._EV_KWH
        ev_room = (100.0 - self._EV_SOC0_PCT) / 100.0 * self._EV_KWH
        A = np.zeros(N)
        for j in range(self._EV_DEADLINE_H):
            A[EV.start + j] = self._EV_ETA
        add(A, ev_need, np.inf)
        for k in range(H):
            A = np.zeros(N)
            for j in range(k + 1):
                A[EV.start + j] = self._EV_ETA
            add(A, -np.inf, ev_room)

        # comfort band on the realized air temperature
        for k in range(H):
            A = np.zeros(N)
            A[P] = self._t_in_M[k]
            add(A, self._T_MIN - self._t_in_d[k], self._T_MAX - self._t_in_d[k])

        # exactly one appliance start each
        A = np.zeros(N); A[DISH] = 1; add(A, 1, 1)
        A = np.zeros(N); A[WASH] = 1; add(A, 1, 1)

        dish_load = self._appliance_hourly_load("dishwasher", dish_starts)   # H x nd
        wash_load = self._appliance_hourly_load("washing_machine", wash_starts)

        # per-hour energy balance
        for k in range(H):
            A = np.zeros(N)
            A[GIMP.start + k] = 1
            A[P.start + k] = -1
            A[EV.start + k] = -1
            A[PCH.start + k] = -1
            A[PDIS.start + k] = 1
            A[DISH] = -dish_load[k]
            A[WASH] = -wash_load[k]
            rhs = float(self._base[k] - self._solar[k])
            add(A, rhs, rhs)

        # DR reduction must be backed by battery discharge during the DR window
        for h in self._DR_HOURS:
            A = np.zeros(N)
            A[DR] = 1
            A[PDIS.start + h] = -1
            add(A, -np.inf, 0)

        res = milp(c=c, constraints=cons, integrality=integrality, bounds=Bounds(lb, ub))
        if not res.success:
            raise RuntimeError(f"HEMS reference MILP failed to converge: {res.message}")
        x = res.x
        soc0_ev = self._EV_SOC0_PCT / 100.0 * self._EV_KWH
        ev_soc7 = soc0_ev + self._EV_ETA * float(np.sum(x[EV][:self._EV_DEADLINE_H]))
        t_in = self._simulate_temp(x[P])
        summary = {
            "dishwasher_start_hour": dish_starts[int(np.argmax(x[DISH]))],
            "washing_machine_start_hour": wash_starts[int(np.argmax(x[WASH]))],
            "ev_soc_pct_by_hour7": round(ev_soc7 / self._EV_KWH * 100.0, 2),
            "dr_reduction_kw": round(float(x[DR][0]), 3),
            "min_indoor_temp_C": round(float(t_in.min()), 3),
            "max_indoor_temp_C": round(float(t_in.max()), 3),
            "grid_import_kwh": round(float(np.sum(x[GIMP])), 3),
            "battery_charge_hours": [h for h in range(H) if x[PCH][h] > 0.05],
            "battery_discharge_hours": [h for h in range(H) if x[PDIS][h] > 0.05],
        }
        return round(float(res.fun), 4), summary

    def _naive_baseline_cost(self) -> float:
        """Cost of a feasible-but-unoptimised schedule, computed (not hand-picked)
        by re-simulating a naive policy through the SAME realized-schedule model:
        EV charged off-peak (forced by the 07:00 deadline), appliances at the
        earliest deadline-feasible hour, HVAC tracking the comfort upper bound, and
        NO home-battery arbitrage or DR participation. This is the zero-credit
        anchor for cost-optimality; the MILP optimum is the full-credit anchor, so
        the cost score measures the fraction of the achievable savings the agent
        actually captured (battery arbitrage, DR, smart appliance/EV placement)."""
        saved = (self._battery_mode, self._ev_plan, self._appliance_plan,
                 self._hvac_setpoint, self._dr_response)
        self._battery_mode = ["hold"] * self._H
        self._ev_plan = (1, self._EV_DEADLINE_H, self._EV_TARGET_PCT)
        self._appliance_plan = {
            "dishwasher": float(self._APPLIANCES["dishwasher"]["before"]
                                - self._APPLIANCES["dishwasher"]["duration_h"]),
            "washing_machine": float(self._APPLIANCES["washing_machine"]["before"]
                                     - self._APPLIANCES["washing_machine"]["duration_h"]),
        }
        self._hvac_setpoint = [self._T_MAX] * self._H
        self._dr_response = None
        cost = self._realized_schedule()["net_cost_usd"]
        (self._battery_mode, self._ev_plan, self._appliance_plan,
         self._hvac_setpoint, self._dr_response) = saved
        return cost

    def _appliance_hourly_load(self, name: str, starts: list) -> np.ndarray:
        """Column-per-start matrix (H x len(starts)) of hourly kW for an appliance.

        A duration that crosses an hour boundary (1.5 h) places full load in the
        first whole hour and the fractional remainder in the next hour."""
        spec = self._APPLIANCES[name]
        dur, load = spec["duration_h"], spec["load_kw"]
        L = np.zeros((self._H, len(starts)))
        for i, s in enumerate(starts):
            remaining = dur
            h = s
            while remaining > 1e-9 and h < self._H:
                frac = min(1.0, remaining)
                L[h, i] += load * frac
                remaining -= frac
                h += 1
        return L

    # ── env lifecycle ───────────────────────────────────────────────────────

    def _reset_impl(self) -> dict:
        self._build_exogenous()
        self._build_thermal_map()
        self._ref_cost, self._ref_summary = self._solve_reference()

        # agent schedule state (all empty / OFF initially)
        self._battery_mode = ["hold"] * self._H        # charge / discharge / hold
        self._ev_plan = None                            # (start, end, target_pct)
        self._appliance_plan = {}                       # name -> start_hour (float)
        self._hvac_setpoint = [None] * self._H          # °C or None (OFF)
        self._dr_response = None                        # committed reduction kW
        self._log: list[dict] = []
        self._declared = False

        # zero-credit cost anchor (computed via the realized-schedule simulator)
        self._naive_cost = self._naive_baseline_cost()

        return {
            "scenario": "Autonomous HEMS — plan a cost-optimal, comfort-feasible "
                        "24-hour home energy schedule (hours 0-23) and respond to "
                        "demand-response event " + self._DR_EVENT_ID + ".",
            "current_time": "00:00",
            "home_battery": {"soc_pct": self._BATT_SOC0_PCT, "capacity_kwh": self._BATT_KWH,
                             "max_charge_discharge_kw": self._BATT_MAX_KW},
            "ev_battery": {"soc_pct": self._EV_SOC0_PCT, "capacity_kwh": self._EV_KWH,
                           "charger_max_kw": self._EV_MAX_KW, "target_soc_pct": self._EV_TARGET_PCT,
                           "departure_hour": self._EV_DEADLINE_H},
            "indoor_temp_C": self._T_IN0,
            "comfort_band_C": [self._T_MIN, self._T_MAX],
            "tariff": {"off_peak_rate": self._OFF_PEAK, "mid_peak_rate": self._MID_PEAK,
                       "peak_rate": self._PEAK},
            "appliances": {n: {"duration_h": s["duration_h"],
                               "must_complete_before_hour": s["before"], "load_kw": s["load_kw"]}
                           for n, s in self._APPLIANCES.items()},
            "dr_event": {"event_id": self._DR_EVENT_ID, "hours": list(self._DR_HOURS),
                         "reduction_kw": self._DR_REDUCTION_KW,
                         "incentive_per_kwh": self._DR_INCENTIVE},
        }

    # ── realized-schedule simulation (the grading model) ──────────────────────

    def _realized_hvac_power(self):
        """Compute the realized HVAC electrical power per hour from the agent's
        setpoints, plus the resulting indoor-air trajectory.

        For each hour the env computes the cooling power needed so the air-node
        temperature lands on the requested setpoint (clamped to [0, cap]); an hour
        with no setpoint (OFF) uses 0 kW. Cooling-only: if the home is already at
        or below the setpoint, no power is drawn. Solved sequentially because each
        hour's required power depends on the realized state of earlier hours."""
        p = np.zeros(self._H)
        x = np.array([self._T_IN0, self._T_MASS0])
        t_in = np.zeros(self._H)
        for k in range(self._H):
            A, Bp, c = self._step_maps(k)
            sp = self._hvac_setpoint[k]
            if sp is None:
                pk = 0.0
            else:
                # air-node temp as a function of p_k: ti(p) = (A x)[0] + Bp[0]*p + c[0]
                base_ti = float((A @ x)[0] + c[0])
                slope = float(Bp[0])                       # < 0 (cooling lowers temp)
                # power to reach setpoint exactly; clamp to [0, cap]
                if slope < -1e-12 and base_ti > sp:
                    pk = (sp - base_ti) / slope
                else:
                    pk = 0.0
                pk = max(0.0, min(self._HVAC_CAP_KW, pk))
            x = A @ x + Bp * pk + c
            p[k] = pk
            t_in[k] = x[0]
        return p, t_in

    def _realized_schedule(self):
        """Re-simulate the agent's full schedule into hourly energy, SoC and
        temperature trajectories, returning a dict of realized quantities."""
        H = self._H
        p_hvac, t_in = self._realized_hvac_power()

        # EV charging trajectory
        ev = np.zeros(H)
        ev_soc_kwh = self._EV_SOC0_PCT / 100.0 * self._EV_KWH
        ev_cap = self._EV_KWH
        ev_soc_traj = []
        if self._ev_plan is not None:
            start, end, target_pct = self._ev_plan
            target_kwh = min(ev_cap, target_pct / 100.0 * ev_cap)
            for h in range(H):
                charge = 0.0
                if start <= h < end and ev_soc_kwh < target_kwh - 1e-9:
                    headroom = (target_kwh - ev_soc_kwh) / self._EV_ETA
                    charge = min(self._EV_MAX_KW, headroom)
                    ev_soc_kwh += charge * self._EV_ETA
                ev[h] = charge
                ev_soc_traj.append(ev_soc_kwh)
        else:
            ev_soc_traj = [ev_soc_kwh] * H
        ev_soc_pct_h7 = ev_soc_traj[self._EV_DEADLINE_H - 1] / ev_cap * 100.0

        # home-battery trajectory
        pch = np.zeros(H); pdis = np.zeros(H)
        soc = self._BATT_SOC0_PCT / 100.0 * self._BATT_KWH
        soc_traj = []
        eta = self._BATT_ETA
        for h in range(H):
            mode = self._battery_mode[h]
            if mode == "charge":
                room = (self._BATT_KWH - soc) / eta
                c = min(self._BATT_MAX_KW, room)
                pch[h] = c
                soc += c * eta
            elif mode == "discharge":
                avail = soc * eta
                dpow = min(self._BATT_MAX_KW, avail)
                pdis[h] = dpow
                soc -= dpow / eta
            soc_traj.append(soc)

        # appliance hourly load
        appl = np.zeros(H)
        for name, start in self._appliance_plan.items():
            spec = self._APPLIANCES[name]
            remaining = spec["duration_h"]
            h = int(start)
            while remaining > 1e-9 and h < H:
                frac = min(1.0, remaining)
                appl[h] += spec["load_kw"] * frac
                remaining -= frac
                h += 1

        # per-hour grid import (no export credit)
        gimp = np.maximum(0.0, self._base + p_hvac + ev + appl + pch - pdis - self._solar)

        cost = float(np.sum(gimp * self._prices))
        # DR incentive: credited only if the committed reduction is backed by
        # battery discharge during the DR window.
        dr_committed = self._dr_response or 0.0
        dr_backed = min(dr_committed, min(pdis[h] for h in self._DR_HOURS)) \
            if self._dr_response is not None else 0.0
        dr_incentive = self._DR_INCENTIVE * dr_backed * len(self._DR_HOURS)
        net_cost = cost - dr_incentive

        return {
            "hourly_grid_import_kw": gimp,
            "hourly_hvac_kw": p_hvac,
            "hourly_ev_kw": ev,
            "battery_discharge_kw": pdis,
            "indoor_temp_C": t_in,
            "ev_soc_pct_by_hour7": ev_soc_pct_h7,
            "min_indoor_temp_C": float(t_in.min()),
            "max_indoor_temp_C": float(t_in.max()),
            "gross_cost_usd": cost,
            "dr_incentive_usd": dr_incentive,
            "dr_backed_kw": dr_backed,
            "net_cost_usd": net_cost,
        }

    def _hard_constraints(self, realized: dict) -> dict:
        ev_ok = realized["ev_soc_pct_by_hour7"] >= self._EV_TARGET_PCT - 1e-6
        comfort_ok = (realized["min_indoor_temp_C"] >= self._T_MIN - 1e-3
                      and realized["max_indoor_temp_C"] <= self._T_MAX + 1e-3)
        dish_ok = ("dishwasher" in self._appliance_plan and
                   self._appliance_plan["dishwasher"] + self._APPLIANCES["dishwasher"]["duration_h"]
                   <= self._APPLIANCES["dishwasher"]["before"] + 1e-9)
        wash_ok = ("washing_machine" in self._appliance_plan and
                   self._appliance_plan["washing_machine"] + self._APPLIANCES["washing_machine"]["duration_h"]
                   <= self._APPLIANCES["washing_machine"]["before"] + 1e-9)
        return {"ev_target_met": bool(ev_ok), "comfort_band_met": bool(comfort_ok),
                "dishwasher_deadline_met": bool(dish_ok),
                "washing_machine_deadline_met": bool(wash_ok)}

    # ── tools ─────────────────────────────────────────────────────────────────

    def _tools(self) -> list[dict]:
        return [
            {"name": "get_home_state",
             "description": "Current home battery SoC (%), EV SoC (%), indoor temperature "
                            "(°C) and occupancy.",
             "parameters": {}},
            {"name": "get_price_forecast_24h",
             "description": "24 hourly electricity prices ($/kWh) for hours 0-23.",
             "parameters": {}},
            {"name": "get_solar_forecast_24h",
             "description": "24 hourly PV generation forecasts (kW) for hours 0-23.",
             "parameters": {}},
            {"name": "schedule_appliance",
             "description": "Schedule a flexible appliance ('dishwasher' / "
                            "'washing_machine') to start at start_hour for duration_h "
                            "hours. The deadline (dishwasher < 08:00, washer < 18:00) is "
                            "a hard constraint checked at grading.",
             "parameters": {"appliance": {"type": "string"},
                            "start_hour": {"type": "number"},
                            "duration_h": {"type": "number"}}},
            {"name": "set_battery_mode",
             "description": "Set the home-battery mode for one hour: 'charge', "
                            "'discharge' or 'hold' (max 3 kW, 0-10 kWh).",
             "parameters": {"hour": {"type": "integer"},
                            "mode": {"type": "string", "enum": ["charge", "discharge", "hold"]}}},
            {"name": "set_ev_charging",
             "description": "Program the EV charger to charge from start_hour to end_hour "
                            "(exclusive), targeting target_soc_pct (max 7.2 kW). The EV "
                            "must reach 80% by 07:00 (hard constraint).",
             "parameters": {"start_hour": {"type": "integer"},
                            "end_hour": {"type": "integer"},
                            "target_soc_pct": {"type": "number"}}},
            {"name": "set_hvac",
             "description": "Set the HVAC indoor setpoint (°C) for an [start, end) hour "
                            "range. Hours with no setpoint leave HVAC OFF. Indoor "
                            "temperature must stay within [20, 24] °C at all times (hard "
                            "constraint), checked against the home's 2R2C thermal model.",
             "parameters": {"hour_range": {"type": "array"},
                            "setpoint_C": {"type": "number"}}},
            {"name": "send_dr_response",
             "description": "Respond to the demand-response event with a committed load "
                            "reduction (kW) over the DR window (hours 17-18). The reduction "
                            "must be backed by battery discharge during those hours to earn "
                            "the $0.10/kWh incentive.",
             "parameters": {"event_id": {"type": "string"},
                            "reduction_kw": {"type": "number"}}},
            {"name": "log_decision",
             "description": "Record a scheduling decision with its cost/comfort rationale.",
             "parameters": {"action": {"type": "string"},
                            "justification": {"type": "string"}}},
            {"name": "declare_done",
             "description": "End the episode; the realized 24-hour schedule is then graded.",
             "parameters": {}},
        ]

    def _dispatch_tool(self, tool: str, args: dict):
        if tool == "get_home_state":
            return {"battery_soc_pct": self._BATT_SOC0_PCT, "ev_soc_pct": self._EV_SOC0_PCT,
                    "indoor_temp_C": self._T_IN0, "occupancy": True}
        if tool == "get_price_forecast_24h":
            return {"prices_per_kwh": [round(float(p), 4) for p in self._prices]}
        if tool == "get_solar_forecast_24h":
            return {"solar_kw": [round(float(s), 4) for s in self._solar]}
        if tool == "schedule_appliance":
            return self._tool_schedule_appliance(
                str(args["appliance"]), float(args["start_hour"]),
                float(args.get("duration_h", 0)))
        if tool == "set_battery_mode":
            return self._tool_set_battery_mode(int(args["hour"]), str(args["mode"]).lower())
        if tool == "set_ev_charging":
            return self._tool_set_ev_charging(
                int(args["start_hour"]), int(args["end_hour"]), float(args["target_soc_pct"]))
        if tool == "set_hvac":
            return self._tool_set_hvac(args["hour_range"], float(args["setpoint_C"]))
        if tool == "send_dr_response":
            return self._tool_send_dr(str(args.get("event_id", "")), float(args["reduction_kw"]))
        if tool == "log_decision":
            self._log.append({"action": args.get("action", ""),
                              "justification": args.get("justification", "")})
            return {"logged": True, "log_entries": len(self._log)}
        if tool == "declare_done":
            self._declared = True
            self._done = True
            realized = self._realized_schedule()
            hard = self._hard_constraints(realized)
            return {"status": "episode_complete",
                    "estimated_net_cost_usd": round(realized["net_cost_usd"], 4),
                    "hard_constraints": hard}
        raise ValueError(f"Unknown tool: {tool}")

    def _tool_schedule_appliance(self, name: str, start_hour: float, duration_h: float):
        if name not in self._APPLIANCES:
            return {"rejected": True, "reason": f"unknown appliance {name!r} "
                    f"(valid: {list(self._APPLIANCES)})"}
        spec = self._APPLIANCES[name]
        finish = start_hour + spec["duration_h"]
        meets_deadline = finish <= spec["before"] + 1e-9
        self._appliance_plan[name] = start_hour
        return {"accepted": True, "appliance": name, "start_hour": start_hour,
                "finish_hour": round(finish, 3), "must_finish_before": spec["before"],
                "meets_deadline": bool(meets_deadline)}

    def _tool_set_battery_mode(self, hour: int, mode: str):
        if mode not in ("charge", "discharge", "hold"):
            return {"rejected": True, "reason": f"invalid mode {mode!r}"}
        if not (0 <= hour < self._H):
            return {"rejected": True, "reason": f"hour {hour} out of range 0-23"}
        self._battery_mode[hour] = mode
        return {"accepted": True, "hour": hour, "mode": mode}

    def _tool_set_ev_charging(self, start: int, end: int, target_pct: float):
        start = max(0, min(self._H, start))
        end = max(0, min(self._H, end))
        if end <= start:
            return {"rejected": True, "reason": "end_hour must be after start_hour"}
        target_pct = max(0.0, min(100.0, target_pct))
        self._ev_plan = (start, end, target_pct)
        # report the energy / time feasibility for the agent
        need_kwh = max(0.0, (target_pct - self._EV_SOC0_PCT) / 100.0 * self._EV_KWH)
        hours_at_max = need_kwh / self._EV_ETA / self._EV_MAX_KW
        return {"accepted": True, "start_hour": start, "end_hour": end,
                "target_soc_pct": target_pct, "energy_needed_kwh": round(need_kwh, 3),
                "hours_at_max_power": round(hours_at_max, 3),
                "window_hours": end - start}

    def _tool_set_hvac(self, hour_range, setpoint_C: float):
        try:
            lo, hi = int(hour_range[0]), int(hour_range[1])
        except (TypeError, ValueError, IndexError):
            return {"rejected": True, "reason": "hour_range must be [start, end]"}
        lo = max(0, min(self._H, lo))
        hi = max(0, min(self._H, hi))
        for h in range(lo, hi):
            self._hvac_setpoint[h] = setpoint_C
        return {"accepted": True, "hours": list(range(lo, hi)), "setpoint_C": setpoint_C}

    def _tool_send_dr(self, event_id: str, reduction_kw: float):
        if event_id and event_id != self._DR_EVENT_ID:
            return {"rejected": True, "reason": f"unknown event {event_id!r} "
                    f"(active event: {self._DR_EVENT_ID})"}
        reduction_kw = max(0.0, reduction_kw)
        self._dr_response = reduction_kw
        return {"accepted": True, "event_id": self._DR_EVENT_ID,
                "committed_reduction_kw": reduction_kw,
                "note": "credited only if backed by battery discharge during hours 17-18"}

    # ── scoring ─────────────────────────────────────────────────────────────

    def _evaluate_impl(self) -> tuple[float, dict]:
        """Deterministic score in [0,1]:
          HARD CONSTRAINTS (gating, 0.55): EV>=80% by 07:00, indoor temp in
            [20,24] all hours, both appliance deadlines met — each worth an equal
            share, and the cost/DR credit below is heavily discounted while any is
            violated, so a cheap-but-infeasible schedule cannot score high.
          COST-OPTIMALITY (0.35): closeness of the realized net cost to the MILP
            reference cost (full credit at or below the optimum).
          DR RESPONSE (0.10): committed reduction backed by battery discharge.
        """
        realized = self._realized_schedule()
        hard = self._hard_constraints(realized)
        hard_flags = [hard["ev_target_met"], hard["comfort_band_met"],
                      hard["dishwasher_deadline_met"], hard["washing_machine_deadline_met"]]
        n_hard = len(hard_flags)
        n_met = sum(hard_flags)
        all_hard_ok = n_met == n_hard

        # Hard-constraint block (0.55): proportional, but only full when ALL met.
        hard_score = 0.55 * (n_met / n_hard)

        # Cost-optimality (0.35): the realized net cost is scored on the computed
        # band [MILP optimum (full credit), naive-feasible baseline (zero credit)],
        # i.e. the FRACTION of the achievable savings the agent captured. Both
        # anchors are real solver/simulator outputs. Beating the MILP optimum
        # (only possible via an infeasible shortcut) is clamped to full credit.
        net_cost = realized["net_cost_usd"]
        ref = self._ref_cost
        span = self._naive_cost - ref
        if span > 1e-9:
            cost_credit = (self._naive_cost - net_cost) / span
        else:
            cost_credit = 1.0 if net_cost <= ref + 1e-9 else 0.0
        cost_credit = max(0.0, min(1.0, cost_credit))
        if not all_hard_ok:
            cost_credit *= 0.2   # an infeasible schedule cannot bank cost credit
        cost_score = 0.35 * cost_credit

        # DR response (0.10): committed reduction backed by battery discharge.
        dr_credit = min(1.0, realized["dr_backed_kw"] / self._DR_REDUCTION_KW) \
            if self._DR_REDUCTION_KW > 0 else 0.0
        dr_score = 0.10 * dr_credit
        if not all_hard_ok:
            dr_score *= 0.3   # DR credit barely counts on an infeasible schedule

        score = hard_score + cost_score + dr_score
        score = max(0.0, min(1.0, score))

        return score, {
            "reference_cost_usd": self._ref_cost,
            "naive_baseline_cost_usd": round(self._naive_cost, 4),
            "reference_schedule": self._ref_summary,
            "realized_schedule": {
                "net_cost_usd": round(realized["net_cost_usd"], 4),
                "gross_cost_usd": round(realized["gross_cost_usd"], 4),
                "dr_incentive_usd": round(realized["dr_incentive_usd"], 4),
                "ev_soc_pct_by_hour7": round(realized["ev_soc_pct_by_hour7"], 2),
                "min_indoor_temp_C": round(realized["min_indoor_temp_C"], 3),
                "max_indoor_temp_C": round(realized["max_indoor_temp_C"], 3),
                "dr_backed_kw": round(realized["dr_backed_kw"], 3),
            },
            "hard_constraints": hard,
            "hard_constraints_met": f"{n_met}/{n_hard}",
            "all_hard_constraints_satisfied": bool(all_hard_ok),
            "log_entries": len(self._log),
            "hard_constraint_score": round(hard_score, 4),
            "cost_optimality_score": round(cost_score, 4),
            "dr_response_score": round(dr_score, 4),
        }

    def _task_description(self) -> str:
        return (
            "You are an autonomous Home Energy Management System (HEMS) agent. It is "
            "00:00; plan a complete 24-hour (hours 0-23) operating schedule for a smart "
            "home that MINIMISES electricity cost while satisfying every hard comfort "
            "constraint and responding to demand-response event " + self._DR_EVENT_ID + ".\n"
            "Assets: a 10 kWh home battery (SoC 60%, ±3 kW), a 60 kWh EV (SoC 30%, "
            "charger 7.2 kW), HVAC, and two deferrable appliances (dishwasher 1.5 h, "
            "washer 1 h).\n"
            "HARD constraints (violating any sharply lowers your score): EV must reach "
            "80% SoC by 07:00; indoor temperature must stay within [20, 24] °C every "
            "hour; the dishwasher must finish before 08:00 and the washer before 18:00.\n"
            "Tariff: off-peak $0.08 (01:00-06:00), peak $0.24 (17:00-20:00), mid-peak "
            "$0.12 otherwise. Solar peaks at 4.2 kW at noon. A DR event requests a 2 kW "
            "reduction over hours 17-18 for a $0.10/kWh incentive.\n"
            "Inspect the home state, prices and solar first; charge the EV off-peak; "
            "charge the battery on midday solar and discharge it into the peak; place "
            "appliances in cheap deadline-feasible windows; keep HVAC inside the comfort "
            "band (pre-cooling before the peak helps); respond to the DR event with "
            "battery-backed reduction; log your decisions, then call declare_done.")

    def _goal_state(self) -> dict:
        return {"horizon_hours": self._H,
                "ev_target_soc_pct_by_hour7": self._EV_TARGET_PCT,
                "comfort_band_C": [self._T_MIN, self._T_MAX],
                "appliance_deadlines": {n: s["before"] for n, s in self._APPLIANCES.items()},
                "dr_event": {"event_id": self._DR_EVENT_ID, "reduction_kw": self._DR_REDUCTION_KW},
                "reference_cost_usd": self._ref_cost}
