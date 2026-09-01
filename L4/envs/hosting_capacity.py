"""
hosting_capacity_L4_001 — autonomous PV hosting-capacity search on a real
IEEE 33-bus radial distribution feeder.

Grounding
---------
Reproduces the simulation-driven PV hosting-capacity (HC) analysis loop used in
utility distribution planning: iteratively grow the PV injection at a candidate
point of interconnection, re-solve the AC power flow, and find the largest size
for which every steady-state operating limit (ANSI Range A voltage band
[0.95, 1.05] pu and the conductor thermal limit) is still satisfied, then report
the binding constraint. The streamlined / iterative HC methodology follows
EPRI's DRIVE (Distribution Resource Integration and Value Estimation) method and
NREL's published hosting-capacity analyses for PV-rich radial circuits, both of
which establish that on a lightly-loaded end-of-feeder radial circuit the
steady-state OVERVOLTAGE limit at the interconnection bus is reached before the
substation-getaway thermal limit.

The agentic power-flow-tool loop (perceive limits -> drive a power-flow tool to
bracket and bisect the capacity -> verify -> report) is the methodology of
PFAgent (arXiv:2604.10846, the agentic power-flow basis also referenced by the
sibling grid_edge_control env).

Network: pandapower `case33bw` (Thurner et al., IEEE TPWRS 2018) — the standard
12.66 kV radial distribution feeder, 33 buses, 32 in-service lines.

Index mapping (task 1-indexed prose -> pandapower 0-indexed objects)
--------------------------------------------------------------------
The task prompt names buses/lines 1-indexed; case33bw is 0-indexed, so task
"Bus N" == pandapower bus N-1:
  * task "Bus 1"  (substation / slack)  -> pandapower bus 0  (the ext_grid bus)
  * task "Bus 18" (far end of the main feeder, point of interconnection)
                                          -> pandapower bus 17
  * task "Line 1-2" (substation getaway) -> pandapower line 0 (bus0 -> bus1)
The env accepts the task's 1-indexed bus ids on its tools and translates
internally; tool results report back in the task's 1-indexed convention.

Grounded reference (real solver, no magic constants)
----------------------------------------------------
At reset the env runs its OWN fine bisection on a live `pp.runpp` to compute the
exact maximum unity-power-factor PV MW at Bus 18 for which all bus voltages are
<= 1.05 pu AND all line loadings are <= 100% (resolved to <= 0.05 MW), and
records which limit binds first as PV grows and where. These are the grading
reference — the agent must rediscover them. With case33bw scaled to the stated
midday light-load condition this yields HC ~= 1.13 MW, binding constraint =
OVERVOLTAGE at Bus 18, with the getaway thermal loading still well under 20% at
the capacity point (voltage rise binds first, exactly as the HC literature
predicts for an end-of-feeder radial injection).

What is real vs simplified
--------------------------
REAL: the network, the PV `sgen` injection, EVERY voltage and line-loading value
(all from a live `pp.runpp` on the mutated net, never hardcoded), the voltage
rise with injection, the reactive-power sensitivity (an absorbing power factor
genuinely lowers the Bus 18 voltage via a negative-q sgen), and the bisected HC
reference.

SIMPLIFIED: snapshot HC at a single study condition (the midday light-load
operating point, load scaled to ~30% of nameplate) rather than a full 8760-hour
sweep; the reported answer is the unity-power-factor capacity (Volt-VAr control
is offered only as a logged sensitivity); the substation is the slack ext_grid
held at 1.00 pu rather than an explicit regulator; and the conductor thermal
rating is an engineered-but-realistic `max_i_ka` set so the ~6.0 MVA getaway
limit in the prose is represented (it never binds here, by design — voltage
does).
"""
import warnings

warnings.filterwarnings("ignore")
import math

import pandapower as pp
import pandapower.networks as pn

from derbench_env import DERBenchEnv


@DERBenchEnv.register("hosting_capacity_L4_001")
class IEEE33HostingCapacityEnv(DERBenchEnv):
    """Find the max unity-PF PV MW at Bus 18 before the first limit is violated."""

    MAX_STEPS = 40

    # ── study condition (engineered against real case33bw physics) ────────────
    _LOAD_SCALE = 0.30            # midday light load (~30% of nameplate)
    _NOMINAL_KV = 12.66
    _GETAWAY_MVA = 6.0            # task-stated substation-getaway thermal rating
    _GETAWAY_LINE = 0             # pandapower line 0 == task "Line 1-2"
    _SLACK_BUS = 0               # pandapower bus 0 == task "Bus 1" (substation slack)
    _PCC_BUS = 17                # pandapower bus 17 == task "Bus 18" (interconnection)
    _VMIN, _VMAX = 0.95, 1.05
    _LOADING_LIMIT = 100.0
    _HC_TOL_MW = 0.05            # reference bisection resolution

    # task 1-indexed <-> pandapower 0-indexed bus mapping
    def _to_pp_bus(self, task_bus: int) -> int:
        return int(task_bus) - 1

    def _to_task_bus(self, pp_bus: int) -> int:
        return int(pp_bus) + 1

    # getaway thermal rating in kA at the nominal voltage
    @property
    def _rating_ka(self) -> float:
        return self._GETAWAY_MVA / (math.sqrt(3) * self._NOMINAL_KV)

    # ── network construction ──────────────────────────────────────────────────

    def _build_net(self, pv_mw: float = 0.0, q_mvar: float = 0.0):
        net = pn.case33bw()
        net.load["p_mw"] = net.load["p_mw"] * self._LOAD_SCALE
        net.load["q_mvar"] = net.load["q_mvar"] * self._LOAD_SCALE
        net.line["max_i_ka"] = self._rating_ka
        net.ext_grid["vm_pu"] = 1.0
        self._pv_idx = pp.create_sgen(net, bus=self._PCC_BUS, p_mw=pv_mw,
                                      q_mvar=q_mvar, name="PV-Bus18")
        return net

    def _solve(self):
        try:
            pp.runpp(self._net, numba=False)
            self._converged = True
        except Exception:
            self._converged = False

    @staticmethod
    def _q_for_pf(p_mw: float, pf: float, mode: str) -> float:
        """Reactive setpoint (MVAr) for a PV operating at |pf|, signed by mode.

        In pandapower an sgen with q_mvar < 0 absorbs reactive power (inductive,
        lowers local voltage); q_mvar > 0 injects (capacitive, raises voltage)."""
        pf = max(1e-6, min(1.0, float(pf)))
        s = p_mw / pf
        q_mag = math.sqrt(max(0.0, s * s - p_mw * p_mw))
        return -q_mag if mode == "absorbing" else q_mag

    # ── exact HC reference (the env's own bisection on real power flow) ────────

    def _feasible_at(self, pv_mw: float) -> bool:
        """True iff a unity-PF injection of pv_mw at Bus 18 violates no limit.

        Deliberately checks only the UPPER voltage bound (vm_pu <= VMAX): PV
        active-power injection on a radial feeder raises local voltage, so the
        injection-driven HC is overvoltage-bound — the undervoltage limit can
        never bind as PV grows (it only relaxes), so it is not part of the HC
        feasibility test here."""
        net = self._build_net(pv_mw=pv_mw)
        try:
            pp.runpp(net, numba=False)
        except Exception:
            return False
        v_ok = float(net.res_bus.vm_pu.max()) <= self._VMAX + 1e-9
        t_ok = float(net.res_line.loading_percent.max()) <= self._LOADING_LIMIT + 1e-9
        return v_ok and t_ok

    def _binding_at(self, pv_mw: float):
        """(constraint_type, task_bus_or_line_id) for the FIRST limit hit just
        above the HC: which limit is violated at pv_mw, by how much.

        Precondition: call ONLY at an INFEASIBLE pv_mw (a size at which at least
        one limit is exceeded); at a feasible size both v_excess and t_excess are
        -1.0 and the return would be a meaningless 'overvoltage' default."""
        net = self._build_net(pv_mw=pv_mw)
        pp.runpp(net, numba=False)
        vmax = float(net.res_bus.vm_pu.max())
        vbus = int(net.res_bus.vm_pu.idxmax())
        lmax = float(net.res_line.loading_percent.max())
        lline = int(net.res_line.loading_percent.idxmax())
        v_over = vmax > self._VMAX + 1e-9
        t_over = lmax > self._LOADING_LIMIT + 1e-9
        # which margin is breached more (relative to its limit) -> binds first
        v_excess = (vmax - self._VMAX) if v_over else -1.0
        t_excess = (lmax - self._LOADING_LIMIT) / self._LOADING_LIMIT if t_over else -1.0
        if v_excess >= t_excess:
            return "overvoltage", self._to_task_bus(vbus)
        from_b = self._to_task_bus(int(net.line.at[lline, "from_bus"]))
        to_b = self._to_task_bus(int(net.line.at[lline, "to_bus"]))
        return "thermal_overload", f"{from_b}-{to_b}"

    def _compute_hc_reference(self):
        """Bisect the real power flow for the exact unity-PF HC + binding constraint."""
        lo, hi = 0.0, 10.0
        if not self._feasible_at(lo):
            # degenerate: even no PV violates a limit -> HC is 0
            self._hc_mw = 0.0
            self._binding_constraint, self._binding_id = self._binding_at(lo)
            return
        # grow the upper bracket until infeasible (HC is finite for PV injection)
        while self._feasible_at(hi) and hi < 1e4:
            hi *= 2.0
        while (hi - lo) > self._HC_TOL_MW:
            mid = 0.5 * (lo + hi)
            if self._feasible_at(mid):
                lo = mid
            else:
                hi = mid
        self._hc_mw = round(lo, 3)
        # binding constraint determined just above the HC (first limit exceeded)
        self._binding_constraint, self._binding_id = self._binding_at(hi)

    # ── env lifecycle ──────────────────────────────────────────────────────────

    def _reset_impl(self) -> dict:
        # exact grounded HC reference, computed from real power flow
        self._compute_hc_reference()

        # live working network the agent mutates (starts at no PV)
        self._net = self._build_net(pv_mw=0.0)
        self._pv_mw = 0.0
        self._pv_pf = 1.0
        self._pv_pf_mode = "injecting"
        self._solve()
        self._baseline_pcc_v = round(float(self._net.res_bus.vm_pu[self._PCC_BUS]), 4)
        self._baseline_getaway_loading = round(
            float(self._net.res_line.loading_percent[self._GETAWAY_LINE]), 2)

        # bookkeeping for scoring
        self._n_power_flows = 0
        self._max_feasible_verified = None  # largest unity-PF MW the agent ran PF on and saw feasible
        self._last_pf_feasible = None
        self._log: list[dict] = []
        self._reported = None

        return {
            "feeder": "IEEE 33-bus radial (pandapower case33bw)",
            "nominal_kv": self._NOMINAL_KV,
            "interconnection_bus": self._to_task_bus(self._PCC_BUS),  # 18
            "substation_slack_bus": self._to_task_bus(self._SLACK_BUS),  # 1
            "getaway_line": "1-2",
            "study_condition": "midday solar peak, load ~30% of nameplate",
            "baseline_no_pv": {
                "bus_18_voltage_pu": self._baseline_pcc_v,
                "bus_1_voltage_pu": 1.0,
                "getaway_line_loading_pct": self._baseline_getaway_loading,
                "violations": [],
            },
            "voltage_limits_pu": [self._VMIN, self._VMAX],
            "line_loading_limit_pct": self._LOADING_LIMIT,
            "getaway_thermal_rating_mva": self._GETAWAY_MVA,
            "search_resolution_mw": 0.1,
        }

    # ── tools ───────────────────────────────────────────────────────────────────

    def _tools(self) -> list[dict]:
        return [
            {"name": "get_limits",
             "description": "Return the operating limits: steady-state voltage band "
                            "[0.95, 1.05] pu (ANSI Range A) and the 100% line thermal "
                            "loading limit.",
             "parameters": {}},
            {"name": "set_pv_injection",
             "description": "Set the active-power PV injection (MW) at the given bus for "
                            "the next power flow (unity power factor unless changed). "
                            "Overwrites any previous injection at that bus.",
             "parameters": {"bus_id": {"type": "integer"}, "mw": {"type": "number"}}},
            {"name": "set_pv_power_factor",
             "description": "Set the PV inverter power factor (0 < pf <= 1) and mode "
                            "('absorbing' = inductive/Volt-VAr down, lowers voltage; "
                            "'injecting' = capacitive, raises voltage). Applies to the PV "
                            "at the given bus on the next power flow.",
             "parameters": {"bus_id": {"type": "integer"}, "pf": {"type": "number"},
                            "mode": {"type": "string", "enum": ["absorbing", "injecting"]}}},
            {"name": "run_power_flow",
             "description": "Run a full AC power flow on the current network. Returns "
                            "convergence, per-bus voltages (pu) with the max voltage and "
                            "its bus, per-line loadings (%) with the max loading and its "
                            "line, and whether any limit is violated.",
             "parameters": {}},
            {"name": "log_action",
             "description": "Record a step or decision with a justification for the "
                            "planning report.",
             "parameters": {"action": {"type": "string"},
                            "justification": {"type": "string"}}},
            {"name": "report_hosting_capacity",
             "description": "Submit the final answer (terminal, call exactly once, last): "
                            "the maximum hostable MW, the binding constraint type "
                            "('overvoltage' or 'thermal_overload'), and the bus or line id "
                            "where it binds.",
             "parameters": {"mw": {"type": "number"},
                            "binding_constraint": {"type": "string",
                                                   "enum": ["overvoltage", "thermal_overload"]},
                            "bus_or_line_id": {"type": "string"}}},
        ]

    def _dispatch_tool(self, tool: str, args: dict):
        if tool == "get_limits":
            return {"voltage_band_pu": [self._VMIN, self._VMAX],
                    "line_loading_limit_pct": self._LOADING_LIMIT,
                    "standard": "ANSI C84.1 Range A"}
        if tool == "set_pv_injection":
            return self._tool_set_pv(int(args["bus_id"]), float(args["mw"]))
        if tool == "set_pv_power_factor":
            return self._tool_set_pf(int(args["bus_id"]), float(args["pf"]),
                                     str(args.get("mode", "injecting")))
        if tool == "run_power_flow":
            return self._tool_run_pf()
        if tool == "log_action":
            self._log.append({"action": args.get("action", ""),
                              "justification": args.get("justification", "")})
            return {"logged": True, "log_entries": len(self._log)}
        if tool == "report_hosting_capacity":
            return self._tool_report(args)
        raise ValueError(f"Unknown tool: {tool}")

    def _tool_set_pv(self, task_bus: int, mw: float):
        if self._to_pp_bus(task_bus) != self._PCC_BUS:
            return {"rejected": True, "bus_id": task_bus,
                    "reason": f"PV interconnection is studied only at Bus "
                              f"{self._to_task_bus(self._PCC_BUS)} (far end of the main "
                              f"feeder); bus {task_bus} is not the point of interconnection."}
        if mw < 0:
            return {"rejected": True, "bus_id": task_bus, "reason": "mw must be non-negative"}
        self._pv_mw = float(mw)
        q = self._q_for_pf(self._pv_mw, self._pv_pf, self._pv_pf_mode)
        self._net.sgen.at[self._pv_idx, "p_mw"] = self._pv_mw
        self._net.sgen.at[self._pv_idx, "q_mvar"] = q
        return {"accepted": True, "bus_id": task_bus, "pv_mw": self._pv_mw,
                "power_factor": self._pv_pf, "pf_mode": self._pv_pf_mode,
                "pv_q_mvar": round(q, 4)}

    def _tool_set_pf(self, task_bus: int, pf: float, mode: str):
        if self._to_pp_bus(task_bus) != self._PCC_BUS:
            return {"rejected": True, "bus_id": task_bus,
                    "reason": f"no PV at bus {task_bus}; the PV is at Bus "
                              f"{self._to_task_bus(self._PCC_BUS)}."}
        if not (0.0 < pf <= 1.0):
            return {"rejected": True, "reason": "pf must satisfy 0 < pf <= 1"}
        if mode not in ("absorbing", "injecting"):
            return {"rejected": True, "reason": "mode must be 'absorbing' or 'injecting'"}
        self._pv_pf = float(pf)
        self._pv_pf_mode = mode
        q = self._q_for_pf(self._pv_mw, self._pv_pf, self._pv_pf_mode)
        self._net.sgen.at[self._pv_idx, "q_mvar"] = q
        return {"accepted": True, "bus_id": task_bus, "power_factor": self._pv_pf,
                "pf_mode": self._pv_pf_mode, "pv_q_mvar": round(q, 4),
                "note": "absorbing q < 0 lowers the local voltage; injecting q > 0 raises it"}

    def _tool_run_pf(self):
        self._solve()
        if not self._converged:
            self._last_pf_feasible = False
            return {"converged": False}
        vb = self._net.res_bus.vm_pu
        ll = self._net.res_line.loading_percent
        vmax = float(vb.max()); vmax_bus = int(vb.idxmax())
        lmax = float(ll.max()); lmax_line = int(ll.idxmax())
        v_violation = vmax > self._VMAX + 1e-9 or float(vb.min()) < self._VMIN - 1e-9
        t_violation = lmax > self._LOADING_LIMIT + 1e-9
        feasible = not (v_violation or t_violation)
        self._n_power_flows += 1
        self._last_pf_feasible = feasible
        # track the largest unity-PF injection the agent has verified feasible
        if feasible and abs(self._pv_pf - 1.0) < 1e-9 and self._pv_pf_mode == "injecting":
            if self._max_feasible_verified is None or self._pv_mw > self._max_feasible_verified:
                self._max_feasible_verified = self._pv_mw
        return {
            "converged": True,
            "pv_mw": self._pv_mw,
            "bus_voltages_pu": {str(self._to_task_bus(b)): round(float(vb[b]), 4)
                                for b in vb.index},
            "max_voltage_pu": round(vmax, 4),
            "max_voltage_bus": self._to_task_bus(vmax_bus),
            "min_voltage_pu": round(float(vb.min()), 4),
            "line_loadings_pct": {
                f"{self._to_task_bus(int(self._net.line.at[i, 'from_bus']))}-"
                f"{self._to_task_bus(int(self._net.line.at[i, 'to_bus']))}":
                    round(float(ll[i]), 2) for i in self._net.line.index},
            "max_loading_pct": round(lmax, 2),
            "max_loading_line": f"{self._to_task_bus(int(self._net.line.at[lmax_line,'from_bus']))}-"
                                f"{self._to_task_bus(int(self._net.line.at[lmax_line,'to_bus']))}",
            "voltage_violation": bool(v_violation),
            "thermal_violation": bool(t_violation),
            "any_violation": bool(not feasible),
        }

    def _tool_report(self, args: dict):
        mw = float(args.get("mw", 0.0))
        binding = str(args.get("binding_constraint", "")).lower()
        bus_or_line = str(args.get("bus_or_line_id", ""))
        # verify the reported value against the real physics at the moment of report
        verified_feasible = self._feasible_at(mw)
        self._reported = {"mw": mw, "binding_constraint": binding,
                          "bus_or_line_id": bus_or_line,
                          "verified_feasible": verified_feasible}
        self._done = True
        return {"status": "episode_complete", "reported_mw": mw,
                "binding_constraint": binding, "bus_or_line_id": bus_or_line,
                "reported_size_verified_feasible": verified_feasible}

    # ── scoring ─────────────────────────────────────────────────────────────────

    def _evaluate_impl(self) -> tuple[float, dict]:
        """Deterministic score in [0,1] reflecting correctness of the reported HC.

          ACCURACY (0.50): closeness of the reported MW to the true bisected HC —
            full credit within 0.1 MW, linearly degrading to 0 by 1.0 MW error.
          BINDING CONSTRAINT (0.30): +0.15 correct constraint TYPE
            ('overvoltage' vs 'thermal_overload'), +0.15 correct bus/line id.
          VERIFICATION GATE (0.20): the agent must have run >=1 power flow AND the
            reported size must NOT violate a limit (verified against real physics).
            Reporting an over-limit value, or never running power flow, forfeits
            this block; an over-limit report is additionally capped so a "close but
            illegal" answer can never reach the high band.
        """
        rep = self._reported
        hc = self._hc_mw

        if rep is None:
            return 0.0, {
                "reported": None, "true_hc_mw": hc,
                "true_binding_constraint": self._binding_constraint,
                "true_binding_id": self._binding_id,
                "power_flows_run": self._n_power_flows,
                "note": "no report_hosting_capacity call — episode produced no answer",
            }

        err = abs(rep["mw"] - hc)
        # accuracy: 1.0 within 0.1 MW, linear to 0 at 1.0 MW error
        if err <= 0.1:
            acc = 1.0
        elif err >= 1.0:
            acc = 0.0
        else:
            acc = 1.0 - (err - 0.1) / 0.9
        accuracy_score = 0.50 * acc

        # binding constraint
        type_ok = rep["binding_constraint"] == self._binding_constraint
        id_ok = str(rep["bus_or_line_id"]) == str(self._binding_id)
        binding_score = (0.15 if type_ok else 0.0) + (0.15 if id_ok else 0.0)

        # verification gate
        ran_pf = self._n_power_flows > 0
        reported_legal = bool(rep["verified_feasible"])
        verification_score = 0.20 if (ran_pf and reported_legal) else 0.0

        score = accuracy_score + binding_score + verification_score
        # hard cap: an over-limit (illegal) reported size can never reach the high
        # band even if numerically close to the true HC.
        if not reported_legal:
            score = min(score, 0.45)
        if not ran_pf:
            score = min(score, 0.30)
        score = max(0.0, min(1.0, score))

        return score, {
            "reported_mw": rep["mw"],
            "reported_binding_constraint": rep["binding_constraint"],
            "reported_bus_or_line_id": rep["bus_or_line_id"],
            "true_hc_mw": hc,
            "true_binding_constraint": self._binding_constraint,
            "true_binding_id": self._binding_id,
            "mw_error": round(err, 4),
            "reported_size_verified_feasible": reported_legal,
            "power_flows_run": self._n_power_flows,
            "max_feasible_verified_mw": self._max_feasible_verified,
            "binding_type_correct": bool(type_ok),
            "binding_id_correct": bool(id_ok),
            "log_entries": len(self._log),
            "accuracy_score": round(accuracy_score, 4),
            "binding_score": round(binding_score, 4),
            "verification_score": round(verification_score, 4),
        }

    def _task_description(self) -> str:
        return (
            "You are an autonomous distribution-planning agent for an IEEE 33-bus radial "
            "feeder (nominal 12.66 kV, substation at Bus 1). A developer wants to "
            "interconnect a single utility-scale PV plant at Bus 18 (the far end of the "
            "main feeder). Determine the maximum PV hosting capacity (MW) that can be added "
            "at Bus 18 before the first operating limit is violated, then report the binding "
            "constraint. The feeder is lightly loaded (midday solar peak, ~30% of nameplate). "
            "Workflow: (1) call get_limits to confirm the [0.95, 1.05] pu voltage band and "
            "the 100% thermal limit; (2) establish the no-PV baseline with "
            "set_pv_injection(18, 0.0) then run_power_flow; (3) coarse-bracket the capacity "
            "by raising the PV MW in ~1 MW steps and running power flow after each until a "
            "limit is first violated; (4) bisect between the last feasible and first "
            "infeasible size until the max feasible MW is resolved to within 0.1 MW; "
            "(5) identify whether overvoltage at Bus 18 or thermal overload on Line 1-2 binds "
            "first; (6) optionally test an absorbing power factor as a logged sensitivity but "
            "keep the unity-PF figure as the answer; (7) log every decision; (8) call "
            "report_hosting_capacity exactly once with the verified maximum MW, the binding "
            "constraint, and the bus/line id. Do NOT report a capacity you have not verified "
            "with run_power_flow, and do NOT report a size that exceeds any limit.")

    def _goal_state(self) -> dict:
        return {
            "interconnection_bus": self._to_task_bus(self._PCC_BUS),
            "voltage_limits_pu": [self._VMIN, self._VMAX],
            "line_loading_limit_pct": self._LOADING_LIMIT,
            "search_resolution_mw": 0.1,
            "report_once": "report_hosting_capacity",
        }
