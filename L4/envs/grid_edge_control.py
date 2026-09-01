"""
grid_edge_control_L4_001 — autonomous voltage-restoration episode on a real
IEEE 33-bus distribution feeder.

Grounding
---------
Reproduces the Grid-Agent autonomous remediation loop (arXiv:2508.05702,
"Grid-Agent: An LLM-Powered Multi-Agent System for Power Grid Control"):
perceive grid telemetry -> diagnose the violation root cause -> generate and
execute a prioritized corrective-control sequence, verifying each action with
an embedded power flow, to restore bus voltages to the [0.95, 1.05] pu band and
clear line overloads. Two disclosed deviations from the paper: (a) its
multi-agent pipeline (topology/planner/executor/validator agents) is collapsed
into a single-agent ReAct episode — the candidate LLM plays planner+executor
and the env's live power flow plays validator; (b) the paper's corrective
actions are switch reconfiguration, load curtailment and battery dispatch,
whereas this env uses the standard volt/VAR control set (capacitor switching ->
tap changer -> generation redispatch), conventional distribution-operations
practice rather than Grid-Agent's action set. GridMind (arXiv:2509.02494) and
PFAgent (arXiv:2604.10846) inform the multi-agent / power-flow-tool framing but
Grid-Agent is the reproduced control loop.

Network: pandapower `case33bw` (Thurner et al., IEEE TPWRS 2018) — the standard
12.66 kV radial distribution feeder, 33 buses, 32 in-service lines + 5
normally-open tie switches.

What is real vs simplified
--------------------------
REAL: the network, the post-event topology (a trunk line is tripped and a
normally-open tie is closed so power genuinely reroutes through a longer path),
and EVERY violation/observation — all bus voltages and line loadings come from a
live `pp.runpp` on the mutated net, never hardcoded. Capacitor banks inject
reactive power (`pp.create_shunt`), the tap changer shifts the substation
voltage setpoint, and the generator (`pp.create_sgen`) injects real power; each
tool mutates the net and the next power flow reflects it.

SIMPLIFIED: the substation regulator is modeled as the slack `ext_grid` voltage
setpoint (0.0125 pu/step) rather than an explicit OLTC transformer object; line
thermal ratings are assigned engineered-but-realistic `max_i_ka` values (trunk
0.40 kA, downstream laterals 0.080 kA) so the rerouted flow produces a genuine,
convergent lateral overload. Device bus locations and capacitor sizing were
adapted from the task prose to the locations the real case33bw physics supports
(the published task JSON is reconciled to these computed values).
"""
import warnings
warnings.filterwarnings("ignore")
import copy

import pandapower as pp
import pandapower.networks as pn

from derbench_env import DERBenchEnv


@DERBenchEnv.register("grid_edge_control_L4_001")
class IEEE33VoltageRestorationEnv(DERBenchEnv):
    """Restore voltages + clear overloads on a post-event IEEE 33-bus feeder."""

    MAX_STEPS = 25

    # ── post-event configuration (engineered against real case33bw physics) ────
    _SLACK_VSET = 1.03            # substation regulated voltage at t=0 (tap 0)
    _TAP_STEP_PU = 0.0125         # pu per tap step
    _TAP_MIN, _TAP_MAX = -3, 3
    _TRIP_LINE = 7                # case33bw line idx 7 = bus7-bus8 (the tripped trunk line)
    _TIE_LINE = 35                # normally-open tie idx 35 = bus17-bus32 (closed to reroute)
    _BRANCH3_LINES = [24, 25, 26, 27, 28, 29, 30, 31]   # downstream laterals (thin conductors)
    _TRUNK_RATING_KA = 0.40
    _LATERAL_RATING_KA = 0.080
    _CAP_Q_MVAR = 0.6            # reactive injection per capacitor bank when ON (capacitive)
    _GEN_BUS = 30                # controllable DG on the reconnected reroute lateral
    _GEN_MAX_MW = 2.0
    _VMIN, _VMAX = 0.95, 1.05
    _LOADING_LIMIT = 100.0

    # capacitor banks: bank_id -> bus (placed at the deep-undervoltage section)
    _CAP_BANKS = {"CB-8": 8, "CB-17": 17, "CB-14": 14}

    # ── env lifecycle ──────────────────────────────────────────────────────────

    def _build_net(self):
        net = pn.case33bw()
        net.ext_grid["vm_pu"] = self._SLACK_VSET
        net.line["max_i_ka"] = self._TRUNK_RATING_KA
        net.line.loc[net.line.index.isin(self._BRANCH3_LINES),
                     "max_i_ka"] = self._LATERAL_RATING_KA
        # the event: trip a trunk line, close a normally-open tie to reroute
        net.line.at[self._TRIP_LINE, "in_service"] = False
        net.line.at[self._TIE_LINE, "in_service"] = True
        # control devices, all initially inactive
        self._cap_idx = {bid: pp.create_shunt(net, bus=bus, q_mvar=0.0,
                                               name=bid)
                         for bid, bus in self._CAP_BANKS.items()}
        self._gen_idx = pp.create_sgen(net, bus=self._GEN_BUS, p_mw=0.0,
                                       q_mvar=0.0, name="G1")
        return net

    def _reset_impl(self) -> dict:
        self._net = self._build_net()
        self._tap = 0
        self._cap_state = {bid: False for bid in self._CAP_BANKS}
        self._gen_mw = 0.0
        # action bookkeeping for scoring
        self._cap_calls = 0
        self._tap_calls = 0
        self._gen_calls = 0
        self._isolate_calls = 0
        self._isolated_lines: list[str] = []
        self._log: list[dict] = []
        self._first_cap_step = None
        self._first_gen_step = None
        self._solve()
        bus_v, line_l = self._violations()
        # Post-event baseline violation counts, computed ONCE from the real solved
        # net so the partial-credit denominator can never drift from the actual
        # baseline on a pandapower version bump.
        self._n_baseline_bus = len(bus_v)
        self._n_baseline_line = len(line_l)
        return {
            "feeder": "IEEE 33-bus radial (pandapower case33bw)",
            "event": "Line bus7-bus8 tripped; tie bus17-bus32 auto-closed to reconnect "
                     "the de-energized section via a longer electrical path.",
            "converged": bool(self._converged),
            "initial_violations": {
                "bus_voltage_violations": bus_v,
                "line_loading_violations": line_l,
            },
            "control_devices": {
                "capacitor_banks": [{"bank_id": b, "bus": self._CAP_BANKS[b],
                                     "state": "OFF"} for b in self._CAP_BANKS],
                "transformer": {"transformer_id": "T1", "tap": 0,
                                "range": [self._TAP_MIN, self._TAP_MAX],
                                "pu_per_step": self._TAP_STEP_PU},
                "generator": {"gen_id": "G1", "bus": self._GEN_BUS,
                              "p_mw": 0.0, "max_mw": self._GEN_MAX_MW},
            },
            "voltage_limits_pu": [self._VMIN, self._VMAX],
            "line_loading_limit_pct": self._LOADING_LIMIT,
        }

    # ── power flow + violation extraction (the ground truth) ────────────────────

    def _solve(self):
        try:
            pp.runpp(self._net, numba=False)
            self._converged = True
        except Exception:
            self._converged = False

    def _violations(self):
        """Return (bus_voltage_violations, line_loading_violations) from the live
        solved net. Empty lists if the net is in-range. All values computed."""
        if not self._converged:
            return ([], [])
        vb = self._net.res_bus.vm_pu
        bus_v = [{"bus_id": int(b), "voltage_pu": round(float(vb[b]), 4),
                  "type": "undervoltage" if vb[b] < self._VMIN else "overvoltage"}
                 for b in vb.index
                 if vb[b] < self._VMIN or vb[b] > self._VMAX]
        ll = self._net.res_line.loading_percent
        line_l = []
        for i in self._net.line.index:
            if not self._net.line.at[i, "in_service"]:
                continue
            if ll[i] > self._LOADING_LIMIT:
                line_l.append({
                    "line_id": f"{int(self._net.line.at[i,'from_bus'])}-"
                               f"{int(self._net.line.at[i,'to_bus'])}",
                    "line_index": int(i),
                    "loading_pct": round(float(ll[i]), 1)})
        bus_v.sort(key=lambda d: d["voltage_pu"])
        line_l.sort(key=lambda d: -d["loading_pct"])
        return (bus_v, line_l)

    # ── tools ───────────────────────────────────────────────────────────────────

    def _tools(self) -> list[dict]:
        return [
            {"name": "get_telemetry",
             "description": "Voltage magnitude (pu) at a bus plus loading (%) of every "
                            "in-service line touching it, from the live network state.",
             "parameters": {"bus_id": {"type": "integer"}}},
            {"name": "run_power_flow",
             "description": "Run a full AC power flow on the current network. Returns "
                            "convergence plus all bus voltage violations and line "
                            "loading violations.",
             "parameters": {}},
            {"name": "adjust_tap_changer",
             "description": "Set substation transformer T1 tap to an absolute position "
                            "in [-3, 3] (each step ~0.0125 pu, positive raises voltage).",
             "parameters": {"transformer_id": {"type": "string"},
                            "tap_position": {"type": "integer"}}},
            {"name": "switch_capacitor_bank",
             "description": "Switch a capacitor bank ON or OFF. ON injects reactive "
                            "power locally, raising nearby voltage and reducing reactive "
                            "current draw.",
             "parameters": {"bank_id": {"type": "string"},
                            "state": {"type": "string", "enum": ["ON", "OFF"]}}},
            {"name": "redispatch_generator",
             "description": "Set the ABSOLUTE active-power output of generator G1 in MW "
                            "(clamped to [0, max]); this is a setpoint, not a delta. "
                            "Use only after reactive controls are exhausted.",
             "parameters": {"gen_id": {"type": "string"},
                            "p_mw": {"type": "number"}}},
            {"name": "isolate_line",
             "description": "Disconnect a line. Use ONLY for an actively faulted line — "
                            "the already-tripped line must NOT be re-isolated and "
                            "isolating healthy lines worsens the situation.",
             "parameters": {"line_id": {"type": "string"}}},
            {"name": "log_action",
             "description": "Record a decision (including a decision NOT to act) with a "
                            "justification for the operations log.",
             "parameters": {"action": {"type": "string"},
                            "justification": {"type": "string"}}},
            {"name": "declare_done",
             "description": "End the episode; the final solved network is then graded.",
             "parameters": {}},
        ]

    def _dispatch_tool(self, tool: str, args: dict):
        if tool == "get_telemetry":
            return self._tool_get_telemetry(int(args["bus_id"]))
        if tool == "run_power_flow":
            self._solve()
            bus_v, line_l = self._violations()
            return {"converged": bool(self._converged),
                    "bus_voltage_violations": bus_v,
                    "line_loading_violations": line_l,
                    "n_violations": len(bus_v) + len(line_l)}
        if tool == "adjust_tap_changer":
            return self._tool_tap(str(args.get("transformer_id", "T1")),
                                  int(args["tap_position"]))
        if tool == "switch_capacitor_bank":
            return self._tool_cap(str(args["bank_id"]), str(args["state"]).upper())
        if tool == "redispatch_generator":
            return self._tool_gen(str(args.get("gen_id", "G1")),
                                  float(args.get("p_mw", args.get("mw_change", 0))))
        if tool == "isolate_line":
            return self._tool_isolate(str(args["line_id"]))
        if tool == "log_action":
            self._log.append({"action": args.get("action", ""),
                              "justification": args.get("justification", "")})
            return {"logged": True, "log_entries": len(self._log)}
        if tool == "declare_done":
            self._done = True
            self._solve()
            bus_v, line_l = self._violations()
            return {"status": "episode_complete",
                    "final_bus_violations": len(bus_v),
                    "final_line_violations": len(line_l)}
        raise ValueError(f"Unknown tool: {tool}")

    def _tool_get_telemetry(self, bus_id: int):
        if bus_id not in self._net.bus.index:
            return {"error": f"Unknown bus_id {bus_id}; valid buses are "
                             f"{int(self._net.bus.index.min())}.."
                             f"{int(self._net.bus.index.max())}."}
        self._solve()
        v = round(float(self._net.res_bus.vm_pu[bus_id]), 4) if self._converged else None
        lines = []
        for i in self._net.line.index:
            if not self._net.line.at[i, "in_service"]:
                continue
            if bus_id in (int(self._net.line.at[i, "from_bus"]),
                          int(self._net.line.at[i, "to_bus"])):
                lid = f"{int(self._net.line.at[i,'from_bus'])}-{int(self._net.line.at[i,'to_bus'])}"
                lines.append({"line_id": lid,
                              "loading_pct": round(float(self._net.res_line.loading_percent[i]), 1)})
        return {"bus_id": bus_id, "voltage_pu": v, "connected_lines": lines}

    def _tool_tap(self, transformer_id: str, position: int):
        if transformer_id != "T1":
            raise ValueError(f"Unknown transformer: {transformer_id} (only 'T1' exists)")
        position = max(self._TAP_MIN, min(self._TAP_MAX, int(position)))
        self._tap = position
        self._tap_calls += 1
        self._net.ext_grid["vm_pu"] = self._SLACK_VSET + position * self._TAP_STEP_PU
        self._solve()
        return {"transformer_id": "T1", "tap": self._tap,
                "slack_vm_pu": round(self._SLACK_VSET + position * self._TAP_STEP_PU, 4),
                "converged": bool(self._converged)}

    def _tool_cap(self, bank_id: str, state: str):
        if bank_id not in self._cap_idx:
            raise ValueError(f"Unknown capacitor bank: {bank_id}")
        self._cap_calls += 1
        if self._first_cap_step is None:
            self._first_cap_step = self._step_count
        on = (state == "ON")
        self._cap_state[bank_id] = on
        # shunt q_mvar negative = capacitive injection (raises voltage)
        self._net.shunt.at[self._cap_idx[bank_id], "q_mvar"] = -self._CAP_Q_MVAR if on else 0.0
        self._solve()
        return {"bank_id": bank_id, "state": state, "bus": self._CAP_BANKS[bank_id],
                "bus_voltage_pu": round(float(self._net.res_bus.vm_pu[self._CAP_BANKS[bank_id]]), 4)
                                  if self._converged else None,
                "converged": bool(self._converged)}

    def _tool_gen(self, gen_id: str, mw: float):
        if gen_id != "G1":
            raise ValueError(f"Unknown generator: {gen_id} (only 'G1' exists)")
        self._gen_calls += 1
        if self._first_gen_step is None:
            self._first_gen_step = self._step_count
        self._gen_mw = max(0.0, min(self._GEN_MAX_MW, float(mw)))
        self._net.sgen.at[self._gen_idx, "p_mw"] = self._gen_mw
        self._solve()
        return {"gen_id": "G1", "p_mw": self._gen_mw, "converged": bool(self._converged)}

    def _tool_isolate(self, line_id: str):
        self._isolate_calls += 1
        self._isolated_lines.append(line_id)
        idx = self._resolve_line(line_id)
        if idx is not None:
            self._net.line.at[idx, "in_service"] = False
            self._solve()
        return {"line_id": line_id, "isolated": idx is not None,
                "warning": "isolate_line should only target an actively faulted line",
                "converged": bool(self._converged)}

    def _resolve_line(self, line_id: str):
        """Map a 'from-to' or integer line_id to a pandapower line index."""
        s = str(line_id).strip()
        if s.isdigit():
            idx = int(s)
            return idx if idx in self._net.line.index else None
        if "-" in s:
            try:
                a, b = (int(x) for x in s.split("-"))
            except ValueError:
                return None
            for i in self._net.line.index:
                fb, tb = int(self._net.line.at[i, "from_bus"]), int(self._net.line.at[i, "to_bus"])
                if {fb, tb} == {a, b}:
                    return i
        return None

    # ── scoring ─────────────────────────────────────────────────────────────────

    def _evaluate_impl(self) -> tuple[float, dict]:
        """Deterministic score. Primary = constraint satisfaction on the FINAL
        solved network; secondary = action appropriateness (reactive controls
        before generator redispatch, no isolate_line misuse)."""
        self._solve()
        bus_v, line_l = self._violations()
        n_bus, n_line = len(bus_v), len(line_l)
        constraints_ok = self._converged and n_bus == 0 and n_line == 0

        # Primary (0.80): partial credit toward zero violations off the post-event
        # baseline (the violation counts captured at reset from the real solved
        # net), so a no-op scores near zero and only a fully restored net earns the
        # full block.
        if constraints_ok:
            primary = 0.80
        else:
            n_baseline_bus = max(1, self._n_baseline_bus)
            n_baseline_line = max(1, self._n_baseline_line)
            bus_credit = max(0.0, 1.0 - n_bus / n_baseline_bus)
            line_credit = max(0.0, 1.0 - n_line / n_baseline_line)
            primary = 0.80 * 0.5 * (bus_credit + line_credit) if self._converged else 0.0

        # Action appropriateness (0.20):
        appropriateness = 0.0
        # +0.10 used reactive/cap-bank control before generator redispatch
        used_caps = self._cap_calls > 0
        gen_after_caps = (self._first_gen_step is None) or (
            self._first_cap_step is not None and self._first_cap_step <= self._first_gen_step)
        if used_caps and gen_after_caps:
            appropriateness += 0.10
        elif used_caps:
            appropriateness += 0.05
        # +0.10 did NOT misuse isolate_line
        if self._isolate_calls == 0:
            appropriateness += 0.10

        # isolate_line misuse is a hard cap on top of the action penalty: a
        # restoration that leans on tripping lines must never reach the clean
        # ceiling even if it happens to satisfy constraints.
        score = primary + appropriateness
        if self._isolate_calls > 0:
            score = min(score, 0.85)
        score = max(0.0, min(1.0, score))

        return score, {
            "converged": bool(self._converged),
            "constraints_satisfied": bool(constraints_ok),
            "final_bus_violations": n_bus,
            "final_line_violations": n_line,
            "final_min_voltage_pu": round(float(self._net.res_bus.vm_pu.min()), 4)
                                    if self._converged else None,
            "final_max_loading_pct": round(float(self._net.res_line.loading_percent.max()), 1)
                                     if self._converged else None,
            "cap_bank_calls": self._cap_calls,
            "tap_calls": self._tap_calls,
            "gen_calls": self._gen_calls,
            "isolate_line_calls": self._isolate_calls,
            "reactive_before_generator": bool(used_caps and gen_after_caps),
            "log_entries": len(self._log),
            "primary_score": round(primary, 4),
            "appropriateness_score": round(appropriateness, 4),
        }

    def _task_description(self) -> str:
        return (
            "You are an autonomous grid control operator on an IEEE 33-bus radial "
            "distribution feeder (12.66 kV). At t=0 the trunk line bus7-bus8 tripped; "
            "a normally-open tie (bus17-bus32) auto-closed to reconnect the de-energized "
            "section via a longer electrical path. The rerouted load has caused "
            "undervoltage at the reconnected end-of-feeder buses and an overload on a "
            "downstream lateral. Diagnose with get_telemetry/run_power_flow, then restore "
            "all bus voltages to [0.95, 1.05] pu and clear all line overloads using the "
            "least-invasive controls first: switch capacitor banks ON, then adjust the T1 "
            "tap changer, then redispatch generator G1. Do NOT isolate additional lines "
            "(the faulted line is already tripped). Log your decisions and call "
            "declare_done when the network is violation-free.")

    def _goal_state(self) -> dict:
        return {"all_bus_voltages_pu_within": [self._VMIN, self._VMAX],
                "all_line_loading_pct_at_most": self._LOADING_LIMIT,
                "preferred_action_order": ["capacitor_banks", "tap_changer", "generator_redispatch"],
                "forbidden": "isolate_line on any healthy/already-tripped line"}
