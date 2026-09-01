"""
fault_protection_L4_001 — autonomous distribution FLISR episode (Fault
Location, Isolation, Service Restoration) on a radial feeder.

Grounding
---------
FLISR is a CLASSICAL distribution-automation function, not an LLM-agent paper —
this task tests the agentic *application* of a standards-based function, and we
do NOT cite any FLISR-specific LLM-agent paper (none exists; the task's
`reference_papers` field is honestly recorded as such). The protection /
restoration model is grounded in:

  * IEEE Std C37.112-2018, "Inverse-Time Characteristics Equations for
    Overcurrent Relays" — the inverse-time overcurrent basis for detecting and
    locating a permanent fault from the fault-current path (the faulted-circuit
    indicators here encode exactly that path: every section that carried fault
    current toward the fault reads TRUE, everything beyond reads FALSE). Listed
    in the task's source_platforms.
  * Symmetrical-components fault analysis (Fortescue, 1918, doi:10.1109/
    T-AIEE.1918.4765570) — the foundational method by which fault current and
    its propagation along the radial feeder are computed; the faulted-section
    localization here is the discrete graph abstraction of that current path.
  * FLISR distribution automation (NREL/EPRI distribution-automation function,
    https://www.nrel.gov/grid/distribution-automation.html) — the
    detect -> locate -> isolate-faulted-section -> reconfigure-to-restore-
    healthy-load workflow this episode reproduces, including back-feeding
    stranded healthy load through a normally-open tie to an adjacent feeder
    subject to that feeder's spare capacity and thermal limits.

For the AGENTIC framing (perceive -> plan a switching sequence -> verify with an
embedded check -> declare done) we follow the autonomous grid-operations agent
loop of Grid-Agent (arXiv:2508.05702, "Grid-Agent: An LLM-Powered Multi-Agent
System for Power Grid Control"), the same agentic basis used by the sibling
grid_edge_control and cybersecurity envs. Grid-Agent is the agentic-operations
methodology; the protection/restoration content is the classical FLISR function
above. No FLISR-specific LLM-agent paper is cited because none exists.

The feeder model (deterministic graph -> exact ground truth)
------------------------------------------------------------
A single radial feeder of five line sections S1..S5 in series from the
substation, separated by remotely-controllable switches, with a normally-open
tie to an adjacent feeder at the downstream end:

    [SUB] --CB-- (S1) --SW12-- (S2) --SW23-- (S3*) --SW34-- (S4) --SW45-- (S5) --TIE-- [ADJ]

  * Sections carry load (kW): S1=200, S2=150, S3=300, S4=250, S5=180.
  * S3 is the PERMANENTLY faulted section.
  * Switches are graph edges; a switch state of "closed" connects its two
    endpoints. CB (substation breaker) and TIE (normally-open) are switches too.
  * SUB and ADJ are the two sources. A section is ENERGIZED iff it is connected
    through closed switches to a live source.

At any configuration the env computes — from the live graph, never hardcoded —
which sections are energized, whether the faulted section S3 is energized
(fault-free iff it is NOT), whether the configuration is RADIAL (no closed loop;
in particular no closed path linking SUB and ADJ, which would parallel the two
feeders), and the restored healthy load (sum of healthy energized section loads,
with the load drawn through the tie checked against the tie's spare capacity).

Exact optimal solution (deterministic graph logic, no solver)
-------------------------------------------------------------
  * Faulted section: the last section whose indicator is TRUE adjacent to the
    first FALSE — here indicators are TRUE on S1,S2,S3 and FALSE on S4,S5, so the
    fault is in S3.
  * Minimal isolation: open exactly the two switches bounding S3 — SW23 and
    SW34 (2 operations).
  * Maximal restorable healthy load: all upstream healthy load (S1+S2 = 350 kW,
    still fed from the substation through CB & SW12) PLUS the stranded downstream
    healthy load (S4+S5 = 430 kW) back-fed via the tie IFF the tie spare
    capacity (500 kW) covers it — 430 <= 500, so it can be restored:
    350 + 430 = 780 kW of the 1080 kW total (the 300 kW in faulted S3 stays
    unserved until repair).
"""
from derbench_env import DERBenchEnv


@DERBenchEnv.register("fault_protection_L4_001")
class FLISRFeederEnv(DERBenchEnv):
    """Locate -> isolate -> restore on a radial feeder with a normally-open tie."""

    MAX_STEPS = 25

    # ── feeder definition (from the published task context) ──────────────────
    _SECTIONS = ["S1", "S2", "S3", "S4", "S5"]
    _SECTION_LOADS_KW = {"S1": 200, "S2": 150, "S3": 300, "S4": 250, "S5": 180}
    _FAULTED_SECTION = "S3"
    _FAULT_INDICATORS = {"S1": True, "S2": True, "S3": True, "S4": False, "S5": False}
    _TIE_SPARE_KW = 500.0
    # Per-section feeder line thermal rating. Non-binding in this scenario (the
    # tie spare capacity is the active back-feed constraint, 500 < 600), but
    # surfaced via get_ratings because the prompt asks the agent to check it.
    _LINE_THERMAL_LIMIT_KW = 600.0

    # sources
    _SUB = "SUB"   # substation
    _ADJ = "ADJ"   # adjacent feeder (reachable only through the normally-open tie)

    # switches as graph edges (switch_id -> (node_a, node_b)); nodes are sources
    # or section names. Closed switch == edge present in the connectivity graph.
    _SWITCH_EDGES = {
        "CB":   (_SUB, "S1"),
        "SW12": ("S1", "S2"),
        "SW23": ("S2", "S3"),
        "SW34": ("S3", "S4"),
        "SW45": ("S4", "S5"),
        "TIE":  ("S5", _ADJ),
    }
    # initial states: everything closed except the normally-open tie
    _INITIAL_STATE = {"CB": True, "SW12": True, "SW23": True, "SW34": True,
                      "SW45": True, "TIE": False}

    # ── exact optimal solution (computed deterministically) ──────────────────

    @classmethod
    def _locate_fault(cls):
        """Faulted section from the indicator pattern: the deepest section that
        reported fault current (TRUE) before the first that did not (FALSE).

        Assumes a single radial feeder with a contiguous TRUE-prefix indicator
        pattern (TRUE on every section up to and including the fault, FALSE
        beyond)."""
        last_true = None
        for s in cls._SECTIONS:
            if cls._FAULT_INDICATORS[s]:
                last_true = s
            else:
                break
        if last_true is None:
            raise ValueError("no faulted section located")
        return last_true

    @classmethod
    def _bounding_switches(cls, section):
        """The two switches adjacent to a section in the radial chain."""
        return sorted(sw for sw, (a, b) in cls._SWITCH_EDGES.items()
                      if section in (a, b))

    @classmethod
    def _optimal(cls):
        """Exact optimal (faulted section, isolation switches, restorable load).

        Upstream healthy = sections before the fault (fed from the substation).
        Downstream healthy = sections after the fault (stranded; back-fed via the
        tie only if the tie spare capacity covers their total load)."""
        faulted = cls._locate_fault()
        fi = cls._SECTIONS.index(faulted)
        iso = cls._bounding_switches(faulted)
        upstream = cls._SECTIONS[:fi]
        downstream = cls._SECTIONS[fi + 1:]
        up_load = sum(cls._SECTION_LOADS_KW[s] for s in upstream)
        down_load = sum(cls._SECTION_LOADS_KW[s] for s in downstream)
        restorable = up_load
        if down_load <= cls._TIE_SPARE_KW + 1e-9 and down_load <= cls._LINE_THERMAL_LIMIT_KW + 1e-9:
            restorable += down_load
        return faulted, iso, restorable, up_load, down_load

    # ── env lifecycle ────────────────────────────────────────────────────────

    def _reset_impl(self) -> dict:
        self._sw = dict(self._INITIAL_STATE)
        self._log: list[dict] = []
        self._switch_ops = 0          # count of open/close operations the agent issues
        self._energize_checks = 0
        (self._opt_faulted, self._opt_iso, self._opt_restore,
         self._opt_up, self._opt_down) = self._optimal()
        return {
            "feeder": "single radial feeder S1..S5 from substation, normally-open tie "
                      "to an adjacent feeder at the downstream end of S5",
            "event": "A permanent fault has just occurred on one line section. Locate it, "
                     "isolate it with the minimum switching, and restore the maximum "
                     "healthy load without overloading the tie; keep the config radial.",
            "switch_states": dict(self._sw),
            "n_sections": len(self._SECTIONS),
            "hint": "Fault-circuit indicators read TRUE on every section that carried "
                    "fault current (upstream of and including the fault) and FALSE beyond.",
        }

    # ── connectivity / energization (the ground-truth physics) ───────────────

    def _adjacency(self):
        """Undirected adjacency of the live (closed-switch) graph: node -> set."""
        adj: dict[str, set] = {self._SUB: set(), self._ADJ: set()}
        for s in self._SECTIONS:
            adj[s] = set()
        for sw, (a, b) in self._SWITCH_EDGES.items():
            if self._sw[sw]:
                adj[a].add((b, sw))
                adj[b].add((a, sw))
        return adj

    def _energized_sections(self):
        """Sections connected through closed switches to ANY live source."""
        adj = self._adjacency()
        seen = set()
        stack = [self._SUB, self._ADJ]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            for nbr, _sw in adj.get(node, ()):
                if nbr not in seen:
                    stack.append(nbr)
        return {s for s in self._SECTIONS if s in seen}

    def _is_radial(self):
        """Radial == no closed loop AND no section fed by two sources. Both fail
        the same way: if the substation and the adjacent feeder end up in the
        SAME connected component of the closed-switch graph they parallel the two
        feeders (a tie closed without first isolating the in-between section), and
        any genuine cycle would likewise link a node back to a source. We detect
        non-radial == SUB and ADJ are mutually reachable, OR an undirected cycle
        exists. The radial feeder model here has a single tie, so SUB<->ADJ
        reachability is the operative non-radial condition; a generic cycle check
        is included for robustness."""
        adj = self._adjacency()
        # (1) two sources in one component -> parallel feed -> non-radial.
        seen = set()
        stack = [self._SUB]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            for nbr, _sw in adj.get(node, ()):
                if nbr not in seen:
                    stack.append(nbr)
        if self._ADJ in seen:
            return False
        # (2) generic undirected-cycle check across all components.
        visited = set()
        for start in adj:
            if start in visited:
                continue
            stack = [(start, None)]
            while stack:
                node, via = stack.pop()
                if node in visited:
                    return False  # reached an already-visited node via a new edge -> cycle
                visited.add(node)
                for nbr, sw in adj.get(node, ()):
                    if sw == via:
                        continue  # don't immediately backtrack along the arrival edge
                    if nbr in visited:
                        return False
                    stack.append((nbr, sw))
        return True

    def _restored_load(self):
        """Total energized HEALTHY load (faulted section excluded even if it is
        somehow energized — that load is not 'served', it is a fault)."""
        en = self._energized_sections()
        return sum(self._SECTION_LOADS_KW[s] for s in en if s != self._FAULTED_SECTION)

    def _tie_load_kw(self):
        """Healthy load drawn from the ADJ source through the closed tie. Found
        by removing the substation source and seeing what ADJ alone energizes."""
        if not self._sw["TIE"]:
            return 0.0
        adj = self._adjacency()
        seen = set()
        stack = [self._ADJ]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            for nbr, _sw in adj.get(node, ()):
                if nbr == self._SUB:        # tie must not source the substation
                    continue
                if nbr not in seen:
                    stack.append(nbr)
        return sum(self._SECTION_LOADS_KW[s] for s in self._SECTIONS
                   if s in seen and s != self._FAULTED_SECTION)

    def _status(self):
        """Live FLISR status of the CURRENT switch configuration."""
        energized = self._energized_sections()
        fault_energized = self._FAULTED_SECTION in energized
        radial = self._is_radial()
        tie_load = self._tie_load_kw()
        tie_ok = tie_load <= self._TIE_SPARE_KW + 1e-9
        thermal_ok = tie_load <= self._LINE_THERMAL_LIMIT_KW + 1e-9
        return {
            "energized_sections": sorted(energized),
            "faulted_section": self._FAULTED_SECTION,
            "faulted_section_energized": bool(fault_energized),
            "fault_free": bool(not fault_energized),
            "radial": bool(radial),
            "tie_load_kw": round(tie_load, 1),
            "tie_spare_capacity_kw": self._TIE_SPARE_KW,
            "within_tie_capacity": bool(tie_ok),
            "within_thermal_limit": bool(thermal_ok),
            "restored_healthy_load_kw": round(self._restored_load(), 1),
            "within_ratings": bool(tie_ok and thermal_ok),
        }

    # ── tools ────────────────────────────────────────────────────────────────

    def _tools(self) -> list[dict]:
        return [
            {"name": "get_topology",
             "description": "Buses/line sections, every switch and its current state "
                            "(open/closed), and the normally-open tie to the adjacent feeder.",
             "parameters": {}},
            {"name": "get_fault_indicators",
             "description": "Faulted-circuit indicators: which sections reported fault "
                            "current. TRUE upstream of and including the fault, FALSE beyond.",
             "parameters": {}},
            {"name": "get_section_loads",
             "description": "Load (kW) on each line section.",
             "parameters": {}},
            {"name": "get_ratings",
             "description": "The tie feeder's spare capacity (kW) and the per-section line "
                            "thermal limit (kW).",
             "parameters": {}},
            {"name": "open_switch",
             "description": "Open a switch (disconnect its two endpoints). Rejected for an "
                            "unknown switch_id.",
             "parameters": {"switch_id": {"type": "string"}}},
            {"name": "close_switch",
             "description": "Close a switch (connect its two endpoints). Rejected for an "
                            "unknown switch_id.",
             "parameters": {"switch_id": {"type": "string"}}},
            {"name": "energize_check",
             "description": "Status of the CURRENT switch configuration: which sections are "
                            "energized, whether the faulted section is de-energized "
                            "(fault-free), whether the topology is radial, and whether the "
                            "tie/thermal ratings hold.",
             "parameters": {}},
            {"name": "log_decision",
             "description": "Record a switching decision with its justification.",
             "parameters": {"action": {"type": "string"},
                            "justification": {"type": "string"}}},
            {"name": "declare_done",
             "description": "End the episode; the final switch configuration is then graded.",
             "parameters": {}},
        ]

    def _dispatch_tool(self, tool: str, args: dict):
        if tool == "get_topology":
            return {
                "sections": list(self._SECTIONS),
                "switches": {sw: {"between": list(self._SWITCH_EDGES[sw]),
                                  "state": "closed" if self._sw[sw] else "open"}
                             for sw in self._SWITCH_EDGES},
                "tie_switch": "TIE",
                "sources": {"substation": self._SUB, "adjacent_feeder": self._ADJ},
            }
        if tool == "get_fault_indicators":
            return {"fault_indicators": dict(self._FAULT_INDICATORS),
                    "note": "TRUE = section carried fault current."}
        if tool == "get_section_loads":
            return {"section_loads_kw": dict(self._SECTION_LOADS_KW),
                    "total_load_kw": sum(self._SECTION_LOADS_KW.values())}
        if tool == "get_ratings":
            return {"tie_spare_capacity_kw": self._TIE_SPARE_KW,
                    "line_thermal_limit_kw": self._LINE_THERMAL_LIMIT_KW}
        if tool == "open_switch":
            return self._tool_set_switch(str(args.get("switch_id", "")), open_it=True)
        if tool == "close_switch":
            return self._tool_set_switch(str(args.get("switch_id", "")), open_it=False)
        if tool == "energize_check":
            self._energize_checks += 1
            return self._status()
        if tool == "log_decision":
            self._log.append({"action": args.get("action", ""),
                              "justification": args.get("justification", "")})
            return {"logged": True, "log_entries": len(self._log)}
        if tool == "declare_done":
            self._done = True
            st = self._status()
            return {"status": "episode_complete",
                    "final_switch_states": {sw: "closed" if self._sw[sw] else "open"
                                            for sw in self._SWITCH_EDGES},
                    "fault_free": st["fault_free"], "radial": st["radial"],
                    "restored_healthy_load_kw": st["restored_healthy_load_kw"]}
        raise ValueError(f"Unknown tool: {tool}")

    def _tool_set_switch(self, switch_id: str, *, open_it: bool):
        if switch_id not in self._SWITCH_EDGES:
            return {"rejected": True, "switch_id": switch_id,
                    "reason": f"unknown switch_id {switch_id!r} "
                              f"(valid: {sorted(self._SWITCH_EDGES)})"}
        target = (not open_it)  # closed == True
        if self._sw[switch_id] != target:
            self._switch_ops += 1
        self._sw[switch_id] = target
        return {"accepted": True, "switch_id": switch_id,
                "state": "open" if open_it else "closed",
                "switch_operations": self._switch_ops}

    # ── scoring ──────────────────────────────────────────────────────────────

    def _evaluate_impl(self) -> tuple[float, dict]:
        """Deterministic score in [0,1] that reflects FLISR correctness.

        HARD GATE: the faulted section must be de-energized (fault-free) AND the
          final configuration must be radial. If either fails, the score is
          capped low (only small partial credit) — an agent that leaves the fault
          energized or creates a loop cannot score well.
        CORRECTNESS COMPONENTS (when the gate holds):
          - isolation (0.30): both bounding switches of the faulted section open;
          - minimal switching (0.15): no operations beyond the optimal count
            (open the 2 bounding switches + close the tie = 3 ops when the tie is
            usable), graded down for each extra op;
          - restoration (0.40): restored healthy load as a fraction of the
            optimal restorable load, WITHOUT overloading the tie (an over-capacity
            tie closure forfeits this block);
          - verification (0.15): ran energize_check at least once.
        """
        st = self._status()
        fault_free = st["fault_free"]
        radial = st["radial"]
        within_ratings = st["within_ratings"]
        restored = self._restored_load()

        # exact optimal switching op count: 2 isolation opens + (1 tie close if
        # the downstream healthy load can be back-fed).
        opt_ops = 2 + (1 if self._opt_restore > self._opt_up + 1e-9 else 0)

        bounding_open = all(not self._sw[sw] for sw in self._opt_iso)

        # ── HARD GATE ────────────────────────────────────────────────────────
        gate_ok = fault_free and radial
        if not gate_ok:
            # small partial credit only for having located+started isolation, but
            # always far below the 0.5 "good" floor.
            partial = 0.0
            if bounding_open:
                partial += 0.15
            partial += 0.10 * (1.0 if self._energize_checks > 0 else 0.0)
            score = min(0.35, partial)
            return self._finish(score, st, restored, opt_ops, bounding_open,
                                gate_ok=False)

        # ── CORRECTNESS COMPONENTS ───────────────────────────────────────────
        iso_score = 0.30 if bounding_open else 0.0

        extra_ops = max(0, self._switch_ops - opt_ops)
        minimal_score = 0.15 * max(0.0, 1.0 - 0.5 * extra_ops)

        # restoration: fraction of the optimal restorable load actually restored,
        # forfeited if the tie is overloaded (closing it beyond its spare).
        if within_ratings and self._opt_restore > 1e-9:
            restore_frac = min(1.0, restored / self._opt_restore)
        elif not within_ratings:
            restore_frac = 0.0
        else:
            restore_frac = 1.0
        restore_score = 0.40 * restore_frac

        verify_score = 0.15 if self._energize_checks > 0 else 0.0

        score = iso_score + minimal_score + restore_score + verify_score
        score = max(0.0, min(1.0, score))
        return self._finish(score, st, restored, opt_ops, bounding_open, gate_ok=True)

    def _finish(self, score, st, restored, opt_ops, bounding_open, *, gate_ok):
        details = {
            "faulted_section": self._opt_faulted,
            "optimal_isolation_switches": list(self._opt_iso),
            "optimal_restored_load_kw": round(self._opt_restore, 1),
            "optimal_switch_operations": opt_ops,
            "faulted_section_isolated": bool(bounding_open),
            "faulted_section_energized": st["faulted_section_energized"],
            "fault_free": st["fault_free"],
            "radial": st["radial"],
            "within_ratings": st["within_ratings"],
            "within_tie_capacity": st["within_tie_capacity"],
            "tie_load_kw": st["tie_load_kw"],
            "restored_load_kw": round(restored, 1),
            "switch_operations": self._switch_ops,
            "energize_checks": self._energize_checks,
            "final_switch_states": {sw: "closed" if self._sw[sw] else "open"
                                    for sw in self._SWITCH_EDGES},
            "hard_gate_satisfied": bool(gate_ok),
            "log_entries": len(self._log),
        }
        return max(0.0, min(1.0, score)), details

    def _task_description(self) -> str:
        return (
            "You are an autonomous distribution FLISR (Fault Location, Isolation, Service "
            "Restoration) agent for a single radial feeder of five line sections S1..S5 in "
            "series from the substation, with remotely-controllable switches CB, SW12, SW23, "
            "SW34, SW45 between/around the sections and a normally-open tie (TIE) to an "
            "adjacent feeder at the downstream end of S5. A permanent fault has just occurred "
            "on one section. Workflow: read get_topology, get_fault_indicators, "
            "get_section_loads and get_ratings; LOCATE the faulted section (indicators read "
            "TRUE on every section that carried fault current and FALSE beyond, so the fault "
            "is in the deepest TRUE section); ISOLATE it with the MINIMUM switching by opening "
            "exactly the two switches bounding that section; RESTORE the maximum healthy load "
            "— upstream healthy sections stay fed from the substation, and stranded downstream "
            "healthy sections can be back-fed by closing the normally-open tie ONLY IF the "
            "tie's spare capacity covers their total load (never overload the adjacent feeder); "
            "run energize_check to verify the configuration is radial, fault-free and within "
            "ratings; log your decisions; then declare_done. Keep the final configuration "
            "radial (no loops) and the faulted section de-energized.")

    def _goal_state(self) -> dict:
        return {
            "faulted_section": self._opt_faulted,
            "isolation_switches": list(self._opt_iso),
            "optimal_restored_load_kw": round(self._opt_restore, 1),
            "upstream_healthy_load_kw": round(self._opt_up, 1),
            "downstream_healthy_load_kw": round(self._opt_down, 1),
            "tie_spare_capacity_kw": self._TIE_SPARE_KW,
            "requirements": ["faulted section de-energized", "configuration radial",
                             "tie not overloaded", "minimal switching"],
        }
