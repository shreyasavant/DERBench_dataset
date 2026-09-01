# DERBench: A Benchmark for Evaluating LLMs on Distributed Energy Resource Operations

DERBench is a hierarchical benchmark of **174 distributed-energy-resource (DER) engineering tasks** spanning nine problem categories and four capability tiers. The lower tiers test knowledge and scenario reasoning against reference-anchored rubrics; the upper tiers are graded **executably** — generated code is run against precomputed answer keys, and agentic episodes are scored against a reference the environment computes at reset (a solver optimum, a power-flow result, or measured detection accuracy).

This repository holds the dataset: task files, provenance sidecars, the input data and answer keys for the execution tier, and the interactive environments for the agentic tier.

## Repository layout

```
L1/tasks/<category>_L1.json        82 knowledge-retrieval tasks (9 files)
L2/tasks/<category>_L2.json        51 scenario-reasoning tasks (9 files)
L3/tasks/<category>_L3.json        31 execution-based coding tasks (9 files)
L4/tasks/<category>_L4.json        10 agentic episodes (9 files)
L{1,2,3}/tasks/*.provenance.json   source-verification sidecars (see "Provenance sidecars")
L{1,2,3}/tasks/*.groundtruth.json  independent re-derivation sidecars
L3/data/                           L3 input data + *_reference.json answer keys
L3/data/raw/                       NASA POWER hourly weather, as downloaded
L3/data/synthetic/                 extended pool of generated L3 instances (not in the 174)
L4/envs/<category>.py              one interactive environment module per category
```

Task files are named `<category>_L<tier>.json`; task ids follow `<category>_L<tier>_<nnn>`.

## Categories

| Slug | Category |
|---|---|
| `vpp_scheduling` | VPP Aggregation & Market Scheduling |
| `forecasting` | DER Forecasting |
| `hosting_capacity` | Hosting Capacity & Interconnection |
| `grid_edge_control` | Grid-Edge & Inverter Control |
| `fault_protection` | Fault Diagnosis, Protection & Islanding |
| `state_estimation` | State Estimation & Load Disaggregation |
| `ev_v2g` | EV / V2G Integration |
| `p2p_trading` | Transactive Energy & P2P Markets |
| `cybersecurity` | Cybersecurity |

Task counts per category and tier:

| | cyber | ev_v2g | fault | forecast | grid_edge | hosting | p2p | state_est | vpp | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 8 | 8 | 8 | 9 | 8 | 8 | 8 | 13 | 12 | **82** |
| L2 | 5 | 5 | 5 | 6 | 5 | 5 | 5 | 8 | 7 | **51** |
| L3 | 3 | 3 | 3 | 4 | 3 | 3 | 3 | 5 | 4 | **31** |
| L4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | **10** |

## Tiers

| Tier | What the model does | Ground truth | Scoring |
|---|---|---|---|
| **L1** Knowledge retrieval | Answers a short domain question | Per-criterion reference answers, each tied to a named authority (standard, textbook, or paper) | `weighted_rubric`: each criterion is checked by the mechanism named in `scored_by`; score = weighted fraction of criteria met |
| **L2** Scenario reasoning | Analyses an inline scenario (`context`) and recommends actions | Per-criterion reference answers anchored to the pinned scenario data | `weighted_rubric`, as L1 |
| **L3** Executable code | Writes a program that loads the shipped data, runs the analysis, and reports named metrics | `L3/data/*_reference.json` answer keys recomputed by solver/simulation | `execution_based`: the program is run in a sandbox and graded on execution success, coverage of `must_report_metrics`, and numerical agreement with the reference within the stated tolerance |
| **L4** Agentic decision-making | Drives an interactive environment through tool calls until the episode ends | Computed at `reset()` by the environment (LP/MILP optimum, power flow, or measured F1/MAPE against hidden labels) | `l4_interactive`: executable outcome score plus hard-constraint compliance from `evaluate()` |

## Task file format

Every task file is `{"metadata": {...}, "tasks": [...]}`; L4 files use `"agentic_tasks"` instead of `"tasks"`. `metadata` records the category, tier, the source platforms the category draws on, and (L4 only) the environment module path.

Fields common to all tasks:

| Field | Meaning |
|---|---|
| `task_id`, `category`, `tier` | Identity |
| `variant` | Always `original` in the task files; other variants are derived from the fields below |
| `prompt` | The text sent to the model |
| `context` | Pinned scenario data (L2), data-file paths and success criteria (L3), or episode parameters (L4); empty for L1 |
| `tools` | Tool names the tier exposes (empty for L1/L2; e.g. `execute_python` for L3) |
| `scoring` | Scoring mechanism: `weighted_rubric`, `execution_based`, or `l4_interactive` |
| `capability_dimensions` | Capability tags the task exercises |
| `use_case`, `role_task` | The practitioner role and workflow the task is anchored to |
| `perturbation_code` | Identifier of the perturbation applied to produce `perturbed_prompt` |

### L1 / L2: `scoring_criteria`

Each L1/L2 task carries a list of criteria (582 in total). A criterion looks like:

```json
{
  "criterion": "Defines a VPP as a cloud/ICT-coordinated aggregation of DER operated as a single dispatchable entity ...",
  "capability": "factual_recall",
  "weight": "vital",
  "authority": {"name": "DOE 'Pathways to Commercial Liftoff: Virtual Power Plants' (2023)", "ref": "https://liftoff.energy.gov/vpp/"},
  "reference_answer": "...",
  "scored_by": "judge_binary"
}
```

- `weight` ∈ {`vital`, `important`, `minor`} (3 / 2 / 1).
- `scored_by` ∈ {`exact_match`, `numeric_match`, `set_coverage`, `judge_binary`}. Numeric and set criteria are checked deterministically; `judge_binary` criteria are decided by an LLM judge against `reference_answer`. `numeric_match` criteria may carry `accept_alternates` for unit-equivalent values.
- Every criterion names an `authority`.

### L1 / L2: prompt variants

Every L1 and L2 task ships three prompts:

| Field | Purpose |
|---|---|
| `prompt` | Original problem |
| `perturbed_prompt` | Surface-form rewrite that preserves every graded quantity (robustness probe) |
| `knowledge_enhanced_prompt` | Original plus explicit domain knowledge (knowledge-integration probe) |

L3 and L4 tasks carry no text variants: rewriting an executable task leaks or truncates solution code.

### L3: `context`, `ground_truth`, `success_criteria`

- `context.<*_file>` — repository-relative paths to the input data (`data_file`, `timeseries_file`, `network_file`, `consumption_file`, `weather_file`). Eight of the 31 tasks read shipped files; the rest synthesise their inputs in code as the prompt specifies.
- `context.data_description`, `context.instructions` — what the data contains and what the program must produce.
- `context.success_criteria` — the machine-checked contract: `must_report_metrics`, numeric thresholds, and a `tolerance` statement tied to the reference values.
- `ground_truth` — prose statement of the expected outcome; `ground_truth_answer` — the scalar reference when one exists (otherwise `null`).
- `metrics`, `official_scoring_standard` — the scoring contract as text.

### L4: episodes

- `metadata.environment` — path of the environment module (e.g. `L4/envs/ev_v2g.py`).
- `context` — episode parameters (fleet, tariff, limits, event definitions) and, where applicable, `computed_optimum`.
- `ground_truth` — description of correct agent behaviour (which tools to call, what to compute, what constraints to respect).
- `reference_papers[]` — `{citation, doi, role}` for the third-party work each environment is modelled on.

## Provenance sidecars

Seven sidecar files sit next to the task files. They are audit trails produced during construction and are **not** read by the evaluation harness.

| File | Entries | Content |
|---|---|---|
| `L1/tasks/cybersecurity_L1.provenance.json` | 5 | source verification |
| `L1/tasks/fault_protection_L1.provenance.json` | 5 | source verification |
| `L2/tasks/forecasting_L2.provenance.json` | 4 | source verification |
| `L3/tasks/fault_protection_L3.provenance.json` | 2 | source verification |
| `L1/tasks/vpp_scheduling_L1.groundtruth.json` | 9 | independent re-derivation of the reference answers |
| `L2/tasks/ev_v2g_L2.groundtruth.json` | 3 | independent re-derivation |
| `L3/tasks/grid_edge_control_L3.groundtruth.json` | 2 | independent re-derivation, with `computed_values` vs `reference_values` from a fresh power-flow run |

`*.provenance.json` entries record `sources` (URL, access date, notes), a `status` (`verified`, `self_contained`, `candidate_unverified`), `answer_source_mismatch`, a `contamination_signal`, the `search_queries` used, and `confidence`. `*.groundtruth.json` entries record a `verdict` (`correct`, `reproduced`), the `evidence`, and `confidence`. All entries carry `verifier: "agent-draft"`: they were produced by an LLM-agent verification workflow and are recorded verbatim; they are not expert sign-off.

## L3 input data (`L3/data/`)

### Shipped inputs

| File | Rows | Columns | Source | Used by |
|---|---|---|---|---|
| `pv_forecast_train_test.csv` | 32,064 (15-min) | GHI, temperature, wind, humidity, PV output, clear-sky GHI | NASA POWER hourly reanalysis, Albuquerque NM (35.2 N, 106.6 W), 2022 | `forecasting_L3_001` |
| `pv_probabilistic_2yr.csv` | 70,074 (15-min) | GHI, DNI, DHI, temperature, cloud cover, PV output, clear-sky GHI | NASA POWER, Golden CO (39.7 N, 105.1 W), 2021–2022 | `forecasting_L3_002` |
| `prosumer_outdoor_temp_hourly.csv` | 8,784 | outdoor temperature | NASA POWER, Sacramento CA (38.57 N, 121.49 W), 2022 | `forecasting_L3_003` |
| `prosumer_household_annual_hourly.csv` | 8,784 | household consumption | 2R2C RC building thermal model driven by the Sacramento weather | `forecasting_L3_003` |
| `prosumer_10homes_hourly.csv` | 22,080 | per-home consumption, outdoor and indoor temperature | 2R2C RC model, 10 HVAC/thermal-mass profiles | `vpp_scheduling_L3_003` |
| `cigre_mv_voltvar_snapshot.json` | — | pandapower network | CIGRE MV benchmark (TF C6.04.02) with PV scaled to an 8 MW high-penetration scenario, midday snapshot | `grid_edge_control_L3_001` |
| `cigre_mv_pv_bess.json` | — | pandapower network | Same network plus two 2 MW / 4 MWh BESS at the largest-PV feeder-end buses | `grid_edge_control_L3_002` |
| `cigre_load_pv_24hr.csv` | 96 (15-min) | load and PV multipliers | 24-h QSTS profile | `grid_edge_control_L3_002` |
| `cybersecurity_inverter_telemetry_1000.csv` | 1,000 | P, Q, V, f, label | Synthetic inverter telemetry with physical P–V–Q correlation and three injected attack classes | `cybersecurity_L3_001` |
| `cybersecurity_replay_telemetry_288.csv` | 288 | P, Q, V, f, SOC, label | Synthetic telemetry with a replayed segment | `cybersecurity_L3_003` |

PV output in the weather files is computed from measured GHI with a temperature-derated DC model (η = 0.185, −0.4 %/°C). Sub-hourly rows are interpolated from the hourly source.

**Temporal relabeling (disclosed).** Calendar years in the weather-derived files are shifted (2022 → 2024; 2021–2022 → 2023–2024) so prompts can reference a consistent study year. The weather is real and physically consistent; the calendar/weekday alignment is nominal (a representative-meteorological-year convention). No claim is made that the data is the actual weather of the relabeled year.

### Answer keys

| File | Reference for |
|---|---|
| `cigre_voltvar_reference.json` | initial/final voltage extremes and overvoltage bus set after volt-var control |
| `cigre_reference_qsts.json` | violations before/after the volt-var → BESS → curtailment mitigation cascade over 24 h |
| `cybersecurity_reference_metrics.json` | precision/recall/F1 of a z-score detector and `IsolationForest(contamination=0.05, n_estimators=200, random_state=42)`, measured on the shipped telemetry |
| `prosumer_household_reference.json` | demand-response savings (RC counterfactual minus metered) and the CalTRACK baseline fit quality |
| `prosumer_10homes_reference.json` | per-home flexibility (kW shed, hours-to-comfort-limit) and the resulting ranking |

Answer keys are computed by simulation or solver, not hand-entered; every generator is seeded, and each key has been re-derived from the shipped files independently of the generator.

### `raw/`

The four NASA POWER downloads (`abq_2022`, `golden_2021`, `golden_2022`, `sacramento_2022`; 8,760 hourly rows each; GHI, clear-sky GHI, DNI, DHI, temperature, wind, humidity), unmodified.

### `synthetic/` — extended pool

Deterministically generated additional L3 instances in the same task-JSON schema: `forecasting_extended.jsonl` (32), `grid_edge_control_extended.jsonl` (30), `hosting_capacity_extended.jsonl` (48), `state_estimation_extended.jsonl` (40), `vpp_scheduling_extended.jsonl` (16), plus `se_artifacts/case33bw_scale{050,080,100,120}.csv` (IEEE 33-bus measurement sets at four load scalings). Each instance varies feeder, load, price, or date and recomputes its ground truth by simulation. These are **not** part of the 174-task set and are not used in the reported leaderboard.

## L4 environments (`L4/envs/`)

One module per category (`vpp_scheduling.py` serves both VPP episodes). Each module defines a subclass of `DERBenchEnv` registered under its task id, exposing:

```
reset() -> observation            # solves the reference (LP/MILP, power flow, or labelled data) for this seed
available_tools() -> [tool specs] # the JSON tool schema the agent may call
step(action) -> (observation, done)
evaluate() -> {score, hard-constraint compliance, ...}
goal_state() -> reference the episode is graded against
```

Ground truth is computed inside the environment at reset — for example the EV depot environment solves the minimum-cost feasible charging schedule with `scipy.optimize.linprog` and grades the agent's schedule against it — so no episode depends on a stored magic constant. Reference values in the task JSON (`context.computed_optimum`) are recorded for readability and match what the environment recomputes.

The modules import `DERBenchEnv` from the evaluation harness (`eval/derbench_env.py`) and are not runnable without it.

## Verification status

- All 174 tasks pass the dataset validator: required fields, presence of both prompt variants on every L1/L2 task, resolution of every `context.*_file` path to a file inside the repository, and per-category tier coverage. Consistency between each criterion's `scored_by` mechanism and its reference type is enforced by a separate lint gate in the evaluation harness.
- L3 answer keys and L4 environment physics are pinned by tests in the evaluation harness.
- Tasks went through a group expert review-and-revision cycle during construction. Per-task independent expert sign-off was not performed; the sidecars above document what was machine- and agent-verified.

## Third-party data

- **NASA POWER** hourly reanalysis (SYN1DEG), NASA Langley Research Center — public domain.
- **CIGRE MV benchmark network** (CIGRE Task Force C6.04.02), as distributed with pandapower.
- **IEEE 33-bus** test feeder (`case33bw`), as distributed with pandapower.

All other data is generated as described above.

## License

Released under a permissive code and data license; see `LICENSE`.

## Citation

The accompanying paper is under double-blind review. Until the camera-ready version is available, please cite this anonymized repository.
