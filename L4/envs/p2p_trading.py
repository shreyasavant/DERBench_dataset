"""
p2p_trading_L4_001 — autonomous single-agent bidding in a 6-round peer-to-peer
energy market.

Grounding
---------
Grounded in the SETTING of LLM-MARL-P2P (arXiv:2507.14995, "LLM-Enhanced
Multi-Agent Reinforcement Learning with Expert Workflow for Real-Time P2P
Energy Trading", Lou et al.): LLM-guided prosumer agents trading peer-to-peer
at 15-minute intervals. The MARKET MECHANISM here is NOT the paper's: the paper
uses no auction (prosumers accept a centrally coordinated P2P price set by the
DSO, and its trained MARL agents decide device dispatch, not bids). This env's
uniform-price double auction over a counterparty order book is DER-Bench's own
construction — a standard textbook P2P clearing mechanism — and the single
zero-training LLM bidding agent replaces the paper's trained MARL policies.

The agent controls one prosumer (P3) while the rest of the market is modeled as
a fixed counterparty order book. The market clears each round with a real,
deterministic UNIFORM-PRICE DOUBLE AUCTION over that book. The counterparty book for each
round is calibrated so that, absent P3, the round clears at the published price
trajectory ($0.09 / $0.13 / $0.07). Because P3 is small (≤2.3 kWh PV surplus +
≤2.5 kWh battery discharge) relative to the book (tens of kWh per side), P3 is
effectively a PRICE-TAKER: a sell offer priced at or below the clearing price
transacts at the (uniform) clearing price for its matched quantity, and a buy
bid priced at or above the clearing price transacts likewise — neither moves the
clearing price materially (documented price-taker assumption).

Grounded reference (computed optimum, no magic constant)
--------------------------------------------------------
P3's optimal accumulated P2P profit is computed at reset by a deterministic
fractional-knapsack over the 6 rounds (the cash-flow-optimal allocation):

  Sellable inventory = PV surplus (2.3 kWh, depletable pool) + battery
    dischargeable energy (SoC 45% → 4.5 kWh stored; usable floor 20% → 2.0 kWh;
    so 2.5 kWh dischargeable), = 4.8 kWh total.
  Per-round offer cap = 2.3 kWh (the published "never bid more than your PV
    surplus per round" limit).
  Selling is worthwhile only where the clearing price exceeds the $0.06 grid
    transaction-cost threshold (so all of rounds 1-6 except none — the cheap
    rounds at $0.07 are marginally above the floor but well below the $0.13
    peak, so they are NOT used while higher-priced capacity remains).
  Each kWh is therefore allocated to the highest-priced rounds first, capped at
    2.3 kWh/round: 2.3 @ $0.13 (R3) + 2.3 @ $0.13 (R4) + 0.2 @ $0.09 = $0.616.
  Buying is never in the profit optimum: there is no later high-price round to
    resell into, so buying spends cash for zero P2P revenue (matching the
    published ground truth, which says abstain in the cheap rounds and only
    consider buying as cost-avoidance, never as a revenue play).

What is real vs simplified
--------------------------
REAL: the uniform-price double-auction clearing over the counterparty book
(merit-order match of demand vs supply curves, marginal-price discovery); P3's
resource bookkeeping (depletable PV-surplus pool, battery SoC drawn down on
discharge / up on charge, 20-90% SoC bounds enforced); the per-round 2.3 kWh
offer cap; accumulated net P2P cash flow (sell revenue − buy cost); and the
computed optimal-profit reference (a real fractional-knapsack solve at reset,
never a hardcoded number).

SIMPLIFIED: the counterparty book is a FIXED synthetic-but-calibrated order book
per round (not other live agents), so its clearing price absent P3 is exactly
the published trajectory; P3 is a price-taker (its small volume does not shift
the clear); a single 15-min interval per round with no inter-round forecasting
of the book beyond the published trajectory hint; battery round-trip efficiency
is taken as unity (energy out = SoC drawn) — the task gives no efficiency, and
the focus is price-timing, not loss accounting.
"""
import warnings

warnings.filterwarnings("ignore")

from derbench_env import DERBenchEnv


@DERBenchEnv.register("p2p_trading_L4_001")
class P2PSingleAgentBiddingEnv(DERBenchEnv):
    """Adaptive 6-round P2P bidding for Prosumer P3, real double-auction clearing."""

    MAX_STEPS = 60

    # ── prosumer P3 profile (from the published task) ─────────────────────────
    _BATT_CAPACITY_KWH = 10.0
    _SOC0_PCT = 45.0
    _SOC_MIN_PCT = 20.0
    _SOC_MAX_PCT = 90.0
    _PV_SURPLUS_KWH = 2.3
    _LOAD_KW = 1.1
    _GRID_BUY = 0.15
    _GRID_SELL = 0.06          # transaction-cost / sell-worthwhile threshold
    _PER_ROUND_OFFER_CAP_KWH = 2.3

    _TOTAL_ROUNDS = 6
    _ROUND_DURATION_MIN = 15

    # published clearing-price trajectory (the counterparty book is calibrated so
    # that, absent P3, each round clears at exactly these prices).
    _TRAJECTORY = {1: 0.09, 2: 0.09, 3: 0.13, 4: 0.13, 5: 0.07, 6: 0.07}
    _MARKET_CONDITION = {
        1: "sellers competitive, below grid buy rate",
        2: "sellers competitive, below grid buy rate",
        3: "sellers dominant, fewer offers",
        4: "sellers dominant, fewer offers",
        5: "large buyer enters, oversupply response drives price down",
        6: "large buyer enters, oversupply response drives price down",
    }

    # ── derived resource bounds ───────────────────────────────────────────────

    def _stored_kwh(self, soc_pct: float) -> float:
        return soc_pct / 100.0 * self._BATT_CAPACITY_KWH

    def _battery_dischargeable_kwh(self) -> float:
        """Energy the battery can release before hitting the 20% SoC floor."""
        return max(0.0, self._stored_kwh(self._soc_pct)
                   - self._stored_kwh(self._SOC_MIN_PCT))

    def _battery_chargeable_kwh(self) -> float:
        """Energy the battery can absorb before hitting the 90% SoC ceiling."""
        return max(0.0, self._stored_kwh(self._SOC_MAX_PCT)
                   - self._stored_kwh(self._soc_pct))

    def _sellable_now_kwh(self) -> float:
        """Total energy P3 could offer to sell right now: remaining PV-surplus
        pool plus what the battery can discharge above the floor."""
        return self._pv_surplus_remaining + self._battery_dischargeable_kwh()

    # ── counterparty order book (calibrated to the trajectory) ────────────────

    def _counterparty_book(self, rnd: int) -> dict:
        """Fixed buy/sell ladders whose uniform-price clear (absent P3) equals
        the published trajectory price for `rnd`.

        The book is a small merit-order ladder on each side. The marginal
        matched pair sits at the trajectory price, so the uniform clearing price
        is exactly _TRAJECTORY[rnd]. P3's ≤2.3 kWh is small versus the ~30 kWh
        of book depth at the margin, so P3 is a price-taker."""
        p = self._TRAJECTORY[rnd]
        # Demand (buy) curve: willing buyers at descending prices.
        # Supply (sell) curve: willing sellers at ascending prices.
        # Calibrated so the marginal cleared pair meets exactly at `p`.
        buys = [
            {"price": round(p + 0.06, 4), "qty": 8.0},
            {"price": round(p + 0.03, 4), "qty": 12.0},
            {"price": round(p, 4), "qty": 15.0},
        ]
        sells = [
            {"price": round(p - 0.04, 4), "qty": 10.0},
            {"price": round(p - 0.02, 4), "qty": 10.0},
            {"price": round(p, 4), "qty": 15.0},
        ]
        return {"buys": buys, "sells": sells}

    @staticmethod
    def _clear_uniform_price(buys: list, sells: list):
        """Uniform-price double-auction clear of a demand/supply order book.

        Returns (clearing_price, cleared_qty). Buys are sorted descending by
        price (highest willingness-to-pay first), sells ascending (cheapest
        first); we walk both ladders accumulating matched quantity while
        buy_price >= sell_price, and the clearing price is the midpoint of the
        last matched marginal pair (a standard uniform-price rule)."""
        b = sorted(buys, key=lambda x: -x["price"])
        s = sorted(sells, key=lambda x: x["price"])
        i = j = 0
        bq = b[0]["qty"] if b else 0.0
        sq = s[0]["qty"] if s else 0.0
        matched = 0.0
        last_b = last_s = None
        while i < len(b) and j < len(s) and b[i]["price"] >= s[j]["price"]:
            take = min(bq, sq)
            matched += take
            last_b, last_s = b[i]["price"], s[j]["price"]
            bq -= take
            sq -= take
            if bq <= 1e-12:
                i += 1
                bq = b[i]["qty"] if i < len(b) else 0.0
            if sq <= 1e-12:
                j += 1
                sq = s[j]["qty"] if j < len(s) else 0.0
        if last_b is None:
            return None, 0.0
        price = round((last_b + last_s) / 2.0, 4)
        return price, round(matched, 4)

    # ── env lifecycle ─────────────────────────────────────────────────────────

    def _reset_impl(self) -> dict:
        self._round = 1
        self._soc_pct = self._SOC0_PCT
        self._pv_surplus_remaining = self._PV_SURPLUS_KWH
        self._revenue = 0.0           # accumulated net P2P cash flow (sell − buy)
        self._pending_bids: dict[int, dict] = {}   # bid_id -> bid (this round)
        self._next_bid_id = 1
        self._round_log: dict[int, list] = {}
        self._cleared_rounds: list[dict] = []
        self._violations: list[dict] = []          # constraint breaches (gating)
        self._sold_by_round: dict[int, float] = {}
        self._declared = False

        # grounded reference: optimal accumulated profit (real knapsack solve)
        self._optimal_profit, self._optimal_plan = self._solve_optimum()

        return {
            "scenario": "Autonomous bidding agent for Prosumer P3 in a 6-round "
                        "peer-to-peer energy market (each round = a 15-min "
                        "interval). Adapt your bidding as the clearing price "
                        "shifts across the rounds.",
            "prosumer_id": "P3",
            "total_rounds": self._TOTAL_ROUNDS,
            "round_duration_min": self._ROUND_DURATION_MIN,
            "current_round": self._round,
            "battery_soc_pct": self._soc_pct,
            "battery_capacity_kwh": self._BATT_CAPACITY_KWH,
            "soc_bounds_pct": [self._SOC_MIN_PCT, self._SOC_MAX_PCT],
            "pv_surplus_kwh": self._pv_surplus_remaining,
            "current_load_kw": self._LOAD_KW,
            "grid_buy_rate": self._GRID_BUY,
            "grid_sell_rate": self._GRID_SELL,
            "transaction_cost_threshold": self._GRID_SELL,
            "per_round_offer_cap_kwh": self._PER_ROUND_OFFER_CAP_KWH,
            "note": "Submit sell offers / buy bids each round, then call "
                    "clear_round() to clear the market for that round. Selling "
                    "is only worthwhile when the clearing price exceeds "
                    f"${self._GRID_SELL}/kWh. The episode runs "
                    f"{self._TOTAL_ROUNDS} rounds.",
        }

    def _solve_optimum(self):
        """Cash-flow-optimal accumulated P2P profit over the 6 rounds.

        Deterministic fractional-knapsack: allocate the depletable sellable
        inventory (PV surplus + battery dischargeable, capped at the per-round
        offer cap) to the highest-priced rounds whose price clears the $0.06
        threshold. Buying is never profitable (no later resale round), so it is
        excluded. Returns (optimal_profit_usd, {round: kwh_sold}).

        NOTE on excluding buy-side arbitrage: it is safe to ignore "buy cheap,
        resell dear" ONLY because native sellable inventory (4.8 kWh) already
        saturates the two high-price $0.13 rounds at the 2.3 kWh/round cap
        (2.3+2.3 = 4.6 kWh of capacity, fully filled). With no spare high-price
        capacity left, energy bought low has nowhere more valuable to be sold,
        so a buy-then-resell leg cannot beat this allocation (the reviewer
        verified buy-arbitrage tops out at $0.515 < $0.616). If the trajectory
        or caps are ever edited so the high-price rounds are NOT saturated by
        native inventory, this knapsack would no longer be the true optimum and
        must be re-derived to include the buy leg."""
        inventory = self._PV_SURPLUS_KWH + self._battery_dischargeable_kwh()
        rounds_desc = sorted(self._TRAJECTORY,
                             key=lambda r: -self._TRAJECTORY[r])
        remaining = inventory
        profit = 0.0
        plan: dict[int, float] = {}
        for r in rounds_desc:
            price = self._TRAJECTORY[r]
            if price <= self._GRID_SELL + 1e-12:
                continue
            q = min(self._PER_ROUND_OFFER_CAP_KWH, remaining)
            if q <= 1e-12:
                break
            plan[r] = round(q, 6)
            profit += q * price
            remaining -= q
        return round(profit, 6), plan

    # ── observation helpers ───────────────────────────────────────────────────

    def _market_state(self) -> dict:
        rnd = self._round
        if rnd > self._TOTAL_ROUNDS:
            return {"market_closed": True, "round": rnd,
                    "total_rounds": self._TOTAL_ROUNDS}
        book = self._counterparty_book(rnd)
        last = self._cleared_rounds[-1]["clearing_price"] if self._cleared_rounds else None
        dominant = ("sellers" if rnd in (3, 4) else
                    "buyers" if rnd in (5, 6) else "balanced")
        return {
            "round": rnd,
            "total_rounds": self._TOTAL_ROUNDS,
            "order_book": book,
            "last_clearing_price": last,
            # Deliberate hint, not a leak: the published prompt already gives the
            # full price trajectory ($0.09/$0.13/$0.07 by round), so exposing the
            # current round's indicative clearing price here only surfaces
            # information the agent is told up front. The "adaptation under
            # uncertainty" framing is about timing the KNOWN trajectory (hold for
            # the $0.13 rounds, abstain at the $0.07 floor), not about forecasting
            # an unknown price.
            "indicative_clearing_price": self._TRAJECTORY[rnd],
            "dominant_side": dominant,
            "market_condition": self._MARKET_CONDITION[rnd],
        }

    def _my_status(self) -> dict:
        return {
            "round": self._round,
            "battery_soc_pct": round(self._soc_pct, 4),
            "battery_dischargeable_kwh": round(self._battery_dischargeable_kwh(), 4),
            "battery_chargeable_kwh": round(self._battery_chargeable_kwh(), 4),
            "pv_surplus_remaining_kwh": round(self._pv_surplus_remaining, 4),
            "sellable_now_kwh": round(self._sellable_now_kwh(), 4),
            "current_load_kw": self._LOAD_KW,
            "accumulated_p2p_revenue_usd": round(self._revenue, 4),
        }

    # ── tools ─────────────────────────────────────────────────────────────────

    def _tools(self) -> list[dict]:
        return [
            {"name": "get_market_state",
             "description": "Current round number, the active counterparty order "
                            "book (buys/sells with price & qty), the last "
                            "clearing price ($/kWh), the indicative clearing "
                            "price for this round, and the dominant market side.",
             "parameters": {}},
            {"name": "get_my_status",
             "description": "P3's battery SoC (%), dischargeable/chargeable "
                            "energy (kWh), remaining PV surplus (kWh), current "
                            "load (kW), and accumulated P2P revenue ($) so far.",
             "parameters": {}},
            {"name": "submit_bid",
             "description": "Submit a sell offer (type='sell') or buy bid "
                            "(type='buy') for quantity_kwh at price_per_kwh for "
                            "the CURRENT round. A sell offer is rejected if "
                            "quantity_kwh exceeds the 2.3 kWh per-round offer cap "
                            "or the currently sellable energy (PV surplus + "
                            "battery dischargeable above the 20% floor); a buy "
                            "bid is rejected if it exceeds the battery's "
                            "chargeable headroom to 90% SoC. Returns a bid_id.",
             "parameters": {"type": {"type": "string", "enum": ["buy", "sell"]},
                            "quantity_kwh": {"type": "number"},
                            "price_per_kwh": {"type": "number"}}},
            {"name": "withdraw_bid",
             "description": "Withdraw a previously submitted (un-cleared) bid for "
                            "the current round by its bid_id.",
             "parameters": {"bid_id": {"type": "integer"}}},
            {"name": "log_strategy",
             "description": "Record strategic reasoning for the current round "
                            "(what market signals drove the decision, any change "
                            "from the previous round).",
             "parameters": {"reasoning": {"type": "string"}}},
            {"name": "clear_round",
             "description": "Clear the market for the current round: P3's "
                            "submitted bids are matched against the counterparty "
                            "book at the uniform clearing price, SoC / PV surplus "
                            "/ revenue update, and the round advances. After "
                            "round 6 clears the episode ends.",
             "parameters": {}},
            {"name": "declare_done",
             "description": "End the episode early; the accumulated P2P profit is "
                            "then graded against the computed optimum.",
             "parameters": {}},
        ]

    def _dispatch_tool(self, tool: str, args: dict):
        if tool == "get_market_state":
            return self._market_state()
        if tool == "get_my_status":
            return self._my_status()
        if tool == "submit_bid":
            return self._tool_submit_bid(
                str(args.get("type", "")).lower(),
                float(args.get("quantity_kwh", 0.0)),
                float(args.get("price_per_kwh", 0.0)))
        if tool == "withdraw_bid":
            return self._tool_withdraw(int(args.get("bid_id", -1)))
        if tool == "log_strategy":
            self._round_log.setdefault(self._round, []).append(
                str(args.get("reasoning", "")))
            return {"logged": True, "round": self._round,
                    "log_entries_this_round": len(self._round_log[self._round])}
        if tool == "clear_round":
            return self._tool_clear_round()
        if tool == "declare_done":
            self._declared = True
            self._done = True
            return {"status": "episode_complete",
                    "rounds_cleared": len(self._cleared_rounds),
                    "accumulated_p2p_revenue_usd": round(self._revenue, 4)}
        raise ValueError(f"Unknown tool: {tool}")

    def _tool_submit_bid(self, btype: str, qty: float, price: float):
        if self._round > self._TOTAL_ROUNDS:
            return {"rejected": True, "reason": "market closed (all 6 rounds cleared)"}
        if btype not in ("buy", "sell"):
            return {"rejected": True, "reason": f"type must be 'buy' or 'sell', got {btype!r}"}
        if qty <= 0:
            return {"rejected": True, "reason": "quantity_kwh must be positive"}
        if price < 0:
            return {"rejected": True, "reason": "price_per_kwh must be non-negative"}

        if btype == "sell":
            # already-pending sell volume this round counts against both the
            # per-round offer cap and the sellable-energy headroom.
            pending_sell = sum(b["qty"] for b in self._pending_bids.values()
                               if b["type"] == "sell")
            # per-round offer cap applies to the AGGREGATE of sell bids in a
            # round, not just each individual bid (else two sub-cap bids could
            # collectively exceed the cap and beat the computed optimum).
            if pending_sell + qty > self._PER_ROUND_OFFER_CAP_KWH + 1e-9:
                return {"rejected": True, "type": "sell", "quantity_kwh": qty,
                        "reason": f"quantity_kwh={qty} would bring this round's "
                                  f"total sell offers to "
                                  f"{round(pending_sell + qty, 4)} kWh, exceeding "
                                  f"the per-round offer cap of "
                                  f"{self._PER_ROUND_OFFER_CAP_KWH} kWh",
                        "pending_sell_this_round_kwh": round(pending_sell, 4),
                        "per_round_offer_cap_kwh": self._PER_ROUND_OFFER_CAP_KWH}
            sellable = self._sellable_now_kwh() - pending_sell
            if qty > sellable + 1e-9:
                return {"rejected": True, "type": "sell", "quantity_kwh": qty,
                        "reason": f"quantity_kwh={qty} exceeds sellable energy "
                                  f"{round(sellable, 4)} kWh (PV surplus + battery "
                                  f"dischargeable above the 20% SoC floor, net of "
                                  f"pending offers)",
                        "sellable_now_kwh": round(sellable, 4)}
        else:  # buy
            pending_buy = sum(b["qty"] for b in self._pending_bids.values()
                              if b["type"] == "buy")
            room = self._battery_chargeable_kwh() - pending_buy
            if qty > room + 1e-9:
                return {"rejected": True, "type": "buy", "quantity_kwh": qty,
                        "reason": f"quantity_kwh={qty} exceeds battery chargeable "
                                  f"headroom {round(room, 4)} kWh (to the 90% SoC "
                                  f"ceiling, net of pending bids)",
                        "battery_chargeable_kwh": round(room, 4)}

        bid_id = self._next_bid_id
        self._next_bid_id += 1
        self._pending_bids[bid_id] = {"type": btype, "qty": qty, "price": price}
        return {"accepted": True, "bid_id": bid_id, "type": btype,
                "quantity_kwh": qty, "price_per_kwh": price, "round": self._round}

    def _tool_withdraw(self, bid_id: int):
        if bid_id in self._pending_bids:
            removed = self._pending_bids.pop(bid_id)
            return {"withdrawn": True, "bid_id": bid_id, "bid": removed}
        return {"withdrawn": False, "bid_id": bid_id,
                "reason": "no such pending bid in the current round"}

    def _tool_clear_round(self):
        rnd = self._round
        if rnd > self._TOTAL_ROUNDS:
            return {"rejected": True, "reason": "market already closed"}

        book = self._counterparty_book(rnd)
        clearing_price, _book_qty = self._clear_uniform_price(book["buys"], book["sells"])
        if clearing_price is None:
            clearing_price = self._TRAJECTORY[rnd]

        sold_kwh = 0.0
        bought_kwh = 0.0
        round_cash = 0.0
        fills = []
        for bid_id, bid in sorted(self._pending_bids.items()):
            btype, qty, bprice = bid["type"], bid["qty"], bid["price"]
            if btype == "sell":
                # P3 is a price-taker: a sell offer at or below the clearing
                # price transacts at the uniform clearing price, bounded by the
                # energy actually still available this round.
                if bprice <= clearing_price + 1e-12:
                    # belt-and-suspenders: bound the fill by BOTH remaining
                    # sellable inventory AND the per-round offer cap, so the
                    # aggregate cleared sell in a round can never exceed the cap
                    # even if a malformed bid slipped past submit-time checks.
                    cap_room = self._PER_ROUND_OFFER_CAP_KWH - sold_kwh
                    avail = min(self._sellable_now_kwh() - sold_kwh, cap_room)
                    fill = max(0.0, min(qty, avail))
                    # draw PV surplus first, then battery
                    pv_draw = min(fill, self._pv_surplus_remaining)
                    batt_draw = fill - pv_draw
                    self._pv_surplus_remaining -= pv_draw
                    if batt_draw > 0:
                        self._soc_pct -= batt_draw / self._BATT_CAPACITY_KWH * 100.0
                    sold_kwh += fill
                    round_cash += fill * clearing_price
                    fills.append({"bid_id": bid_id, "type": "sell",
                                  "filled_kwh": round(fill, 4),
                                  "price": clearing_price})
                else:
                    fills.append({"bid_id": bid_id, "type": "sell",
                                  "filled_kwh": 0.0,
                                  "reason": "offer price above clearing price"})
            else:  # buy
                if bprice >= clearing_price - 1e-12:
                    room = self._battery_chargeable_kwh() - bought_kwh
                    fill = max(0.0, min(qty, room))
                    self._soc_pct += fill / self._BATT_CAPACITY_KWH * 100.0
                    bought_kwh += fill
                    round_cash -= fill * clearing_price
                    fills.append({"bid_id": bid_id, "type": "buy",
                                  "filled_kwh": round(fill, 4),
                                  "price": clearing_price})
                else:
                    fills.append({"bid_id": bid_id, "type": "buy",
                                  "filled_kwh": 0.0,
                                  "reason": "bid price below clearing price"})

        # over-cap gating: if the agent attempted to clear more than the
        # per-round offer cap of sell volume in this round, flag it. We compare
        # the total requested clearing-eligible sell quantity against the cap
        # (the fill itself is hard-bounded above, but the gating must still
        # fire so the score multiplier the docstring/JSON promise applies).
        requested_sell = sum(
            bid["qty"] for bid in self._pending_bids.values()
            if bid["type"] == "sell" and bid["price"] <= clearing_price + 1e-12)
        if requested_sell > self._PER_ROUND_OFFER_CAP_KWH + 1e-9:
            self._violations.append(
                {"round": rnd, "type": "over_cap",
                 "requested_sell_kwh": round(requested_sell, 4),
                 "per_round_offer_cap_kwh": self._PER_ROUND_OFFER_CAP_KWH})

        # record any sell below the transaction-cost threshold (gating signal)
        for bid in self._pending_bids.values():
            if (bid["type"] == "sell" and bid["price"] <= clearing_price + 1e-12
                    and clearing_price <= self._GRID_SELL + 1e-9 and bid["qty"] > 1e-9):
                self._violations.append(
                    {"round": rnd, "type": "sell_below_threshold",
                     "clearing_price": clearing_price,
                     "threshold": self._GRID_SELL})
                break

        self._revenue += round_cash
        self._sold_by_round[rnd] = round(sold_kwh, 6)
        self._cleared_rounds.append({
            "round": rnd, "clearing_price": clearing_price,
            "p3_sold_kwh": round(sold_kwh, 4), "p3_bought_kwh": round(bought_kwh, 4),
            "round_cash_usd": round(round_cash, 4),
            "soc_pct_after": round(self._soc_pct, 4),
            "pv_surplus_after": round(self._pv_surplus_remaining, 4),
            "fills": fills,
        })
        self._pending_bids = {}
        self._round += 1
        if self._round > self._TOTAL_ROUNDS:
            self._done = True

        return {
            "cleared_round": rnd, "clearing_price": clearing_price,
            "p3_sold_kwh": round(sold_kwh, 4), "p3_bought_kwh": round(bought_kwh, 4),
            "round_cash_usd": round(round_cash, 4),
            "accumulated_p2p_revenue_usd": round(self._revenue, 4),
            "battery_soc_pct": round(self._soc_pct, 4),
            "pv_surplus_remaining_kwh": round(self._pv_surplus_remaining, 4),
            "next_round": self._round if self._round <= self._TOTAL_ROUNDS else None,
            "market_closed": self._round > self._TOTAL_ROUNDS,
            "fills": fills,
        }

    # ── scoring ─────────────────────────────────────────────────────────────────

    def _evaluate_impl(self) -> tuple[float, dict]:
        """Deterministic score in [0,1] reflecting realized P2P profit vs optimum.

        primary (0.80): realized accumulated P2P profit / computed optimum;
        adaptation credit (0.20): sold into the high-price ($0.13) rounds and
          did NOT dump inventory into the cheap ($0.07) rounds;
        constraint gating: any SoC-bound breach, over-cap offer, or sell below
          the $0.06 transaction threshold multiplies the whole score by 0.3.
        """
        profit = self._revenue
        opt = self._optimal_profit

        # primary: profit fraction (clamped; negative profit -> 0)
        if opt > 1e-9:
            profit_frac = max(0.0, min(1.0, profit / opt))
        else:
            profit_frac = 1.0 if profit >= -1e-9 else 0.0
        primary = 0.80 * profit_frac

        # adaptation credit
        high_sold = sum(self._sold_by_round.get(r, 0.0) for r in (3, 4))
        cheap_sold = sum(self._sold_by_round.get(r, 0.0) for r in (5, 6))
        # optimal high-round volume is min(inventory, 2*cap); reward filling it
        opt_high = sum(self._optimal_plan.get(r, 0.0) for r in (3, 4))
        adaptation = 0.0
        if opt_high > 1e-9:
            adaptation += 0.15 * max(0.0, min(1.0, high_sold / opt_high))
        # penalize dumping into the cheap rounds (proportional to cheap volume)
        if cheap_sold <= 1e-9:
            adaptation += 0.05
        score = primary + adaptation

        # SoC-bound check on the realized trajectory
        soc_violation = (self._soc_pct < self._SOC_MIN_PCT - 1e-6
                         or self._soc_pct > self._SOC_MAX_PCT + 1e-6)
        gated = soc_violation or len(self._violations) > 0
        if gated:
            score *= 0.3

        score = max(0.0, min(1.0, score))

        return score, {
            "realized_p2p_profit_usd": round(profit, 4),
            "optimal_p2p_profit_usd": round(opt, 4),
            "profit_fraction": round(profit_frac, 4),
            "optimal_plan_kwh_by_round": self._optimal_plan,
            "sold_kwh_by_round": {r: round(v, 4) for r, v in self._sold_by_round.items()},
            "high_price_round_sold_kwh": round(high_sold, 4),
            "cheap_round_sold_kwh": round(cheap_sold, 4),
            "final_soc_pct": round(self._soc_pct, 4),
            "soc_bound_violation": bool(soc_violation),
            "threshold_violations": self._violations,
            "rounds_cleared": len(self._cleared_rounds),
            "declared_done": self._declared,
            "primary_score": round(primary, 4),
            "adaptation_score": round(adaptation, 4),
            "constraint_gated": bool(gated),
            "clearing_prices": [c["clearing_price"] for c in self._cleared_rounds],
        }

    def _task_description(self) -> str:
        return (
            "You are an autonomous bidding agent for Prosumer P3 in a 6-round "
            "peer-to-peer energy market (each round is a 15-minute interval). "
            "P3 has a 10 kWh battery at 45% SoC (usable 20-90%, so 2.5 kWh "
            "dischargeable and 4.5 kWh chargeable headroom), 2.3 kWh of PV "
            "surplus to sell, and a 1.1 kW load. Grid buy = $0.15/kWh, grid sell "
            "= $0.06/kWh; selling into the P2P market is only worthwhile when the "
            "clearing price exceeds the $0.06 grid floor. Each round: read the "
            "market with get_market_state and your resources with get_my_status, "
            "decide whether to sell surplus, buy to charge, or abstain, submit "
            "offers/bids with submit_bid (never more than 2.3 kWh per round and "
            "never more than your sellable energy or battery headroom), log your "
            "reasoning with log_strategy, then call clear_round to clear the "
            "market and advance. Maximise accumulated P2P profit by selling into "
            "the high-price rounds and abstaining when the price falls back to "
            "the grid floor. The episode ends after round 6 (or call "
            "declare_done).")

    def _goal_state(self) -> dict:
        return {
            "maximize": "accumulated_p2p_profit_usd",
            "optimal_p2p_profit_usd": self._optimal_profit,
            "optimal_plan_kwh_by_round": self._optimal_plan,
            "per_round_offer_cap_kwh": self._PER_ROUND_OFFER_CAP_KWH,
            "soc_bounds_pct": [self._SOC_MIN_PCT, self._SOC_MAX_PCT],
            "sell_only_above_price": self._GRID_SELL,
            "sell_into_rounds": [3, 4],
            "abstain_in_rounds": [5, 6],
        }
