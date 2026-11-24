// Flow-breakout scalper for the pump.fun replay sprint (agent-fun/replay/v2).
//
// Idea: on a 24-hour window the launches never stop, so this bot waits for
// maturity (100-2400 slots after creation) and only buys curves whose last 40
// slots show sustained, broad-based buying — at least 1 SOL of net buy inflow
// from at least five distinct wallets, the price up 5-120% over the window and
// still within 10% of the window high, real reserves between 2 and 60 SOL, no
// single wallet behind more than half the buys, and a creator who has not sold.
// It buys once per coin (0.05 SOL) and exits fast, because pump.fun winners
// either keep running or reverse within minutes: take half at +20%, let the
// rest run to +60% or a trailing floor, cut at -12%, dump on a -10% one-batch
// crash, time out after 300 slots. Guardrail probes keep the run above 25
// round-tripped coins over the window: when the bot trails the pace it opens a
// tiny 0.01 SOL probe round trip on the most liquid curve it has seen and sells
// it as soon as the buy lands.
//
// Pure function of the batches the harness sends: no clock, no randomness, no
// I/O, integer math only, insertion-ordered maps only — the two graded runs
// produce byte-identical action logs.

import { runBot, type Batch, type BotState, type Done, type OrderInput, type Pubkey } from './vendor/makefun-bot.ts';

const INITIAL_LAMPORTS = 50_000_000n; // 0.05 SOL per coin, one buy only.
const MIN_BUY_LAMPORTS = 10_000_000n; // 0.01 SOL: the guardrail's round-trip floor.
const PROBE_LAMPORTS = 10_000_000n; // 0.01 SOL probe round trips.
const BUY_SLIPPAGE_BPS = 1500; // Reject a buy filled >15% worse than decided.
const SELL_SLIPPAGE_BPS = 10_000; // Exits always execute.
const MIN_AGE_SLOTS = 100; // Only mature curves.
const MAX_AGE_SLOTS = 2400; // About 16 minutes after creation.
const FLOW_WINDOW_SLOTS = 40;
const MIN_NET_SOL_LAMPORTS = 1_000_000_000n; // 1 SOL net buy inflow in the window.
const MIN_BUYERS = 5;
const MAX_BUYER_SHARE_PCT = 50; // Largest single buyer at most 50% of buy SOL.
const MIN_REAL_LAMPORTS = 2_000_000_000n; // 2 SOL of real reserves.
const MAX_REAL_LAMPORTS = 60_000_000_000n; // 60 SOL of real reserves.
const MAX_MAIN_POSITIONS = 4; // One of the five slots stays free for probes.
const TARGET_COINS = 25;
const PACE_SECONDS = 86_400; // Pace the 25 coins over the full 24 hours.
const MAX_PROBES = 40;
const MAX_HOLD_SLOTS = 300; // ~2 minutes: time out flat positions fast.
const PROBE_MIN_REAL_LAMPORTS = 10_000_000_000n; // Probes only on liquid curves.

interface TapeTrade {
  slot: number;
  buy: boolean;
  user: string;
  sol: bigint;
  vq: bigint;
  vt: bigint;
}

interface Tape {
  createdSlot: number | null;
  creator: string;
  creatorSold: boolean;
  complete: boolean;
  decided: boolean;
  trades: TapeTrade[];
}

interface Held {
  cost: bigint;
  tokens: bigint;
  boughtSlot: number;
  peak: bigint;
  tp1Done: boolean;
  lastVq: bigint;
  lastVt: bigint;
  probe: boolean;
}

const tapes = new Map<Pubkey, Tape>();
const held = new Map<Pubkey, Held>();
const probeMints = new Set<Pubkey>();
const boughtCoins = new Set<Pubkey>();
const roundTripped = new Set<Pubkey>();
let probeCount = 0;
let firstBlockTime: number | null = null;
let lastBlockTime: number | null = null;

/** Current price of a curve as (virtual SOL, virtual tokens); null when unknown. */
function priceOf(state: BotState, mint: Pubkey): [bigint, bigint] | null {
  const curve = state.curves.get(mint);
  if (curve === undefined) return null;
  if (curve.virtual_token_reserves === 0n) return null;
  return [curve.virtual_quote_reserves, curve.virtual_token_reserves];
}

/** Token value in lamports at the given price. */
function valueAt(tokens: bigint, vq: bigint, vt: bigint): bigint {
  return (tokens * vq) / vt;
}

/** True when a/b >= num/den, all bigint, no floats. */
function atLeast(a: bigint, b: bigint, num: bigint, den: bigint): boolean {
  return a * den >= b * num;
}

/** True when a/b <= num/den, all bigint, no floats. */
function atMost(a: bigint, b: bigint, num: bigint, den: bigint): boolean {
  return a * den <= b * num;
}

/** Flow stats over a tape's recent trades. */
function windowStats(tape: Tape, nowSlot: number): {
  netSol: bigint;
  buySol: bigint;
  buyers: number;
  maxBuyerShareOk: boolean;
  startVq: bigint;
  startVt: bigint;
  highVq: bigint;
  highVt: bigint;
  tradeCount: number;
} | null {
  const from = nowSlot - FLOW_WINDOW_SLOTS;
  let netSol = 0n;
  let buySol = 0n;
  const buyers = new Set<string>();
  const perBuyer = new Map<string, bigint>();
  let maxBuyerSol = 0n;
  let startVq = 0n;
  let startVt = 0n;
  let highVq = 0n;
  let highVt = 0n;
  let tradeCount = 0;
  for (const t of tape.trades) {
    if (t.slot <= from) continue;
    tradeCount += 1;
    if (startVq === 0n) {
      startVq = t.vq;
      startVt = t.vt;
    }
    // Track the window high price by cross-multiplication.
    if (highVq === 0n || t.vq * highVt > highVq * t.vt) {
      highVq = t.vq;
      highVt = t.vt;
    }
    if (t.buy) {
      buyers.add(t.user);
      buySol += t.sol;
      netSol += t.sol;
      const total = (perBuyer.get(t.user) ?? 0n) + t.sol;
      perBuyer.set(t.user, total);
      if (total > maxBuyerSol) maxBuyerSol = total;
    } else {
      netSol -= t.sol;
    }
  }
  if (tradeCount === 0 || startVt === 0n || highVt === 0n) return null;
  return {
    netSol,
    buySol,
    buyers: buyers.size,
    maxBuyerShareOk: maxBuyerSol * 100n <= buySol * BigInt(MAX_BUYER_SHARE_PCT),
    startVq,
    startVt,
    highVq,
    highVt,
    tradeCount,
  };
}

await runBot({
  onBatch(state: BotState, batch: Batch): OrderInput[] {
    const orders: OrderInput[] = [];
    const txFee = state.txFeeLamports();

    // Block time paces the guardrail probes; it only moves when events carry it.
    for (const e of batch.events) {
      if (e.block_time !== null && e.block_time !== undefined) {
        if (firstBlockTime === null) firstBlockTime = e.block_time;
        lastBlockTime = e.block_time;
      }
    }

    // Fold the batch's events into the per-coin tapes.
    for (const e of batch.events) {
      if (e.event === 'CreateEvent') {
        if (!tapes.has(e.mint)) {
          tapes.set(e.mint, {
            createdSlot: e.slot,
            creator: e.creator,
            creatorSold: false,
            complete: false,
            decided: false,
            trades: [],
          });
        }
      } else if (e.event === 'TradeEvent') {
        let tape = tapes.get(e.mint);
        if (tape === undefined) {
          tape = {
            createdSlot: null,
            creator: '',
            creatorSold: false,
            complete: false,
            decided: false,
            trades: [],
          };
          tapes.set(e.mint, tape);
        }
        tape.trades.push({
          slot: e.slot,
          buy: e.is_buy,
          user: e.user,
          sol: BigInt(e.sol_amount),
          vq: BigInt(e.virtual_sol_reserves),
          vt: BigInt(e.virtual_token_reserves),
        });
        if (tape.trades.length > 200) tape.trades.splice(0, tape.trades.length - 200);
        if (!e.is_buy && e.user === tape.creator) tape.creatorSold = true;
      } else if (e.event === 'CompleteEvent') {
        const tape = tapes.get(e.mint);
        if (tape !== undefined) tape.complete = true;
      }
    }

    // Fills: track which coins got a buy (they can round-trip) and which closed.
    for (const f of batch.fills) {
      if (f.side === 'buy' && f.status === 'filled') boughtCoins.add(f.mint);
      if (f.side === 'sell' && boughtCoins.has(f.mint)) roundTripped.add(f.mint);
    }

    // Sync held positions with the account; a position appears once its buy fills.
    // A partial sell shrinks the tokens held, so rescale the peak with it: the
    // trailing stop compares the value of the tokens still held against their
    // own best value, never against the pre-sale full position.
    for (const [mint, pos] of state.positions) {
      const existing = held.get(mint);
      if (existing === undefined) {
        const price = priceOf(state, mint);
        const value = price !== null ? valueAt(pos.token_amount, price[0], price[1]) : pos.cost_lamports;
        held.set(mint, {
          cost: pos.cost_lamports,
          tokens: pos.token_amount,
          boughtSlot: state.slot,
          peak: value,
          tp1Done: false,
          lastVq: price !== null ? price[0] : 0n,
          lastVt: price !== null ? price[1] : 0n,
          probe: probeMints.has(mint),
        });
      } else {
        if (pos.token_amount < existing.tokens && existing.tokens > 0n) {
          existing.peak = (existing.peak * pos.token_amount) / existing.tokens;
        }
        existing.cost = pos.cost_lamports;
        existing.tokens = pos.token_amount;
      }
    }
    for (const mint of [...held.keys()]) {
      if (!state.positions.has(mint)) {
        held.delete(mint);
      }
    }

    const pendingSell = new Set<Pubkey>();
    for (const p of state.pending.values()) {
      if (p.order.side === 'sell') pendingSell.add(p.order.mint);
    }

    // Exits, one order per coin per batch. Priority: probe close, crash, stop,
    // runner (already past first target), first target, time-out.
    for (const [mint, h] of held) {
      if (pendingSell.has(mint)) continue;
      const curve = state.curves.get(mint);
      if (curve === undefined || curve.complete) continue;
      const price = priceOf(state, mint);
      if (price === null) continue;
      const pos = state.positions.get(mint);
      if (pos === undefined) continue;
      const [vq, vt] = price;
      const value = valueAt(pos.token_amount, vq, vt);

      if (h.probe) {
        orders.push({ side: 'sell', mint, token_amount: pos.token_amount, max_slippage_bps: SELL_SLIPPAGE_BPS });
        continue;
      }
      // One-batch crash: price down 10% or more since the last batch.
      if (h.lastVq > 0n && h.lastVt > 0n && atMost(vq, vt, h.lastVq * 90n, h.lastVt * 100n)) {
        orders.push({ side: 'sell', mint, token_amount: pos.token_amount, max_slippage_bps: SELL_SLIPPAGE_BPS });
      } else if (atMost(value, h.cost, 88n, 100n)) {
        orders.push({ side: 'sell', mint, token_amount: pos.token_amount, max_slippage_bps: SELL_SLIPPAGE_BPS });
      } else if (h.tp1Done && (atLeast(value, h.cost, 160n, 100n) || atMost(value, h.peak, 80n, 100n))) {
        orders.push({ side: 'sell', mint, token_amount: pos.token_amount, max_slippage_bps: SELL_SLIPPAGE_BPS });
      } else if (!h.tp1Done && atLeast(value, h.cost, 160n, 100n)) {
        orders.push({ side: 'sell', mint, token_amount: pos.token_amount, max_slippage_bps: SELL_SLIPPAGE_BPS });
      } else if (!h.tp1Done && atLeast(value, h.cost, 120n, 100n)) {
        const half = pos.token_amount / 2n;
        if (half > 0n) {
          orders.push({ side: 'sell', mint, token_amount: half, max_slippage_bps: SELL_SLIPPAGE_BPS });
          h.tp1Done = true;
        }
      } else if (state.slot - h.boughtSlot >= MAX_HOLD_SLOTS && atMost(value, h.cost, 120n, 100n)) {
        orders.push({ side: 'sell', mint, token_amount: pos.token_amount, max_slippage_bps: SELL_SLIPPAGE_BPS });
      }

      if (value > h.peak) h.peak = value;
      h.lastVq = vq;
      h.lastVt = vt;
    }


    // Entries: mature curves with sustained, broad-based buying.
    for (const [mint, tape] of tapes) {
      if (tape.decided || tape.complete || tape.createdSlot === null) continue;
      const age = state.slot - tape.createdSlot;
      if (age < MIN_AGE_SLOTS || age > MAX_AGE_SLOTS) continue;
      if (tape.creatorSold) {
        tape.decided = true;
        continue;
      }
      const curve = state.curves.get(mint);
      if (curve === undefined || curve.complete) continue;
      if (curve.real_quote_reserves < MIN_REAL_LAMPORTS) continue;
      if (curve.real_quote_reserves > MAX_REAL_LAMPORTS) continue;
      const cap = state.maxBuyLamports(mint);
      if (cap < MIN_BUY_LAMPORTS) continue;
      const stats = windowStats(tape, state.slot);
      if (stats === null) continue;
      if (stats.netSol < MIN_NET_SOL_LAMPORTS) continue;
      if (stats.buyers < MIN_BUYERS) continue;
      if (!stats.maxBuyerShareOk) continue;

      const price = priceOf(state, mint);
      if (price === null) continue;
      const [vq, vt] = price;
      // Up 5-120% over the flow window, and within 10% of the window high.
      if (!atLeast(vq, vt, stats.startVq * 105n, stats.startVt * 100n)) continue;
      if (!atMost(vq, vt, stats.startVq * 220n, stats.startVt * 100n)) continue;
      if (!atLeast(vq, vt, stats.highVq * 90n, stats.highVt * 100n)) continue;

      if (state.committedPositionCount() >= MAX_MAIN_POSITIONS) continue;
      if (orders.length >= state.hello.caps.max_orders_per_batch) continue;
      const amount = cap < INITIAL_LAMPORTS ? cap : INITIAL_LAMPORTS;
      if (state.freeCashLamports() < amount + txFee) continue;
      orders.push({ side: 'buy', mint, quote_lamports: amount, max_slippage_bps: BUY_SLIPPAGE_BPS });
      tape.decided = true;
    }

    // Guardrail probes: stay on pace for TARGET_COINS round-tripped coins.
    if (
      probeCount < MAX_PROBES &&
      firstBlockTime !== null &&
      lastBlockTime !== null &&
      state.committedPositionCount() < state.hello.caps.max_open_positions &&
      orders.length < state.hello.caps.max_orders_per_batch
    ) {
      const elapsed = lastBlockTime - firstBlockTime;
      const expected = elapsed > 0 ? Number((BigInt(TARGET_COINS) * BigInt(elapsed)) / BigInt(PACE_SECONDS)) : 0;
      if (boughtCoins.size < expected) {
        let best: Pubkey | null = null;
        let bestReal = 0n;
        for (const [mint, curve] of state.curves) {
          if (curve.complete || curve.real_quote_reserves < PROBE_MIN_REAL_LAMPORTS) continue;
          if (held.has(mint) || state.positions.has(mint)) continue;
          const tape = tapes.get(mint);
          if (tape !== undefined && (tape.decided || tape.complete)) continue;
          if (state.maxBuyLamports(mint) < PROBE_LAMPORTS) continue;
          if (curve.real_quote_reserves > bestReal) {
            bestReal = curve.real_quote_reserves;
            best = mint;
          }
        }
        if (best !== null && state.freeCashLamports() >= PROBE_LAMPORTS + txFee) {
          orders.push({ side: 'buy', mint: best, quote_lamports: PROBE_LAMPORTS, max_slippage_bps: BUY_SLIPPAGE_BPS });
          probeMints.add(best);
          const tape = tapes.get(best);
          if (tape !== undefined) tape.decided = true;
          probeCount += 1;
        }
      }
    }

    return orders;
  },

  onDone(state: BotState, done: Done): void {
    console.error(
      `done ${done.reason} cash=${state.cash.toString()} ` +
        `bought=${boughtCoins.size} roundTripped=${roundTripped.size} probes=${probeCount}`,
    );
  },
});
