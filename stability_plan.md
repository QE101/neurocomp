# Stability Plan: Implementing Biological Stability Mechanisms

## The Problem
Theta amplifies Hebbian learning quadratically (pre × post = activity²), causing a ~1500-cycle oscillation between good prediction (2.9x mismatch) and catastrophic interference (0.5x). 15+ damping attempts failed. The learning rule is too promiscuous — updates every edge every step.

## The Biological Solutions (5 mechanisms, ordered by expected impact)

### Mechanism 1: Timing-Selective Learning (Highest Priority)
**What:** Replace continuous Hebbian (pre × post every step) with timing-gated Hebbian. Only update edges where pre and post activity are temporally correlated within a narrow window (~5-10 steps).

**How:** Track per-edge "coincidence" — did the source fire BEFORE the destination within the last 5 steps? Only edges with recent coincidence get Hebbian updates. Others are untouched.

```python
# Per-edge coincidence detector
coincidence = (pre_trace > threshold) & (post_activity > threshold)
dw = hebbian * coincidence.float()  # only coincident edges update
```

This uses the existing `pre_trace` and `post_trace` in EdgeStore. When pre_trace is high (source was recently active) AND post is currently active = coincidence = update. Otherwise no update.

**Why it should work:** At any given step, only ~5-10% of edges will have coincident pre/post timing. The other 90% are frozen. The Hebbian rule can't cause interference on frozen edges. Pattern A's coincident edges are mostly different from pattern B's coincident edges because they fire at different times.

**Test:** Theta + timing-selective Hebbian on A-B, 5000 cycles. Compare oscillation range to undamped.

**Files:** Modify the `apply_hebbian` function in experiment script. ~10 lines changed.

### Mechanism 2: BCM Sliding Threshold (High Priority)
**What:** Each node adapts its own plasticity threshold based on recent activity. Active nodes raise their threshold — harder to potentiate. Quiet nodes lower their threshold — easier to potentiate.

**How:**
```python
# Per-node sliding threshold (already have activity_ema)
bcm_threshold = activity_ema ** 2  # quadratic — standard BCM
# LTP when post > threshold, LTD when post < threshold
bcm_sign = torch.sign(ns.output - bcm_threshold)
dw = hebbian * bcm_sign  # positive or negative depending on threshold
```

**Why it should work:** When a node is very active (during its pattern's presentation), its threshold rises. The next pattern's attempt to further potentiate those edges fails because the threshold is too high. Natural per-node overshoot prevention.

**Test:** Theta + BCM + standard Hebbian on A-B, 5000 cycles.

**Files:** Add `bcm_threshold` computation to node update. Modify Hebbian to use BCM sign. ~15 lines.

### Mechanism 3: Neuromodulatory Learning Gate (Medium Priority)
**What:** Learning only occurs when a global neuromodulatory signal is present. Between "learning windows," all edges are frozen regardless of activity.

**How:** Use theta phase as the learning gate. Only update weights during a specific theta phase (e.g., the rising phase, ~40 of 167 steps per theta cycle).

```python
theta_phase = theta.get_phase(step)
learning_window = (theta_phase > 0.5 * pi) & (theta_phase < 1.5 * pi)  # ~half the cycle
dw = raw_dw * learning_window  # zero update outside the window
```

**Why it should work:** Pattern A is presented for 50 steps, covering ~0.3 theta cycles. If the learning window is ~80 steps per cycle, pattern A only learns during the theta phases it overlaps with. Pattern B learns during different theta phases (offset by 50 steps = different phase). The two patterns' learning windows partially separate in time.

**Test:** Theta + phase-gated learning on A-B, 5000 cycles.

**Files:** Add phase check to Hebbian function. ~5 lines.

### Mechanism 4: Extreme Sparsity (Medium Priority)
**What:** Enforce 1-2% activation per pattern instead of 10%. Stronger PV competition drives winner-take-all so each pattern activates a tiny, distinct subset.

**How:** Boost PV inhibition until only 1-2% of excitatory nodes are active at any time. At N=1250 with k=30, this means ~12-25 active nodes per pattern. At N=50K, ~400-800 nodes.

**Why it should work:** 1% activation → pattern overlap near zero → no shared edges → no interference. This is how the brain does it.

**Challenge:** At N=1250, 1% = 10 nodes. Too few for meaningful computation. Need N=10K+ for 1% sparsity to work (100+ active nodes).

**Test:** N=10K + PV boost + theta on A-B, 5000 cycles. Measure activation sparsity AND oscillation.

**Files:** PV weight boost in experiment script. ~3 lines. But need N=10K graph (already have KNN builder).

### Mechanism 5: Sleep Consolidation (Lower Priority, Highest Impact Long-Term)
**What:** Alternate between "wake" (fast learning with input) and "sleep" (slow replay without input). During sleep, replay stored patterns at reduced rate to consolidate cortical weights. The cortex never faces rapid A-B alternation — it gets slow, interleaved replays.

**How:**
```python
# Every N wake cycles, run M sleep cycles
if cycle % wake_cycles == 0:
    for sleep_step in range(sleep_steps):
        # Replay: inject stored pattern at 0.5x strength
        replay_pattern = random.choice(stored_patterns)
        inject(replay_pattern, strength=0.5)
        # Learn at 0.1x rate
        apply_hebbian(graph, la, lr_scale=0.1)
```

**Why it should work:** The cortex learns slowly from curated replays during sleep, not from rapid alternating patterns during wake. Hippocampus handles the fast A-B learning (we showed it works). Cortex consolidates during sleep with reduced learning rate on interleaved patterns.

**Challenge:** Need hippocampal fast encoding + cortical slow consolidation + replay mechanism. Most complex to implement.

**Test:** Wake-sleep cycling on A-B, measure whether cortical weights stabilise after sleep phases.

**Files:** New sleep loop in experiment script. ~30 lines.

## Testing Plan

### Phase 1: Individual Mechanisms (1 day)
Run each mechanism individually with theta on A-B at N=1250, 5000 cycles.
All parallel (separate scripts, same GPU — they're fast individually).

| Test | Script | Expected Time |
|------|--------|---------------|
| Oja's rule (running) | run_oja_hebbian.py | ~2 hrs |
| Timing-selective | run_theta_timing_gate.py | ~2 hrs |
| BCM threshold | run_theta_bcm.py | ~2 hrs |
| Phase-gated learning | run_theta_phase_gate.py | ~2 hrs |

### Phase 2: Best Combinations (1 day)
Take the top 2-3 that show any damping. Combine them. Test at N=1250.

### Phase 3: Scale Test (1 day)
Winning combination at N=50K with theta. The definitive test.

### Phase 4: Multi-pattern Stress Test
Winning combination with 4 and 8 patterns. Does stability hold under complexity?

## Success Criteria

| Metric | Target |
|--------|--------|
| Mismatch (with theta) | > 1.3x sustained |
| Oscillation damping | Second-half range < 50% of first-half |
| Learning preserved | Suppression > 80% |
| No parameters to tune | Mechanism works across seeds and scales |

## Priority Order for Implementation

1. **Timing-selective** (biggest impact, addresses root cause: promiscuous learning)
2. **BCM threshold** (simple, per-node, prevents overshoot directly)
3. **Phase-gated** (leverages theta structure, 5 lines of code)
4. **Extreme sparsity** (needs larger N, addresses representation)
5. **Sleep consolidation** (most complex, most powerful long-term)

Implement 1-3 today, test overnight. Results determine whether we need 4-5.
