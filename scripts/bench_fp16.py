"""Benchmark FP16 mixed precision on STP, message passing, and learning.

Tests which operations benefit from FP16 storage and where precision dies.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import time
import math

BASELINE = math.log(2)
N = 50000
E = 4_700_000
DEVICE = 'cuda'
N_ITER = 500
WARMUP = 100


def bench(fn, label):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N_ITER * 1000
    return ms


def main():
    print("=" * 60)
    print("  FP16 MIXED PRECISION BENCHMARK")
    print("=" * 60)
    print(f"  N={N:,}, E={E:,}\n")

    # Simulate realistic data
    torch.manual_seed(42)
    output_f32 = torch.rand(N, device=DEVICE) * 0.3  # sparse-ish, most near 0
    output_f32[torch.rand(N, device=DEVICE) > 0.17] = 0.0  # 83% silent (true silence)
    output_f16 = output_f32.half()

    src = torch.randint(0, N, (E,), device=DEVICE, dtype=torch.int64)
    dst = torch.randint(0, N, (E,), device=DEVICE, dtype=torch.int64)

    # STP state
    fac_f32 = torch.rand(E, device=DEVICE) * 0.3
    dep_f32 = 0.7 + torch.rand(E, device=DEVICE) * 0.3
    rp_f32 = (fac_f32 + 0.1) * dep_f32
    fac_f16 = fac_f32.half()
    dep_f16 = dep_f32.half()
    rp_f16 = rp_f32.half()

    # Learning state
    weight = torch.rand(E, device=DEVICE) * 0.3 + 0.01
    pre_trace = torch.rand(E, device=DEVICE) * 0.1
    post_trace = torch.rand(E, device=DEVICE) * 0.5
    lr = torch.full((E,), 0.001, device=DEVICE)
    use_pt = torch.zeros(E, dtype=torch.bool, device=DEVICE)
    pred_err = torch.randn(N, device=DEVICE) * 0.1

    U, tau_f, tau_d = 0.1, 100.0, 200.0
    buf = torch.zeros(7, 8, N, device=DEVICE)
    delay_steps = torch.randint(1, 7, (E,), device=DEVICE, dtype=torch.int64)

    # ================================================================
    # 1. STP: FP32 vs FP16
    # ================================================================
    print("--- STP ---")

    def stp_f32():
        f, d, r = fac_f32, dep_f32, rp_f32
        pre = output_f32[src]
        du = -f / tau_f + U * (1.0 - f) * pre
        f.add_(du).clamp_(0.0, 1.0)
        dx = (1.0 - d) / tau_d - f * d * pre
        d.add_(dx).clamp_(0.0, 1.0)
        r.copy_((f + U) * d).clamp_(0.0, 1.0)

    def stp_f16():
        f, d, r = fac_f16, dep_f16, rp_f16
        pre = output_f16[src]
        du = -f / tau_f + U * (1.0 - f) * pre
        f.add_(du).clamp_(0.0, 1.0)
        dx = (1.0 - d) / tau_d - f * d * pre
        d.add_(dx).clamp_(0.0, 1.0)
        r.copy_((f + U) * d).clamp_(0.0, 1.0)

    def stp_f16_gather32():
        """FP16 STP state but gather from FP32 output (realistic: output stays FP32)."""
        f, d, r = fac_f16, dep_f16, rp_f16
        pre = output_f32[src].half()  # gather FP32, cast to FP16
        du = -f / tau_f + U * (1.0 - f) * pre
        f.add_(du).clamp_(0.0, 1.0)
        dx = (1.0 - d) / tau_d - f * d * pre
        d.add_(dx).clamp_(0.0, 1.0)
        r.copy_((f + U) * d).clamp_(0.0, 1.0)

    t32 = bench(stp_f32, "STP FP32")
    t16 = bench(stp_f16, "STP FP16")
    t16g = bench(stp_f16_gather32, "STP FP16+gather32")
    print(f"  FP32:          {t32:.3f} ms")
    print(f"  FP16:          {t16:.3f} ms  ({t32/t16:.2f}x)")
    print(f"  FP16+cast:     {t16g:.3f} ms  ({t32/t16g:.2f}x)")

    # Correctness: how much drift after 100 steps?
    fac_32 = torch.rand(E, device=DEVICE) * 0.3
    dep_32 = 0.7 + torch.rand(E, device=DEVICE) * 0.3
    rp_32 = (fac_32 + 0.1) * dep_32
    fac_16 = fac_32.half()
    dep_16 = dep_32.half()
    rp_16 = rp_32.half()
    for _ in range(100):
        pre32 = output_f32[src]
        pre16 = pre32.half()
        du32 = -fac_32 / tau_f + U * (1.0 - fac_32) * pre32
        fac_32.add_(du32).clamp_(0.0, 1.0)
        dx32 = (1.0 - dep_32) / tau_d - fac_32 * dep_32 * pre32
        dep_32.add_(dx32).clamp_(0.0, 1.0)
        rp_32.copy_((fac_32 + U) * dep_32).clamp_(0.0, 1.0)
        du16 = -fac_16 / tau_f + U * (1.0 - fac_16) * pre16
        fac_16.add_(du16).clamp_(0.0, 1.0)
        dx16 = (1.0 - dep_16) / tau_d - fac_16 * dep_16 * pre16
        dep_16.add_(dx16).clamp_(0.0, 1.0)
        rp_16.copy_((fac_16 + U) * dep_16).clamp_(0.0, 1.0)
    drift_fac = (fac_32 - fac_16.float()).abs().max().item()
    drift_dep = (dep_32 - dep_16.float()).abs().max().item()
    drift_rp = (rp_32 - rp_16.float()).abs().max().item()
    print(f"  Drift (100 steps): fac={drift_fac:.6f} dep={drift_dep:.6f} rp={drift_rp:.6f}")

    # ================================================================
    # 2. Message passing gather-multiply-scatter: FP32 vs FP16
    # ================================================================
    print("\n--- MESSAGE PASSING ---")

    buf_f32 = torch.zeros(7, 8, N, device=DEVICE)
    buf_f16 = torch.zeros(7, 8, N, device=DEVICE, dtype=torch.float16)
    channel = 0
    step = 100

    def mp_f32():
        msg = output_f32[src] * rp_f32 * weight
        flat_idx = (step + delay_steps) % 8 * N + dst
        buf_f32[channel].reshape(-1).index_add_(0, flat_idx, msg)

    def mp_f16_rp():
        """FP16 release_prob, FP32 output and weight."""
        msg = output_f32[src] * rp_f16.float() * weight  # upcast rp
        flat_idx = (step + delay_steps) % 8 * N + dst
        buf_f32[channel].reshape(-1).index_add_(0, flat_idx, msg)

    def mp_f16_full():
        """FP16 output + release_prob, FP32 weight, FP16 buffer."""
        msg = (output_f16[src] * rp_f16 * weight.half())
        flat_idx = (step + delay_steps) % 8 * N + dst
        buf_f16[channel].reshape(-1).index_add_(0, flat_idx, msg)

    t32 = bench(mp_f32, "MP FP32")
    trp = bench(mp_f16_rp, "MP FP16 rp")
    tf = bench(mp_f16_full, "MP FP16 full")
    print(f"  FP32:          {t32:.3f} ms")
    print(f"  FP16 rp only:  {trp:.3f} ms  ({t32/trp:.2f}x)")
    print(f"  FP16 full:     {tf:.3f} ms  ({t32/tf:.2f}x)")

    # ================================================================
    # 3. Learning: FP32 vs mixed (FP16 gathers, FP32 accumulation)
    # ================================================================
    print("\n--- LEARNING ---")

    def learn_f32():
        s = output_f32[src]
        d = output_f32[dst]
        pre_trace.lerp_(s, 0.05)
        post_trace.mul_(0.999).addcmul_(s, d, value=0.0001).clamp_(0.0, 1.0)
        pe = pred_err[dst].abs()
        eg = (pe / 0.1).clamp(0.0, 3.0)
        es = torch.sigmoid((eg - 2.0) * 2.0)
        post_trace.sub_(0.0001 * es * post_trace * post_trace).clamp_(0.0, 1.0)
        pl = eg * (1.0 - 0.9 * post_trace)
        hs = torch.where(use_pt, pre_trace, s)
        dw = lr * pl * (hs * d - 0.013 * weight - d * d * weight)
        weight.add_(dw).clamp_(0.0, 1.0)

    def learn_mixed():
        """FP16 gathers, FP32 accumulation."""
        s = output_f32[src].half()  # gather then cast
        d = output_f32[dst].half()
        # Convert to FP32 for the actual learning math
        sf = s.float()
        df = d.float()
        pre_trace.lerp_(sf, 0.05)
        post_trace.mul_(0.999).addcmul_(sf, df, value=0.0001).clamp_(0.0, 1.0)
        pe = pred_err[dst].abs()
        eg = (pe / 0.1).clamp(0.0, 3.0)
        es = torch.sigmoid((eg - 2.0) * 2.0)
        post_trace.sub_(0.0001 * es * post_trace * post_trace).clamp_(0.0, 1.0)
        pl = eg * (1.0 - 0.9 * post_trace)
        hs = torch.where(use_pt, pre_trace, sf)
        dw = lr * pl * (hs * df - 0.013 * weight - df * df * weight)
        weight.add_(dw).clamp_(0.0, 1.0)

    t32 = bench(learn_f32, "Learn FP32")
    tmix = bench(learn_mixed, "Learn mixed")
    print(f"  FP32:          {t32:.3f} ms")
    print(f"  Mixed:         {tmix:.3f} ms  ({t32/tmix:.2f}x)")

    # ================================================================
    # 4. Node update: FP32 vs FP16
    # ================================================================
    print("\n--- NODE UPDATE ---")

    exc_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    exc_mask[:40000] = True
    exc_f = exc_mask.float()
    inh_mask = ~exc_mask
    inh_f = inh_mask.float()
    sst_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    sst_mask[43500:47000] = True
    pv_f = torch.zeros(N, device=DEVICE)
    pv_f[40000:43500] = 1.0

    basal = torch.randn(N, device=DEVICE) * 0.1
    apical = torch.randn(N, device=DEVICE) * 0.05
    gain = torch.ones(N, device=DEVICE)
    noise = torch.randn(N, device=DEVICE) * 0.005
    inp_b = torch.randn(N, device=DEVICE) * 0.1
    inp_a = torch.randn(N, device=DEVICE) * 0.05
    inp_s = torch.rand(N, device=DEVICE) * 0.1
    inp_p = torch.rand(N, device=DEVICE) * 0.1
    inp_e = torch.randn(N, device=DEVICE) * 0.01
    inp_v = torch.rand(N, device=DEVICE) * 0.1

    basal_f16 = basal.half()
    apical_f16 = apical.half()

    def node_f32():
        b, a = basal, apical
        b.add_((-b / 10.0 + inp_b) * exc_f)
        sg = torch.sigmoid(inp_s * 5.0)
        a.add_((-a / 20.0 + inp_a * (1.0 - sg)) * exc_f)
        pe = b - a
        pv = (1.0 - inp_p).clamp(0.0, 1.0)
        raw = (F.softplus(pe.abs()) - BASELINE).clamp(0.0, 10.0)
        out = torch.where(exc_mask, raw * pv * gain, output_f32)
        ii = inp_b + inp_e * pv_f
        b.add_((-b / 10.0 + ii) * inh_f)
        ir = (F.softplus(b) - BASELINE).clamp(0.0, 10.0)
        io = ir * gain * inh_f
        ss = (1.0 - inp_s - inp_v).clamp(0.0, 1.0)
        io = torch.where(sst_mask, io * ss, io)
        out = torch.where(inh_mask, io, out)
        return (out + noise).clamp(0.0, 10.0)

    def node_f16():
        b, a = basal_f16, apical_f16
        b.add_(((-b / 10.0 + inp_b.half()) * exc_f.half()))
        sg = torch.sigmoid(inp_s.half() * 5.0)
        a.add_((-a / 20.0 + inp_a.half() * (1.0 - sg)) * exc_f.half())
        pe = b - a
        pv = (1.0 - inp_p.half()).clamp(0.0, 1.0)
        raw = (F.softplus(pe.abs().float()).half() - BASELINE).clamp(0.0, 10.0)
        out = torch.where(exc_mask, raw * pv * gain.half(), output_f16)
        ii = inp_b.half() + inp_e.half() * pv_f.half()
        b.add_((-b / 10.0 + ii) * inh_f.half())
        ir = (F.softplus(b.float()).half() - BASELINE).clamp(0.0, 10.0)
        io = ir * gain.half() * inh_f.half()
        ss = (1.0 - inp_s.half() - inp_v.half()).clamp(0.0, 1.0)
        io = torch.where(sst_mask, io * ss, io)
        out = torch.where(inh_mask, io, out)
        return (out + noise.half()).clamp(0.0, 10.0)

    t32 = bench(node_f32, "Node FP32")
    t16 = bench(node_f16, "Node FP16")
    print(f"  FP32:          {t32:.3f} ms")
    print(f"  FP16:          {t16:.3f} ms  ({t32/t16:.2f}x)")

    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
