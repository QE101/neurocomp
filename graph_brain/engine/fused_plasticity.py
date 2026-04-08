"""Fused STP + learning: all edge types in one vectorized pass.

Concatenates per-edge tensors into flat buffers, replaces EdgeStore fields
with views for zero-copy message passing access.

With torch.compile (Linux/WSL): 3.5x total speedup (9.68 ms -> 2.78 ms).
Without (Windows eager):        1.9x total speedup (9.68 ms -> 5.02 ms).

Usage:
    fused = FusedPlasticity(graph, stp_config)
    fused.enable_compile()  # optional, requires Linux/WSL
    fused.stp(node_state.output)
    fused.learn(node_state, exc_mask, current_decay)
    fused.rebuild(graph)    # after structural plasticity
"""

from __future__ import annotations

import sys
import torch
from torch import Tensor
import torch.nn.functional as F

from graph_brain.config import STPConfig
from graph_brain.core.graph import NeuromorphicGraph, EdgeStore
from graph_brain.types import EdgeType


PLASTIC_EDGE_TYPES = [
    EdgeType.DRIVING, EdgeType.MODULATORY, EdgeType.INHIB_PERISOMATIC,
    EdgeType.INHIB_DENDRITIC, EdgeType.DISINHIBITION, EdgeType.RETROGRADE,
]

_LR_MAP = {
    EdgeType.DRIVING: 0.0001,
    EdgeType.MODULATORY: 0.001,
    EdgeType.DISINHIBITION: 0.002,
    EdgeType.INHIB_PERISOMATIC: 0.0001,
    EdgeType.INHIB_DENDRITIC: 0.0001,
    EdgeType.RETROGRADE: 0.001,
}

WEIGHT_DECAY = 0.013
BASELINE = 0.6931471805599453  # ln(2)


# ================================================================
# Compilable standalone functions (no self, no control flow)
# ================================================================

def _stp_core(f_fac, f_dep, f_rp, pre_act, U, tau_f, tau_d):
    """STP math — in-place, compilable."""
    du = -f_fac / tau_f + U * (1.0 - f_fac) * pre_act
    f_fac.add_(du).clamp_(0.0, 1.0)
    dx = (1.0 - f_dep) / tau_d - f_fac * f_dep * pre_act
    f_dep.add_(dx).clamp_(0.0, 1.0)
    f_rp.copy_((f_fac + U) * f_dep).clamp_(0.0, 1.0)


def _learn_core(f_pre, f_post, f_weight, f_lr, f_use_pt,
                src, dst, pred_err_dst, global_nov, decay):
    """Learning math — in-place, compilable."""
    f_pre.lerp_(src, 0.05)
    f_post.mul_(decay).addcmul_(src, dst, value=0.0001).clamp_(0.0, 1.0)
    error_gate = (pred_err_dst.abs() / global_nov).clamp(0.0, 3.0)
    error_signal = torch.sigmoid((error_gate - 2.0) * 2.0)
    f_post.sub_(0.0001 * error_signal * f_post * f_post).clamp_(0.0, 1.0)
    plasticity = error_gate * (1.0 - 0.9 * f_post)
    hebbian_src = torch.where(f_use_pt, f_pre, src)
    dw = f_lr * plasticity * (hebbian_src * dst - WEIGHT_DECAY * f_weight - dst * dst * f_weight)
    f_weight.add_(dw).clamp_(0.0, 1.0)


def _node_core(basal, apical, output, pred_error, gain, activity_ema,
               inp_basal, inp_apical, inp_sst, inp_pv, inp_elec, inp_vip,
               exc_mask, exc_f, inh_mask, inh_f, sst_mask, pv_f,
               basal_tau, apical_tau, input_norm, theta_mod, noise):
    """Full node update — compilable."""
    # Excitatory two-compartment
    basal = basal + (-basal / basal_tau + inp_basal * theta_mod * input_norm) * exc_f
    sst_gate = torch.sigmoid(inp_sst * 5.0)
    apical = apical + (-apical / apical_tau + inp_apical * (1.0 - sst_gate) * input_norm) * exc_f
    pred_err = basal - apical
    pv_gain = (1.0 - inp_pv).clamp(0.0, 1.0)
    raw = (F.softplus(pred_err.abs()) - BASELINE).clamp(0.0, 10.0)
    output = torch.where(exc_mask, raw * pv_gain * gain, output)
    pred_error = torch.where(exc_mask, pred_err, pred_error)
    # Inhibitory (vectorized — PV/SST/VIP non-overlapping)
    inh_input = inp_basal + inp_elec * pv_f
    basal = basal + (-basal / basal_tau + inh_input) * inh_f
    inh_raw = (F.softplus(basal) - BASELINE).clamp(0.0, 10.0)
    inh_out = inh_raw * gain * inh_f
    sst_suppress = (1.0 - inp_sst - inp_vip).clamp(0.0, 1.0)
    inh_out = torch.where(sst_mask, inh_out * sst_suppress, inh_out)
    output = torch.where(inh_mask, inh_out, output)
    # Noise + clamp + EMA
    output = (output + noise).clamp(0.0, 10.0)
    activity_ema = activity_ema + (1.0 / 1000.0) * (output - activity_ema)
    return basal, apical, output, pred_error, activity_ema


# ================================================================
# Main class
# ================================================================

class FusedPlasticity:
    """Fused STP + learning across all edge types.

    Concatenates per-edge tensors into flat buffers and replaces EdgeStore
    fields with views, giving zero-copy access from message passing.
    """

    def __init__(self, graph: NeuromorphicGraph, stp_cfg: STPConfig, delay_buffer=None,
                 fp16_stp: bool = True):
        self.stp_cfg = stp_cfg
        self.device = graph.device
        self.U = stp_cfg.U_baseline
        self.tau_f = stp_cfg.tau_facilitation
        self.tau_d = stp_cfg.tau_depression
        self._compiled = False
        self._fp16_stp = fp16_stp
        self._build(graph)

    def _build(self, graph: NeuromorphicGraph):
        """Build fused buffers and replace EdgeStore fields with views."""
        device = self.device

        self.offsets: dict[EdgeType, tuple[int, int]] = {}
        self.active_types: list[EdgeType] = []
        offset = 0
        for et in PLASTIC_EDGE_TYPES:
            if not graph.has_edge_type(et):
                continue
            store = graph.edge_store(et)
            if store.n_edges == 0:
                continue
            self.active_types.append(et)
            self.offsets[et] = (offset, offset + store.n_edges)
            offset += store.n_edges
        self.n_total = offset

        if self.n_total == 0:
            return

        stp_dtype = torch.float16 if self._fp16_stp else torch.float32
        self.f_facilitation = torch.zeros(self.n_total, device=device, dtype=stp_dtype)
        self.f_depression = torch.ones(self.n_total, device=device, dtype=stp_dtype)
        self.f_release_prob = torch.zeros(self.n_total, device=device, dtype=stp_dtype)
        self.f_weight = torch.zeros(self.n_total, device=device)
        self.f_pre_trace = torch.zeros(self.n_total, device=device)
        self.f_post_trace = torch.zeros(self.n_total, device=device)

        src_parts = []
        dst_parts = []

        for et in self.active_types:
            store = graph.edge_store(et)
            s, e = self.offsets[et]
            self.f_facilitation[s:e].copy_(store.facilitation)
            self.f_depression[s:e].copy_(store.depression)
            self.f_release_prob[s:e].copy_(store.release_prob)
            self.f_weight[s:e].copy_(store.weight)
            self.f_pre_trace[s:e].copy_(store.pre_trace)
            self.f_post_trace[s:e].copy_(store.post_trace)
            src_parts.append(store.src.long())
            dst_parts.append(store.dst.long())
            store.facilitation = self.f_facilitation[s:e]
            store.depression = self.f_depression[s:e]
            store.release_prob = self.f_release_prob[s:e]
            store.weight = self.f_weight[s:e]
            store.pre_trace = self.f_pre_trace[s:e]
            store.post_trace = self.f_post_trace[s:e]

        self.f_src64 = torch.cat(src_parts)
        self.f_dst64 = torch.cat(dst_parts)

        self.f_lr = torch.zeros(self.n_total, device=device)
        for et in self.active_types:
            s, e = self.offsets[et]
            self.f_lr[s:e] = _LR_MAP.get(et, 0.001)

        self.f_use_pretrace = torch.zeros(self.n_total, dtype=torch.bool, device=device)
        if EdgeType.DRIVING in self.offsets:
            s, e = self.offsets[EdgeType.DRIVING]
            self.f_use_pretrace[s:e] = True

        self.f_src_out = torch.zeros(self.n_total, device=device)
        self.f_dst_out = torch.zeros(self.n_total, device=device)

    def rebuild(self, graph: NeuromorphicGraph, delay_buffer=None):
        """Rebuild after structural plasticity changes edge counts."""
        was_compiled = self._compiled
        self._build(graph)
        if was_compiled:
            self.enable_compile()

    def enable_compile(self):
        """Compile STP + learn cores via torch.compile. ~1.8x on top of fusion.

        Requires Linux (WSL). No-op if torch.compile unavailable.
        """
        if sys.platform == 'win32':
            print('  [compile] skipped — Windows (run from WSL for 3.5x speedup)', flush=True)
            return
        try:
            self._stp_fn = torch.compile(_stp_core, mode='default')
            self._learn_fn = torch.compile(_learn_core, mode='default')
            # Warmup both (triggers JIT compilation)
            if self.n_total > 0:
                stp_dtype = torch.float16 if self._fp16_stp else torch.float32
                pre_act = torch.zeros(self.n_total, device=self.device, dtype=stp_dtype)
                self._stp_fn(self.f_facilitation, self.f_depression, self.f_release_prob,
                             pre_act, self.U, self.tau_f, self.tau_d)
                # Learn warmup with matching shapes
                dummy_src = torch.zeros(self.n_total, device=self.device)
                dummy_dst = torch.zeros(self.n_total, device=self.device)
                dummy_err = torch.zeros(self.n_total, device=self.device)
                dummy_nov = torch.tensor(1.0, device=self.device)
                self._learn_fn(self.f_pre_trace, self.f_post_trace, self.f_weight,
                               self.f_lr, self.f_use_pretrace,
                               dummy_src, dummy_dst, dummy_err, dummy_nov, 0.999)
                torch.cuda.synchronize()
            self._compiled = True
            print(f'  [compile] STP + learn compiled (mode=default)', flush=True)
        except Exception as e:
            print(f'  [compile] failed: {e}', flush=True)
            self._compiled = False

    def stp(self, output: Tensor, dt: float = 1.0):
        """Fused STP for all edge types."""
        if self.n_total == 0:
            return

        pre_activity = output[self.f_src64]
        if self._fp16_stp and pre_activity.dtype != torch.float16:
            pre_activity = pre_activity.half()

        if self._compiled:
            self._stp_fn(self.f_facilitation, self.f_depression, self.f_release_prob,
                         pre_activity, self.U, self.tau_f, self.tau_d)
        else:
            du = dt * (-self.f_facilitation / self.tau_f + self.U * (1.0 - self.f_facilitation) * pre_activity)
            self.f_facilitation.add_(du).clamp_(0.0, 1.0)
            dx = dt * ((1.0 - self.f_depression) / self.tau_d - self.f_facilitation * self.f_depression * pre_activity)
            self.f_depression.add_(dx).clamp_(0.0, 1.0)
            torch.mul(self.f_facilitation + self.U, self.f_depression, out=self.f_release_prob)
            self.f_release_prob.clamp_(0.0, 1.0)

    def learn(self, node_state, exc_mask: Tensor, current_decay: float,
              is_replay: bool = False, lr_scale: float = 1.0):
        """Fused learning with adaptive consolidation."""
        if self.n_total == 0:
            return

        pred_err = node_state.prediction_error
        output = node_state.output
        global_novelty = pred_err[exc_mask].abs().mean().clamp(min=0.01)

        torch.index_select(output, 0, self.f_src64, out=self.f_src_out)
        torch.index_select(output, 0, self.f_dst64, out=self.f_dst_out)
        src = self.f_src_out
        dst = self.f_dst_out

        pred_err_dst = pred_err[self.f_dst64]

        if self._compiled and not is_replay:
            self._learn_fn(self.f_pre_trace, self.f_post_trace, self.f_weight,
                           self.f_lr * lr_scale if lr_scale != 1.0 else self.f_lr,
                           self.f_use_pretrace, src, dst, pred_err_dst,
                           global_novelty, current_decay)
        else:
            # Eager path (also handles replay edge case)
            self.f_pre_trace.lerp_(src, 0.05)
            self.f_post_trace.mul_(current_decay)
            self.f_post_trace.addcmul_(src, dst, value=0.0001)
            self.f_post_trace.clamp_(0.0, 1.0)

            dst_error = pred_err_dst.abs()
            error_gate = (dst_error / global_novelty).clamp_(0.0, 3.0)
            error_signal = torch.sigmoid((error_gate - 2.0) * 2.0)
            unconsolidate = error_signal.mul_(0.0001).mul_(self.f_post_trace).mul_(self.f_post_trace)
            self.f_post_trace.sub_(unconsolidate).clamp_(0.0, 1.0)

            plasticity = error_gate * (1.0 - 0.9 * self.f_post_trace)
            hebbian_src = torch.where(self.f_use_pretrace, self.f_pre_trace, src)

            if is_replay and EdgeType.DRIVING in self.offsets:
                dw = self.f_lr * lr_scale * plasticity * (
                    hebbian_src * dst - WEIGHT_DECAY * self.f_weight - dst * dst * self.f_weight)
                s, e = self.offsets[EdgeType.DRIVING]
                self.f_post_trace[s:e].addcmul_(src[s:e], dst[s:e], value=0.01)
                self.f_post_trace[s:e].clamp_(0.0, 1.0)
                dw[s:e] = self.f_lr[s:e] * lr_scale * (
                    self.f_pre_trace[s:e] * dst[s:e]
                    - WEIGHT_DECAY * self.f_weight[s:e]
                    - dst[s:e] * dst[s:e] * self.f_weight[s:e])
            else:
                dw = self.f_lr * lr_scale * plasticity * (
                    hebbian_src * dst - WEIGHT_DECAY * self.f_weight - dst * dst * self.f_weight)

            self.f_weight.add_(dw).clamp_(0.0, 1.0)
