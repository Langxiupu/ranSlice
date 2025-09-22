"""End-to-end simulator for the VR/FileSync slicing scenarios."""
from __future__ import annotations

import importlib.util
import math
import random
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
if HAS_MATPLOTLIB:
    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.ticker import MaxNLocator  # type: ignore
else:
    matplotlib = None  # type: ignore
    plt = None  # type: ignore
    MaxNLocator = None  # type: ignore

# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------
SEED = 2025
DT_MS = 1  # TTI granularity = 1 ms
T_SIM = 40_000  # 40 seconds
WIN_TP = 100  # Throughput window (ms)
WIN_P99 = 100  # Delay P99 window length
THROUGHPUT_THRESHOLD_MBPS = 8.0
DELAY_THRESHOLD_MS = 60.0

RB_CAP = 36  # Available RB per TTI before jitter
B_R_MIN = 20e6  # Slice B minimum rate (bit/s)
A_R_MIN = 24e6  # Slice A minimum rate (bit/s)
PHI_REF_A = 800.0  # Reference spectral efficiency (bit/RB)
PHI_REF_B = 800.0
BETA = 0.05
EPS = 1e-9

# VR traffic
VR_FPS = 60
VR_FRAME_MS = round(1000 / VR_FPS)  # ~17 ms
VR_RATE_PER_APP = 12e6
VR_JITTER = 0.15
VR_DEADLINE = int(DELAY_THRESHOLD_MS)  # ms
# Deterministic spectral efficiency multipliers for each VR app (1.0 = reference).
# The defaults represent a mixed deployment with one high-quality and one
# low-quality channel user. Specific scenarios can override this list to model
# different user mixes.
VR_CHANNEL_QUALITIES = [1.1, 0.5]

# File sync traffic
FS_MEAN_PKT_BITS = 12_000
FS_POISSON_LAMBDA = 800  # packets per second
FS_DEADLINE = 20  # ms


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class Packet:
    __slots__ = ("bits_left", "arrival", "deadline", "depart")

    def __init__(self, size_bits: int, arrival_ts: int, deadline_ms: int) -> None:
        self.bits_left = size_bits
        self.arrival = arrival_ts
        self.deadline = arrival_ts + deadline_ms
        self.depart: Optional[int] = None


class Flow:
    __slots__ = (
        "name",
        "kind",
        "q",
        "p99_window",
        "total_sent_bits",
        "total_generated_bits",
        "generated_packets",
        "completed_packets",
        "channel_quality",
    )

    def __init__(self, name: str, kind: str, channel_quality: float = 1.0) -> None:
        self.name = name
        self.kind = kind
        self.q: Deque[Packet] = deque()
        self.p99_window: Deque[int] = deque()
        self.total_sent_bits = 0
        self.total_generated_bits = 0
        self.generated_packets = 0
        self.completed_packets = 0
        self.channel_quality = channel_quality

    def has_backlog(self) -> bool:
        return bool(self.q)


class Slice:
    __slots__ = (
        "sid",
        "R_min",
        "phi_ref",
        "r_bar",
        "x_bar",
        "flows",
        "phi_inst",
        "rr_index",
        "total_sent_bits",
    )

    def __init__(self, sid: str, r_min: float, phi_ref: float, flows: Iterable[Flow]) -> None:
        self.sid = sid
        self.R_min = r_min
        self.phi_ref = phi_ref
        self.r_bar = 1.0
        self.x_bar = 0.0
        self.flows: List[Flow] = list(flows)
        self.phi_inst = phi_ref
        self.rr_index = 0
        self.total_sent_bits = 0

    def has_backlog(self) -> bool:
        return any(flow.has_backlog() for flow in self.flows)


# ---------------------------------------------------------------------------
# Channel model
# ---------------------------------------------------------------------------
def update_phi(phi_prev: float, drift: float = 0.02, lo: float = 600.0, hi: float = 1000.0) -> float:
    phi = phi_prev * (1.0 + random.uniform(-drift, drift))
    return max(lo, min(hi, phi))


# ---------------------------------------------------------------------------
# Traffic models
# ---------------------------------------------------------------------------
def gen_vr_frame_bits(rate: float = VR_RATE_PER_APP, jitter: float = VR_JITTER) -> int:
    base = rate * (VR_FRAME_MS / 1000.0)
    return max(1, int(base * (1.0 + random.uniform(-jitter, jitter))))


def push_vr_packets(flow: Flow, now_ms: int) -> None:
    if now_ms % VR_FRAME_MS == 0:
        size_bits = gen_vr_frame_bits()
        flow.q.append(Packet(size_bits, now_ms, VR_DEADLINE))
        flow.total_generated_bits += size_bits
        flow.generated_packets += 1


def sample_poisson(lam: float) -> int:
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1


def push_filesync_packets(flow: Flow, now_ms: int) -> None:
    lam_ms = FS_POISSON_LAMBDA / 1000.0
    arrivals = sample_poisson(lam_ms)
    for _ in range(arrivals):
        size_bits = max(4_000, int(random.expovariate(1.0 / max(FS_MEAN_PKT_BITS, EPS))))
        flow.q.append(Packet(size_bits, now_ms, FS_DEADLINE))
        flow.total_generated_bits += size_bits
        flow.generated_packets += 1


# ---------------------------------------------------------------------------
# Scheduler utilities
# ---------------------------------------------------------------------------
def eff_min_rate(r_min: float, phi: float, phi_ref: float) -> float:
    if phi_ref <= EPS:
        return r_min
    if phi >= phi_ref:
        return r_min
    return r_min * phi / max(phi_ref, EPS)


def slice_weight_bandwidth(slice_obj: Slice) -> float:
    r_eff = eff_min_rate(slice_obj.R_min, slice_obj.phi_inst, slice_obj.phi_ref)
    c_i = r_eff / max(slice_obj.phi_ref, EPS)
    return c_i * slice_obj.phi_inst / max(slice_obj.r_bar, EPS)


def ewma_update(slice_obj: Slice, r_inst_bits_per_s: float, selected: bool) -> None:
    slice_obj.r_bar = (1.0 - BETA) * slice_obj.r_bar + BETA * (r_inst_bits_per_s if selected else 0.0)
    slice_obj.x_bar = (1.0 - BETA) * slice_obj.x_bar + BETA * (1.0 if selected else 0.0)


def schedule_weighted_fair(
    slice_obj: Slice,
    rb_budget: int,
    now_ms: int,
    metrics: Optional["Metrics"] = None,
) -> int:
    if not slice_obj.flows or rb_budget <= 0:
        return 0

    bits_sent = 0
    n = len(slice_obj.flows)
    idx = slice_obj.rr_index % n

    while rb_budget > 0:
        served = False
        for _ in range(n):
            flow = slice_obj.flows[idx]
            idx = (idx + 1) % n
            if not flow.q:
                continue
            pkt = flow.q[0]
            quality = max(flow.channel_quality, 0.01)
            bits_per_rb = max(1, int(round(slice_obj.phi_inst * quality)))
            tx_bits = min(bits_per_rb, pkt.bits_left)
            pkt.bits_left -= tx_bits
            flow.total_sent_bits += tx_bits
            bits_sent += tx_bits
            rb_budget -= 1
            served = True
            if metrics is not None:
                metrics.on_bits_sent(flow.name, tx_bits)
            if pkt.bits_left == 0:
                pkt.depart = now_ms
                flow.q.popleft()
                flow.completed_packets += 1
                delay = pkt.depart - pkt.arrival
                flow.p99_window.append(delay)
                if len(flow.p99_window) > WIN_P99:
                    flow.p99_window.popleft()
            break
        if not served:
            break

    slice_obj.rr_index = idx
    slice_obj.total_sent_bits += bits_sent
    return bits_sent


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------
class Metrics:
    def __init__(self, flows: Iterable[Flow], delay_mode: str = "head") -> None:
        self.delay_mode = delay_mode
        self.flow_names = [f.name for f in flows]
        self.tp_win_bits: Dict[str, int] = {name: 0 for name in self.flow_names}
        self.tp_series: Dict[str, List[float]] = {name: [] for name in self.flow_names}
        self.delay_series: Dict[str, List[float]] = {name: [] for name in self.flow_names}
        self.tp_time_axis: List[float] = []
        self.delay_time_axis: List[float] = []

    def on_bits_sent(self, flow_name: str, bits: int) -> None:
        if flow_name in self.tp_win_bits:
            self.tp_win_bits[flow_name] += bits

    def on_tti_end(self, flows: Iterable[Flow], now_ms: int) -> None:
        self.delay_time_axis.append(now_ms / 1000.0)
        for flow in flows:
            if flow.name not in self.delay_series:
                continue
            if self.delay_mode == "p99" and flow.p99_window:
                window = sorted(flow.p99_window)
                rank = max(0, math.ceil(0.99 * len(window)) - 1)
                delay = window[min(rank, len(window) - 1)]
            else:
                delay = now_ms - flow.q[0].arrival if flow.q else 0.0
            self.delay_series[flow.name].append(delay)

        if (now_ms + 1) % WIN_TP == 0:
            self.tp_time_axis.append((now_ms + 1) / 1000.0)
            window_seconds = WIN_TP / 1000.0
            for name in self.flow_names:
                bits = self.tp_win_bits[name]
                mbps = bits / window_seconds / 1e6
                self.tp_series[name].append(mbps)
                self.tp_win_bits[name] = 0


def render_line_chart(
    time_axis: Sequence[float],
    series: Dict[str, Sequence[float]],
    outfile: str,
    title: str,
    y_label: str,
    reference_line: Optional[float] = None,
    reference_label: str = "Deadline",
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    plotted_any = False
    for name, values in series.items():
        if not values:
            continue
        limit = min(len(time_axis), len(values))
        if limit == 0:
            continue
        ax.plot(time_axis[:limit], values[:limit], label=name)
        plotted_any = True

    if reference_line is not None:
        ax.axhline(
            reference_line,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=reference_label,
        )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    if MaxNLocator is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

    if plotted_any or reference_line is not None:
        ax.legend(loc="upper right")
    
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def plot_throughput(metrics: Metrics, outfile: str, scenario_label: str) -> None:
    series = {name.upper(): metrics.tp_series[name] for name in metrics.flow_names}
    render_line_chart(
        metrics.tp_time_axis,
        series,
        outfile,
        title=f"{scenario_label} THROUGHPUT",
        y_label="THROUGHPUT MBPS",
        reference_line=THROUGHPUT_THRESHOLD_MBPS,
        reference_label="Threshold",
    )


def plot_delay(
    metrics: Metrics,
    outfile: str,
    scenario_label: str,
    delay_mode: str,
    deadline: Optional[float] = None,
) -> None:
    series = {name.upper(): metrics.delay_series[name] for name in metrics.flow_names}
    ylabel = "DELAY MS" if delay_mode == "head" else "P99 DELAY MS"
    render_line_chart(
        metrics.delay_time_axis,
        series,
        outfile,
        title=f"{scenario_label} DELAY",
        y_label=ylabel,
        reference_line=deadline,
        reference_label="Threshold",
    )


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------
def run_scenario(
    num_vr_apps: int,
    save_prefix: str,
    delay_mode: str = "head",
    channel_qualities: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    random.seed(SEED)

    qualities = channel_qualities if channel_qualities is not None else VR_CHANNEL_QUALITIES

    slice_a_flows = []
    for i in range(num_vr_apps):
        quality_idx = min(i, len(qualities) - 1)
        channel_quality = qualities[quality_idx]
        slice_a_flows.append(Flow(f"app{i + 1}", "vr", channel_quality=channel_quality))
    slice_b_flow = Flow("filesync", "fs")

    slice_a = Slice("A", A_R_MIN, PHI_REF_A, slice_a_flows)
    slice_b = Slice("B", B_R_MIN, PHI_REF_B, [slice_b_flow])

    metrics = Metrics(slice_a_flows, delay_mode=delay_mode)

    duration_s = T_SIM / 1000.0

    for now_ms in range(T_SIM):
        push_filesync_packets(slice_b_flow, now_ms)
        for flow in slice_a_flows:
            push_vr_packets(flow, now_ms)

        slice_a.phi_inst = update_phi(slice_a.phi_inst)
        slice_b.phi_inst = update_phi(slice_b.phi_inst)

        rb_budget = max(1, int(round(RB_CAP * random.uniform(0.9, 1.1))))

        weights = [
            (slice_weight_bandwidth(slice_a), slice_a),
            (slice_weight_bandwidth(slice_b), slice_b),
        ]
        weights.sort(key=lambda item: item[0], reverse=True)

        selected_slice: Optional[Slice] = None
        for _, candidate in weights:
            if candidate.has_backlog():
                selected_slice = candidate
                break

        sent_bits = 0
        if selected_slice is not None:
            sent_bits = schedule_weighted_fair(
                selected_slice,
                rb_budget=rb_budget,
                now_ms=now_ms,
                metrics=metrics if selected_slice is slice_a else None,
            )

        r_inst = sent_bits * 1000.0  # convert to bit/s for EWMA update
        ewma_update(slice_a, r_inst if selected_slice is slice_a else 0.0, selected_slice is slice_a)
        ewma_update(slice_b, r_inst if selected_slice is slice_b else 0.0, selected_slice is slice_b)

        metrics.on_tti_end(slice_a_flows, now_ms)

    scenario_label = save_prefix.upper().replace("SCENARIO", "S")
    plot_throughput(metrics, f"{save_prefix}_throughput.png", scenario_label)
    plot_delay(
        metrics,
        f"{save_prefix}_delay.png",
        scenario_label,
        delay_mode=delay_mode,
        deadline=DELAY_THRESHOLD_MS if delay_mode == "head" else None,
    )

    stats = {}
    for flow in slice_a_flows:
        stats[flow.name] = flow.total_sent_bits / duration_s / 1e6
    return stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    scenario2_qualities = [1.1, 0.5]
    scenario4_qualities = [1.1, 1.1, 0.5, 0.5]

    scenario2_stats = run_scenario(
        num_vr_apps=2,
        save_prefix="scenario2",
        delay_mode="head",
        channel_qualities=scenario2_qualities,
    )
    scenario4_stats = run_scenario(
        num_vr_apps=4,
        save_prefix="scenario4",
        delay_mode="head",
        channel_qualities=scenario4_qualities,
    )

    target_mbps = THROUGHPUT_THRESHOLD_MBPS

    print(
        f"Scenario S2 average throughput (Mbps) vs threshold {target_mbps:.1f}:"
    )
    for idx, (name, value) in enumerate(scenario2_stats.items()):
        quality = scenario2_qualities[min(idx, len(scenario2_qualities) - 1)]
        meets_qos = value >= target_mbps
        status = "OK" if meets_qos else "VIOLATION"
        print(f"  {name} (quality={quality:.2f}): {value:.2f} -> {status}")

    print(
        f"\nScenario S4 average throughput (Mbps) vs threshold {target_mbps:.1f}:"
    )
    for idx, (name, value) in enumerate(scenario4_stats.items()):
        quality = scenario4_qualities[min(idx, len(scenario4_qualities) - 1)]
        meets_qos = value >= target_mbps
        status = "OK" if meets_qos else "VIOLATION"
        print(f"  {name} (quality={quality:.2f}): {value:.2f} -> {status}")


if __name__ == "__main__":
    main()