"""End-to-end simulator for the VR/FileSync slicing scenarios."""
from __future__ import annotations

import math
import random
import struct
import zlib
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------
SEED = 2025
DT_MS = 1  # TTI granularity = 1 ms
T_SIM = 40_000  # 40 seconds
WIN_TP = 100  # Throughput window (ms)
WIN_P99 = 100  # Delay P99 window length
RB_CAP = 260  # Available RB per TTI before jitter
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
VR_DEADLINE = 20  # ms

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
    )

    def __init__(self, name: str, kind: str) -> None:
        self.name = name
        self.kind = kind
        self.q: Deque[Packet] = deque()
        self.p99_window: Deque[int] = deque()
        self.total_sent_bits = 0
        self.total_generated_bits = 0
        self.generated_packets = 0
        self.completed_packets = 0

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
    bits_per_rb = max(1, int(round(slice_obj.phi_inst)))
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


FONT_5X7: Dict[str, Sequence[str]] = {
    " ": [
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
    ],
    "0": [
        "01110",
        "10001",
        "10011",
        "10101",
        "11001",
        "10001",
        "01110",
    ],
    "1": [
        "00100",
        "01100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ],
    "2": [
        "01110",
        "10001",
        "00001",
        "00110",
        "01000",
        "10000",
        "11111",
    ],
    "3": [
        "11110",
        "00001",
        "00001",
        "01110",
        "00001",
        "00001",
        "11110",
    ],
    "4": [
        "00010",
        "00110",
        "01010",
        "10010",
        "11111",
        "00010",
        "00010",
    ],
    "5": [
        "11111",
        "10000",
        "11110",
        "00001",
        "00001",
        "10001",
        "01110",
    ],
    "6": [
        "00110",
        "01000",
        "10000",
        "11110",
        "10001",
        "10001",
        "01110",
    ],
    "7": [
        "11111",
        "00001",
        "00010",
        "00100",
        "01000",
        "01000",
        "01000",
    ],
    "8": [
        "01110",
        "10001",
        "10001",
        "01110",
        "10001",
        "10001",
        "01110",
    ],
    "9": [
        "01110",
        "10001",
        "10001",
        "01111",
        "00001",
        "00010",
        "01100",
    ],
    "A": [
        "01110",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ],
    "B": [
        "11110",
        "10001",
        "10001",
        "11110",
        "10001",
        "10001",
        "11110",
    ],
    "C": [
        "01110",
        "10001",
        "10000",
        "10000",
        "10000",
        "10001",
        "01110",
    ],
    "D": [
        "11100",
        "10010",
        "10001",
        "10001",
        "10001",
        "10010",
        "11100",
    ],
    "E": [
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "11111",
    ],
    "F": [
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "10000",
    ],
    "G": [
        "01110",
        "10001",
        "10000",
        "10111",
        "10001",
        "10001",
        "01111",
    ],
    "H": [
        "10001",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ],
    "I": [
        "01110",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ],
    "J": [
        "00111",
        "00010",
        "00010",
        "00010",
        "00010",
        "10010",
        "01100",
    ],
    "K": [
        "10001",
        "10010",
        "10100",
        "11000",
        "10100",
        "10010",
        "10001",
    ],
    "L": [
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "11111",
    ],
    "M": [
        "10001",
        "11011",
        "10101",
        "10101",
        "10001",
        "10001",
        "10001",
    ],
    "N": [
        "10001",
        "11001",
        "10101",
        "10011",
        "10001",
        "10001",
        "10001",
    ],
    "O": [
        "01110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "01110",
    ],
    "P": [
        "11110",
        "10001",
        "10001",
        "11110",
        "10000",
        "10000",
        "10000",
    ],
    "Q": [
        "01110",
        "10001",
        "10001",
        "10001",
        "10101",
        "10010",
        "01101",
    ],
    "R": [
        "11110",
        "10001",
        "10001",
        "11110",
        "10100",
        "10010",
        "10001",
    ],
    "S": [
        "01111",
        "10000",
        "10000",
        "01110",
        "00001",
        "00001",
        "11110",
    ],
    "T": [
        "11111",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
    ],
    "U": [
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "01110",
    ],
    "V": [
        "10001",
        "10001",
        "10001",
        "10001",
        "01010",
        "01010",
        "00100",
    ],
    "W": [
        "10001",
        "10001",
        "10001",
        "10101",
        "10101",
        "10101",
        "01010",
    ],
    "X": [
        "10001",
        "01010",
        "00100",
        "00100",
        "00100",
        "01010",
        "10001",
    ],
    "Y": [
        "10001",
        "01010",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
    ],
    "Z": [
        "11111",
        "00001",
        "00010",
        "00100",
        "01000",
        "10000",
        "11111",
    ],
}


class Canvas:
    def __init__(self, width: int, height: int, background: Tuple[int, int, int, int] = (255, 255, 255, 255)) -> None:
        self.width = width
        self.height = height
        bg = bytes(background)
        self.pixels = bytearray(bg * (width * height))

    def set_pixel(self, x: int, y: int, color: Tuple[int, int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = (y * self.width + x) * 4
            self.pixels[idx : idx + 4] = bytes(color)

    def draw_line(self, x0: float, y0: float, x1: float, y1: float, color: Tuple[int, int, int, int], thickness: int = 1) -> None:
        x0_i, y0_i = int(round(x0)), int(round(y0))
        x1_i, y1_i = int(round(x1)), int(round(y1))
        dx = abs(x1_i - x0_i)
        dy = -abs(y1_i - y0_i)
        sx = 1 if x0_i < x1_i else -1
        sy = 1 if y0_i < y1_i else -1
        err = dx + dy
        while True:
            for tx in range(-(thickness // 2), thickness // 2 + 1):
                for ty in range(-(thickness // 2), thickness // 2 + 1):
                    self.set_pixel(x0_i + tx, y0_i + ty, color)
            if x0_i == x1_i and y0_i == y1_i:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0_i += sx
            if e2 <= dx:
                err += dx
                y0_i += sy

    def draw_polyline(self, points: Sequence[Tuple[float, float]], color: Tuple[int, int, int, int], thickness: int = 1) -> None:
        for i in range(len(points) - 1):
            self.draw_line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], color, thickness)

    def draw_text(self, x: int, y: int, text: str, color: Tuple[int, int, int, int], scale: int = 1) -> None:
        cursor_x = x
        upper = text.upper()
        for ch in upper:
            pattern = FONT_5X7.get(ch, FONT_5X7[" "])
            for row_idx, row in enumerate(pattern):
                for col_idx, bit in enumerate(row):
                    if bit == "1":
                        for dx in range(scale):
                            for dy in range(scale):
                                self.set_pixel(cursor_x + col_idx * scale + dx, y + row_idx * scale + dy, color)
            cursor_x += (len(pattern[0]) + 1) * scale

    def draw_rect(self, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int, int]) -> None:
        for x in range(x0, x1 + 1):
            self.set_pixel(x, y0, color)
            self.set_pixel(x, y1, color)
        for y in range(y0, y1 + 1):
            self.set_pixel(x0, y, color)
            self.set_pixel(x1, y, color)

    def fill_rect(self, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int, int]) -> None:
        for y in range(y0, y1):
            for x in range(x0, x1):
                self.set_pixel(x, y, color)


def write_png(path: str, canvas: Canvas) -> None:
    stride = canvas.width * 4
    raw = bytearray()
    for y in range(canvas.height):
        raw.append(0)
        start = y * stride
        raw.extend(canvas.pixels[start : start + stride])
    compressed = zlib.compress(bytes(raw))

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack("!I", len(data))
        crc = zlib.crc32(chunk_type)
        crc = zlib.crc32(data, crc) & 0xFFFFFFFF
        crc_bytes = struct.pack("!I", crc)
        return length + chunk_type + data + crc_bytes

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        ihdr = struct.pack("!IIBBBBB", canvas.width, canvas.height, 8, 6, 0, 0, 0)
        fh.write(chunk(b"IHDR", ihdr))
        fh.write(chunk(b"IDAT", compressed))
        fh.write(chunk(b"IEND", b""))


def choose_tick_step(max_value: float, target_ticks: int = 5) -> float:
    if max_value <= 0:
        return 1.0
    raw = max_value / max(1, target_ticks)
    power = 10 ** int(math.floor(math.log10(raw)))
    for mult in (1, 2, 5, 10):
        step = mult * power
        if step >= raw:
            return max(1.0, step)
    return max(1.0, 10 * power)


def render_line_chart(
    time_axis: Sequence[float],
    series: Dict[str, Sequence[float]],
    outfile: str,
    title: str,
    y_label: str,
    reference_line: Optional[float] = None,
) -> None:
    width, height = 960, 540
    left, right, top, bottom = 80, 200, 80, 80
    canvas = Canvas(width, height)
    colors = [
        (31, 119, 180, 255),
        (255, 127, 14, 255),
        (44, 160, 44, 255),
        (214, 39, 40, 255),
    ]

    plot_width = width - left - right
    plot_height = height - top - bottom

    if not time_axis or all(not values for values in series.values()):
        write_png(outfile, canvas)
        return

    x_min = 0.0
    x_max = max(time_axis)
    y_max = max(max(values) if values else 0.0 for values in series.values())
    if reference_line is not None:
        y_max = max(y_max, reference_line)
    if y_max <= 0:
        y_max = 1.0
    else:
        y_max *= 1.05

    x_range = max(x_max - x_min, 1e-6)
    y_min = 0.0
    y_range = max(y_max - y_min, 1e-6)

    def to_pixel(x_val: float, y_val: float) -> Tuple[float, float]:
        x = left + (x_val - x_min) / x_range * plot_width
        y = top + plot_height - (y_val - y_min) / y_range * plot_height
        return x, y

    axis_color = (0, 0, 0, 255)
    grid_color = (220, 220, 220, 255)

    # Axes
    canvas.draw_line(left, top, left, top + plot_height, axis_color)
    canvas.draw_line(left, top + plot_height, left + plot_width, top + plot_height, axis_color)

    x_step = choose_tick_step(x_max)
    tick = 0.0
    while tick <= x_max + 1e-6:
        px, py = to_pixel(tick, 0.0)
        canvas.draw_line(px, py, px, py + 6, axis_color)
        if tick > 0:
            canvas.draw_line(px, top, px, top + plot_height, grid_color)
        label = str(int(round(tick)))
        canvas.draw_text(int(px) - 6, int(py) + 10, label, axis_color)
        tick += x_step

    y_step = choose_tick_step(y_max)
    tick = y_min
    while tick <= y_max + 1e-6:
        px0, py0 = to_pixel(0.0, tick)
        canvas.draw_line(px0 - 6, py0, px0, py0, axis_color)
        if tick > y_min:
            canvas.draw_line(left, py0, left + plot_width, py0, grid_color)
        label = str(int(round(tick)))
        canvas.draw_text(left - 50, int(py0) - 4, label, axis_color)
        tick += y_step

    if reference_line is not None:
        _, ref_y = to_pixel(0.0, reference_line)
        canvas.draw_line(left, ref_y, left + plot_width, ref_y, (200, 0, 0, 255))
        canvas.draw_text(left + plot_width + 10, int(ref_y) - 4, "DEADLINE", (200, 0, 0, 255))

    # Series lines and legend
    for idx, (name, values) in enumerate(series.items()):
        if not values:
            continue
        color = colors[idx % len(colors)]
        points = [to_pixel(time_axis[i], values[i]) for i in range(min(len(time_axis), len(values)))]
        canvas.draw_polyline(points, color, thickness=2)
        legend_y = top + idx * 20
        canvas.draw_line(width - right + 10, legend_y + 5, width - right + 40, legend_y + 5, color, thickness=3)
        canvas.draw_text(width - right + 50, legend_y, name, axis_color)

    canvas.draw_text(left, 30, title, axis_color, scale=2)
    canvas.draw_text(20, top + plot_height // 2, y_label, axis_color)
    canvas.draw_text(left + plot_width // 2, height - 30, "TIME S", axis_color)

    write_png(outfile, canvas)


def plot_throughput(metrics: Metrics, outfile: str, scenario_label: str) -> None:
    series = {name.upper(): metrics.tp_series[name] for name in metrics.flow_names}
    render_line_chart(
        metrics.tp_time_axis,
        series,
        outfile,
        title=f"{scenario_label} THROUGHPUT",
        y_label="THROUGHPUT MBPS",
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
    )


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------
def run_scenario(num_vr_apps: int, save_prefix: str, delay_mode: str = "head") -> Dict[str, float]:
    random.seed(SEED)

    slice_a_flows = [Flow(f"app{i + 1}", "vr") for i in range(num_vr_apps)]
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
        deadline=VR_DEADLINE if delay_mode == "head" else None,
    )

    stats = {}
    for flow in slice_a_flows:
        stats[flow.name] = flow.total_sent_bits / duration_s / 1e6
    return stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    scenario2_stats = run_scenario(num_vr_apps=2, save_prefix="scenario2", delay_mode="head")
    scenario4_stats = run_scenario(num_vr_apps=4, save_prefix="scenario4", delay_mode="head")

    print("Scenario S2 average throughput (Mbps):")
    for name, value in scenario2_stats.items():
        print(f"  {name}: {value:.2f}")

    print("\nScenario S4 average throughput (Mbps):")
    for name, value in scenario4_stats.items():
        print(f"  {name}: {value:.2f}")


if __name__ == "__main__":
    main()