"""Simulation of bandwidth-provisioned slices using a simplified NVS scheduler.

This module implements the experiment described in the task statement. Two
bandwidth-provisioned slices share a radio resource grid that is scheduled in
1 ms Transmission Time Intervals (TTI). Slice A carries VR rendering flows
while slice B serves a file synchronisation flow. The simulator first runs a
scenario with two VR applications and then doubles the VR count without
changing slice A's reserved bandwidth to expose application-level SLA
violations even when slice-level averages remain satisfied.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from statistics import mean
from typing import Deque, Dict, Iterable, List, Optional


def eff_min_rate(r_min_bps: float, phi: float, phi_ref: float) -> float:
    """Minimum guaranteed rate with reference-rate based scaling."""

    if phi_ref <= 0.0:
        return r_min_bps
    if phi >= phi_ref:
        return r_min_bps
    return r_min_bps * phi / max(phi_ref, 1e-9)


def slice_weight_bandwidth(r_min_bps: float, phi: float, phi_ref: float, r_bar: float) -> float:
    """Weight for a bandwidth-provisioned slice, mirroring NVS equation (7)."""

    if phi <= 0.0:
        return 0.0
    r_eff = eff_min_rate(r_min_bps, phi, phi_ref)
    c_i = r_eff / max(phi_ref, 1e-9)
    return c_i * phi / max(r_bar, 1e-6)


@dataclass
class Packet:
    size_bits: int
    arrival_ts: int
    deadline_ts: int
    remaining_bits: int = field(init=False)
    depart_ts: Optional[int] = None

    def __post_init__(self) -> None:
        self.remaining_bits = self.size_bits


class Flow:
    """Traffic source attached to a slice."""

    def __init__(
        self,
        name: str,
        slice_id: str,
        flow_type: str,
        rng: random.Random,
        traffic_params: Dict[str, float],
        channel_params: Dict[str, float],
    ) -> None:
        self.name = name
        self.slice_id = slice_id
        self.flow_type = flow_type
        self.rng = rng
        self.traffic_params = traffic_params

        self.queue: Deque[Packet] = deque()

        if flow_type == "vr":
            self.frame_interval_ms = traffic_params.get("frame_interval_ms", 1000.0 / 60.0)
            self.next_arrival_ts = 0.0
            self.deadline_ms = traffic_params.get("deadline_ms", 20.0)
            self.sla_ms = traffic_params.get("sla_ms", 20.0)
            self.target_rate_bps = traffic_params["target_rate_mbps"] * 1e6
            self.burst_std_ratio = traffic_params.get("burst_std_ratio", 0.15)
            self.packets_per_burst = max(1, int(traffic_params.get("packets_per_burst", 3)))
        elif flow_type == "filesync":
            self.next_arrival_ts = 0.0
            self.deadline_ms = traffic_params.get("deadline_ms", 100.0)
            self.sla_ms = traffic_params.get("sla_ms", 20.0)
            self.target_rate_bps = traffic_params["target_rate_mbps"] * 1e6
            self.mean_inter_arrival = traffic_params.get("mean_inter_arrival_ms", 8.0)
            self.burst_shape = traffic_params.get("burst_shape", 0.35)
        else:
            raise ValueError(f"Unsupported flow type: {flow_type}")

        self.good_phi = channel_params["good_phi"]
        self.bad_phi = channel_params["bad_phi"]
        self.p_good_to_bad = channel_params.get("p_good_to_bad", 0.1)
        self.p_bad_to_good = channel_params.get("p_bad_to_good", 0.25)
        self.channel_state = "good" if rng.random() < 0.5 else "bad"
        self.current_phi = self.good_phi if self.channel_state == "good" else self.bad_phi

        self.generated_bits = 0
        self.generated_packets = 0
        self.sent_bits = 0
        self.delivered_bits = 0
        self.completed_packets = 0
        self.deadline_misses = 0
        self.expired_bits = 0
        self.latencies: List[float] = []

    def has_backlog(self) -> bool:
        return bool(self.queue)

    def maybe_generate(self, now_ms: int) -> None:
        if self.flow_type == "vr":
            while now_ms >= self.next_arrival_ts - 1e-9:
                frame_interval = self.frame_interval_ms
                mean_bits = self.target_rate_bps * (frame_interval / 1000.0)
                std_bits = mean_bits * self.burst_std_ratio
                burst_bits = max(1, int(self.rng.gauss(mean_bits, std_bits)))
                per_packet = max(1, burst_bits // self.packets_per_burst)
                remaining = burst_bits
                for _ in range(self.packets_per_burst):
                    size = per_packet if remaining >= per_packet else remaining
                    remaining -= size
                    if size <= 0:
                        continue
                    pkt = Packet(
                        size_bits=size,
                        arrival_ts=now_ms,
                        deadline_ts=int(now_ms + self.deadline_ms),
                    )
                    self.queue.append(pkt)
                    self.generated_bits += size
                    self.generated_packets += 1
                self.next_arrival_ts += frame_interval
        elif self.flow_type == "filesync":
            while now_ms >= self.next_arrival_ts - 1e-9:
                interval = max(1e-3, self.rng.expovariate(1.0 / max(self.mean_inter_arrival, 1e-3)))
                self.next_arrival_ts = now_ms + interval
                mean_bits = self.target_rate_bps * (interval / 1000.0)
                std_bits = mean_bits * self.burst_shape
                burst_bits = max(1, int(self.rng.gauss(mean_bits, std_bits)))
                pkt = Packet(
                    size_bits=burst_bits,
                    arrival_ts=now_ms,
                    deadline_ts=int(now_ms + self.deadline_ms),
                )
                self.queue.append(pkt)
                self.generated_bits += burst_bits
                self.generated_packets += 1
        else:
            raise AssertionError("unknown flow type")

    def drop_expired(self, now_ms: int) -> None:
        while self.queue and self.queue[0].deadline_ts < now_ms:
            pkt = self.queue.popleft()
            self.deadline_misses += 1
            self.expired_bits += pkt.remaining_bits

    def step_channel(self) -> None:
        if self.channel_state == "good":
            if self.rng.random() < self.p_good_to_bad:
                self.channel_state = "bad"
        else:
            if self.rng.random() < self.p_bad_to_good:
                self.channel_state = "good"
        self.current_phi = self.good_phi if self.channel_state == "good" else self.bad_phi

    def record_departure(self, pkt: Packet, depart_ts: int) -> None:
        pkt.depart_ts = depart_ts
        self.latencies.append(pkt.depart_ts - pkt.arrival_ts)
        self.completed_packets += 1
        self.delivered_bits += pkt.size_bits

    def sla_violation_ratio(self) -> Optional[float]:
        total_packets = len(self.latencies) + self.deadline_misses
        if total_packets == 0:
            return None
        violations = sum(1 for lat in self.latencies if lat > self.sla_ms) + self.deadline_misses
        return violations / total_packets

    def latency_p99(self) -> Optional[float]:
        samples = list(self.latencies)
        samples.extend([self.deadline_ms + 1.0] * self.deadline_misses)
        if not samples:
            return None
        sorted_lats = sorted(samples)
        if len(sorted_lats) == 1:
            return sorted_lats[0]
        rank = max(0, math.ceil(0.99 * len(sorted_lats)) - 1)
        return sorted_lats[min(rank, len(sorted_lats) - 1)]

    def deadline_miss_ratio(self) -> Optional[float]:
        if self.generated_packets == 0:
            return None
        return self.deadline_misses / self.generated_packets


class Slice:
    def __init__(self, slice_id: str, r_min_mbps: float, phi_ref: float, flows: Iterable[Flow]) -> None:
        self.slice_id = slice_id
        self.r_min_bps = r_min_mbps * 1e6
        self.phi_ref = phi_ref
        self.flows: List[Flow] = list(flows)
        self.r_bar = 1.0
        self.x_bar = 0.0
        self.total_sent_bits = 0
        self.time_selected = 0

    def has_backlog(self) -> bool:
        return any(flow.has_backlog() for flow in self.flows)

    def estimate_phi(self) -> float:
        if not self.flows:
            return 0.0
        active = [flow.current_phi for flow in self.flows if flow.has_backlog()]
        if active:
            return mean(active)
        return mean(flow.current_phi for flow in self.flows)


def schedule_weighted_fair(slice_obj: Slice, rb_budget: int, phi_bits_per_rb: float, now_ms: int) -> int:
    sent_bits = 0
    if not slice_obj.flows or rb_budget <= 0 or phi_bits_per_rb <= 0:
        return 0

    flows = slice_obj.flows
    bits_per_rb = max(1, int(round(phi_bits_per_rb)))
    start_index = 0

    while rb_budget > 0:
        any_served = False
        for offset in range(len(flows)):
            idx = (start_index + offset) % len(flows)
            flow = flows[idx]
            if not flow.queue:
                continue
            pkt = flow.queue[0]
            to_send = min(bits_per_rb, pkt.remaining_bits)
            if to_send <= 0:
                to_send = pkt.remaining_bits
            pkt.remaining_bits -= to_send
            flow.sent_bits += to_send
            sent_bits += to_send
            rb_budget -= 1
            any_served = True
            if pkt.remaining_bits <= 0:
                flow.queue.popleft()
                flow.record_departure(pkt, now_ms)
            start_index = (idx + 1) % len(flows)
            break
        if not any_served:
            break
    return sent_bits


class Simulation:
    def __init__(self, config: Dict, seed: int = 0) -> None:
        self.config = config
        self.duration_ms = config["duration_ms"]
        self.base_rb = config["base_rb"]
        self.fade_period = config.get("rb_fade_period", 500)
        self.beta = config.get("beta", 0.05)
        self.rng = random.Random(seed)

        self.slices: Dict[str, Slice] = {}
        self.all_flows: List[Flow] = []

        for slice_conf in config["slices"]:
            flows = []
            for flow_conf in slice_conf["flows"]:
                flow_rng = random.Random(self.rng.randint(0, 1_000_000))
                flow = Flow(
                    name=flow_conf["name"],
                    slice_id=slice_conf["id"],
                    flow_type=flow_conf["type"],
                    rng=flow_rng,
                    traffic_params=flow_conf["traffic"],
                    channel_params=flow_conf["channel"],
                )
                flows.append(flow)
                self.all_flows.append(flow)
            self.slices[slice_conf["id"]] = Slice(
                slice_id=slice_conf["id"],
                r_min_mbps=slice_conf["r_min_mbps"],
                phi_ref=slice_conf["phi_ref"],
                flows=flows,
            )

    def rb_capacity(self, tti: int) -> int:
        fade = math.sin(2 * math.pi * (tti % self.fade_period) / max(self.fade_period, 1))
        jitter = self.rng.gauss(0.0, 0.05)
        capacity = int(round(self.base_rb * (1.0 + 0.1 * fade + 0.05 * jitter)))
        return max(50, capacity)

    def run(self) -> Dict[str, Dict]:
        for tti in range(self.duration_ms):
            rb_budget = self.rb_capacity(tti)

            for flow in self.all_flows:
                flow.step_channel()
                flow.maybe_generate(tti)
                flow.drop_expired(tti)

            phi_map = {sid: slice_obj.estimate_phi() for sid, slice_obj in self.slices.items()}

            weights = {
                sid: slice_weight_bandwidth(
                    slice_obj.r_min_bps,
                    phi_map[sid],
                    slice_obj.phi_ref,
                    slice_obj.r_bar,
                )
                for sid, slice_obj in self.slices.items()
            }

            selected_slice: Optional[Slice] = None
            for sid in sorted(weights, key=weights.get, reverse=True):
                candidate = self.slices[sid]
                if candidate.has_backlog():
                    selected_slice = candidate
                    break

            sent_bits = 0
            if selected_slice is not None and rb_budget > 0:
                sent_bits = schedule_weighted_fair(
                    selected_slice,
                    rb_budget=rb_budget,
                    phi_bits_per_rb=phi_map[selected_slice.slice_id],
                    now_ms=tti,
                )
                selected_slice.total_sent_bits += sent_bits
                if sent_bits > 0:
                    selected_slice.time_selected += 1

            for sid, slice_obj in self.slices.items():
                if slice_obj is selected_slice:
                    r_inst = sent_bits * 1000.0
                    slice_obj.r_bar = (1.0 - self.beta) * slice_obj.r_bar + self.beta * r_inst
                    slice_obj.x_bar = (1.0 - self.beta) * slice_obj.x_bar + self.beta * 1.0
                else:
                    slice_obj.r_bar = (1.0 - self.beta) * slice_obj.r_bar
                    slice_obj.x_bar = (1.0 - self.beta) * slice_obj.x_bar

        duration_s = self.duration_ms / 1000.0
        results: Dict[str, Dict] = {}

        for sid, slice_obj in self.slices.items():
            slice_metrics = {
                "average_throughput_mbps": slice_obj.total_sent_bits / duration_s / 1e6,
                "time_fraction": slice_obj.time_selected / self.duration_ms,
                "r_bar_final_mbps": slice_obj.r_bar / 1e6,
            }
            flow_metrics = []
            for flow in slice_obj.flows:
                flow_metrics.append(
                    {
                        "flow": flow.name,
                        "type": flow.flow_type,
                        "avg_throughput_mbps": flow.sent_bits / duration_s / 1e6,
                        "p99_latency_ms": flow.latency_p99(),
                        "sla_violation_ratio": flow.sla_violation_ratio(),
                        "deadline_miss_ratio": flow.deadline_miss_ratio(),
                        "delivered_packets": flow.completed_packets,
                        "generated_packets": flow.generated_packets,
                    }
                )
            results[sid] = {"slice_metrics": slice_metrics, "flow_metrics": flow_metrics}

        return results


def build_scenario(num_vr_apps: int) -> Dict:
    vr_flows = []
    for idx in range(num_vr_apps):
        vr_flows.append(
            {
                "name": f"VR-{idx + 1}",
                "type": "vr",
                "traffic": {
                    "target_rate_mbps": 13.5,
                    "frame_interval_ms": 1000.0 / 60.0,
                    "deadline_ms": 20.0,
                    "sla_ms": 20.0,
                    "burst_std_ratio": 0.2,
                    "packets_per_burst": 2,
                },
                "channel": {
                    "good_phi": 300.0,
                    "bad_phi": 140.0,
                    "p_good_to_bad": 0.12,
                    "p_bad_to_good": 0.26,
                },
            }
        )

    file_sync_flow = {
        "name": "FileSync",
        "type": "filesync",
        "traffic": {
            "target_rate_mbps": 22.5,
            "deadline_ms": 100.0,
            "sla_ms": 20.0,
            "mean_inter_arrival_ms": 8.0,
            "burst_shape": 0.2,
        },
        "channel": {
            "good_phi": 300.0,
            "bad_phi": 150.0,
            "p_good_to_bad": 0.10,
            "p_bad_to_good": 0.24,
        },
    }

    slice_a_phi_ref = 0.9 * (0.5 * (300.0 + 140.0))
    slice_b_phi_ref = 0.9 * (0.5 * (300.0 + 150.0))

    return {
        "duration_ms": 60_000,
        "base_rb": 273,
        "rb_fade_period": 800,
        "beta": 0.05,
        "slices": [
            {
                "id": "A",
                "r_min_mbps": 24.0,
                "phi_ref": slice_a_phi_ref,
                "flows": vr_flows,
            },
            {
                "id": "B",
                "r_min_mbps": 20.0,
                "phi_ref": slice_b_phi_ref,
                "flows": [file_sync_flow],
            },
        ],
    }


def render_results(title: str, results: Dict[str, Dict]) -> str:
    lines = [title]
    for sid, slice_result in sorted(results.items()):
        sm = slice_result["slice_metrics"]
        lines.append(
            f"  Slice {sid}: avg throughput={sm['average_throughput_mbps']:.2f} Mbps, "
            f"time share={sm['time_fraction']*100:.1f} %, r_bar={sm['r_bar_final_mbps']:.2f} Mbps"
        )
        for flow in slice_result["flow_metrics"]:
            p99 = "n/a" if flow["p99_latency_ms"] is None else f"{flow['p99_latency_ms']:.1f}"
            sla = "n/a" if flow["sla_violation_ratio"] is None else f"{flow['sla_violation_ratio']*100:.1f}%"
            miss = "n/a" if flow["deadline_miss_ratio"] is None else f"{flow['deadline_miss_ratio']*100:.1f}%"
            lines.append(
                "    "
                f"{flow['flow']:<8} type={flow['type']:<9} avgThroughput={flow['avg_throughput_mbps']:.2f} Mbps "
                f"p99={p99} ms SLA_violation={sla} deadline_miss={miss} "
                f"delivered={flow['delivered_packets']}/{flow['generated_packets']}"
            )
    return "\n".join(lines)


def main() -> None:
    scenario1 = build_scenario(num_vr_apps=2)
    scenario2 = build_scenario(num_vr_apps=4)

    sim1 = Simulation(scenario1, seed=42)
    res1 = sim1.run()
    sim2 = Simulation(scenario2, seed=43)
    res2 = sim2.run()

    print(render_results("Scenario 1: Slice A with 2 VR applications", res1))
    print()
    print(render_results("Scenario 2: Slice A with 4 VR applications", res2))


if __name__ == "__main__":
    main()
