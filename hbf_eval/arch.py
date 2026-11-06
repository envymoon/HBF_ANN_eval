"""How HBM and HBF attach to the GPU .

Shared position (the first experiment family): 8 positions, m of them HBM and 8 - m HBF, each with its own links, so HBM and
HBF transfers run at the same time: T = max(D_H / (m B), D_F / ((8 - m) B)) for streamed reads (Characterizing HBF eq. 5).
m is fixed by the platform (the task), not chosen by the scheme. Each HBF position is one stack: its own
power budget. H3 (HBF behind HBM on every position, one link) is the second family and is not built yet.

Scaling: a position keeps `channels_per_position` of a stack's 16 channels, and its HBM bandwidth and its
power budget are cut by the same factor, so the HBM : HBF bandwidth ratio and the power per channel of a full stack are kept.
"""
import math

from .device import PAGE
from .fastdevice import Device
from .tasks import BURST_TR as BURST

POSITIONS = 8
STACK_CHANNELS = 16
HBF_PER_POSITION = 3.  # HBF capacity of one position, in multiples of the data size
REFRESH = {"default": "default", "scheme": "host"}  # the plan's choice -> the device's refresh policy


class Geometry:
    """Everything fixed by the platform (m) and the device: link counts, bandwidths, budgets, capacities, power."""

    def __init__(self, weather, data_bytes, m=None, arch="shared"):
        if arch != "shared":
            raise NotImplementedError("the H3 architecture is the second experiment family and is not built yet")
        m = weather["m"] if m is None else m
        if not (isinstance(m, int) and 0 <= m < POSITIONS):
            raise ValueError("m (HBM positions) must be an int in [0, 7]")
        w = weather
        self.m, self.cpp = m, int(w["channels_per_position"])
        scale = self.cpp / STACK_CHANNELS
        self.channels = (POSITIONS - m) * self.cpp
        self.units = int(w["units_per_channel"])
        self.banks = int(w["banks_per_die"])
        self.channel_bw = w["stack_bw"] / STACK_CHANNELS
        self.coredie = int(w["coredie"])
        self.coredie_bw = self.channel_bw / self.coredie if self.coredie else 0.  # OCP gives none: link / channels
        self.power = w["stack_power_w"] * scale  # per position (stack), W = uJ/us
        self.e_sense, self.e_prog = w["e_sense"], w["e_prog"]
        self.hbm_bw = m * w["hbm_stack_bw"] * scale
        self.hbm_latency = w["hbm_latency"]
        self.hbm_budget = int(m * w["hbm_fraction"] * data_bytes / POSITIONS)
        self.hbf_pages = (POSITIONS - m) * int(math.ceil(HBF_PER_POSITION * data_bytes / PAGE))
        self.sram_per_channel = int(w["sram_per_channel"])
        self.sram_budget = self.channels * self.sram_per_channel
        self.sram_latency = w["sram_latency"]

    @staticmethod
    def options(weather, data_bytes):
        """What layout sees: the platform, per-position figures and the nominal device parameters (a datasheet's numbers).
        Not shown: the firmware details (buffer policy, timer, jitter) and the read-disturb threshold."""
        w, scale = weather, weather["channels_per_position"] / STACK_CHANNELS
        m = int(w["m"])
        return {"m": m, "positions": POSITIONS, "channels_per_position": int(w["channels_per_position"]),
                "units_per_channel": int(w["units_per_channel"]), "banks_per_die": int(w["banks_per_die"]),
                "dies_per_channel": int(w["units_per_channel"]) // int(w["banks_per_die"]),
                "tR_us": w["tR_nominal"], "channel_GBps": w["stack_bw"] / STACK_CHANNELS / 1e3,
                "coredie_channels": int(w["coredie"]),
                "power_W_per_position": w["stack_power_w"] * scale, "sense_uJ": w["e_sense"], "program_uJ": w["e_prog"],
                "program_us": w["t_prog"], "program_holds": w["scope"], "pages_per_block": int(w["pages_per_block"]),
                "refresh_slack_pages": w["d_max"],
                "hbm_GBps_per_position": w["hbm_stack_bw"] * scale / 1e3, "hbm_latency_us": w["hbm_latency"],
                "hbm_bytes_per_position": int(w["hbm_fraction"] * data_bytes / POSITIONS),
                "hbm_bytes": int(m * w["hbm_fraction"] * data_bytes / POSITIONS),
                "hbf_pages_per_position": int(math.ceil(HBF_PER_POSITION * data_bytes / PAGE)),
                "sram_bytes_per_channel": int(w["sram_per_channel"]), "sram_latency_us": w["sram_latency"],
                "mocs": int(w["mocs"])}

    def device(self, weather, interleave, seed, refresh="default"):
        w = weather
        return Device(channels=self.channels, units=self.units, tR=w["tR"], channel_bw=self.channel_bw, policy=w["policy"],
                      jitter=w["jitter"], tmr=w["tmr"], mocs=int(w["mocs"]), interleave=interleave,
                      pages_per_block=int(w["pages_per_block"]), hbm_bw=self.hbm_bw, hbm_latency=self.hbm_latency,
                      sram_latency=self.sram_latency, seed=seed, banks_per_die=self.banks, channels_per_stack=self.cpp,
                      power_w=self.power, power_depth=self.power * BURST * w["tR"], e_sense=self.e_sense, e_prog=self.e_prog,
                      coredie=self.coredie, coredie_bw=self.coredie_bw, refresh_x=w["disturb_threshold"], t_prog=w["t_prog"],
                      scope=w["scope"], d_max=w["d_max"], retry_p=w["retry_p"], refresh_policy=REFRESH[refresh],
                      hit_us=self.sram_latency)  # a buffer hit reads the base-die buffer (SRAM) before the link

