"""Device model: an event-level model of what the host can see of HBF.

Time is in microseconds, sizes in bytes, bandwidths in bytes per microsecond (1 GB/s = 1e3 B/us), energy in microjoules
and power in watts (1 W = 1 uJ/us).

One read (OCP v0.7.0 §5.2.4/§5.3):
  1. arrive at the channel's base die; beyond the in-flight cap (MOCS) it waits on the host side;
  2. a normal read is held until the base-die timer (TMR) fires; a Batch Read flushes the held reads and itself at once;
  3. dispatch to its unit (die x bank, one plane): a page already in one of the unit's 2 page buffers is a hit and goes
     straight to step 6;
  4. otherwise it joins the unit's sense queue (strict order); at the head it is a hit if an earlier sense loaded its page,
     else it takes a buffer (chosen by the replacement policy among the buffers with no transfer pending; with none free the
     unit stalls until a transfer ends), draws the energy of a sense from its stack's power budget (waiting, in request
     order, when the budget is short) and senses (tR with jitter, plus the v1 amortised read-disturb charge);
  5. the whole 4 KiB page then crosses the die's core-die channel into the base-die buffer (FIFO); the unit holds the page
     in its register, and senses nothing else, until the page is across;
  6. transfer the requested bytes over the channel link (FIFO in ready order, one at a time);
  7. complete: the host sees it at the end of the transfer.

Refresh (OCP §11.5): every sense adds pages_per_block / X pages of refresh work ("debt") to its unit. A
refresh operation rewrites one page (read tR, program t_prog): it waits until its units are idle (they start no new sense
meanwhile), draws (E_sense + E_prog) per page from the power budget, then holds them for tR + t_prog and retires up to one
page of debt from each. Scope "die": the operation holds every plane of a die and rewrites one page on each plane that owes
one (a multi-plane program); scope "plane": it holds one plane. Who decides when to pay is the host's software:
  - policy "default" (HBF-Sim's rule): pay one operation when a die (plane) is idle and owes at least one page; pay at once
    while any plane owes more than d_max;
  - policy "host": only what the host asks for (refresh(t, die, n)); the device still forces payment above 2 d_max.
A sense on a plane owing more than d_max is retried with probability retry_p (one more sense, one more energy draw).
settle() is the time still needed, at the end of a run, to pay what is owed. The refresh's own page transfers (4 KiB each
way, about 0.1% of the link at the rates here) are counted, not timed.

SRAM reads use the same channel link with priority over queued transfers (not over one in progress). HBM is its own link
(shared-position architecture). Everything random comes from the seed (splitmix64): two runs with one seed are identical.
The power budget, core-die channels, refresh debt and retries are each off at their defaults, which leaves the basic
model (channels, banks, page buffers, links) exactly as it is without them.

This file is the reference implementation. device_c.c is the same model in C, event for event and bit for bit, and is what
the runner uses (fastdevice.py); tests/test_device.py checks both.
"""
from collections import deque
import heapq
import math

ARRIVE, TIMER, SENSED, XFERRED, CDCDONE, POWER, REFDONE, HITREADY = 0, 1, 2, 3, 4, 5, 6, 7
IDLE, POWER_WAIT, SENSING, CDC, REFRESH = 0, 1, 2, 3, 4  # unit states
NONE, PENDING, PWAIT, RUNNING = 0, 1, 2, 3  # refresh operation slots
POLICIES = ("lru", "fifo", "random")
SCOPES = ("die", "plane")
REFRESH_POLICIES = ("default", "host")
PAGE = 4096
ALIGN = 64
BANKS_PER_DIE = 16  # the default; a device may set banks_per_die


class DeviceError(Exception):
    """A request the device would refuse (bad page, misaligned, crossing a page): the scheme's fault."""


MASK = (1 << 64) - 1


class SplitMix:
    """splitmix64, the generator device_c.c uses, so both implementations draw the same numbers."""

    def __init__(self, seed):
        self.s = seed & MASK

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def random(self):
        return (self.next() >> 11) * (1. / 9007199254740992.)

    def bit(self):
        return self.next() >> 63


def check_config(units, banks_per_die, channels, channels_per_stack, coredie, scope, refresh_policy, policy, buffers=2):
    if buffers not in (1, 2):
        raise ValueError("buffers must be 1 or 2")
    if policy not in POLICIES:
        raise ValueError("policy must be one of " + ", ".join(POLICIES))
    if banks_per_die < 1 or units % banks_per_die:
        raise ValueError("units per channel must be a multiple of banks_per_die (%d)" % banks_per_die)
    if channels_per_stack < 0 or (channels_per_stack and channels % channels_per_stack):
        raise ValueError("channels must be a multiple of channels_per_stack")
    dies = units // banks_per_die
    if coredie < 0 or (coredie and dies % coredie):
        raise ValueError("dies per channel (%d) must be a multiple of the core-die channels (%d)" % (dies, coredie))
    if scope not in SCOPES:
        raise ValueError("scope must be one of " + ", ".join(SCOPES))
    if refresh_policy not in REFRESH_POLICIES:
        raise ValueError("refresh_policy must be one of " + ", ".join(REFRESH_POLICIES))


class Device:
    def __init__(self, *, channels, units, tR, channel_bw, policy="lru", jitter=0., tmr=0., mocs=1 << 30,
                 interleave=1, disturb_us=0., pages_per_block=256, hbm_bw=0., hbm_latency=0., sram_latency=0., seed=0,
                 banks_per_die=BANKS_PER_DIE, channels_per_stack=0, power_w=0., power_depth=0., e_sense=0., e_prog=0.,
                 coredie=0, coredie_bw=0., refresh_x=0., t_prog=0., scope="die", d_max=math.inf, retry_p=0.,
                 refresh_policy="default", hits=True, buffers=2, hit_us=0.):
        check_config(units, banks_per_die, channels, channels_per_stack, coredie, scope, refresh_policy, policy, buffers)
        self.hits_on, self.nbuf, self.hit_us = bool(hits), int(buffers), float(hit_us)
        self.C, self.U, self.tR, self.bw = int(channels), int(units), float(tR), float(channel_bw)
        self.policy, self.jitter, self.tmr, self.mocs = policy, float(jitter), float(tmr), int(mocs)
        self.g, self.disturb, self.ppb = int(interleave), float(disturb_us), int(pages_per_block)
        self.hbm_bw, self.hbm_latency, self.sram_latency = float(hbm_bw), float(hbm_latency), float(sram_latency)
        self.banks = int(banks_per_die)
        self.dies = self.U // self.banks
        self.cps = int(channels_per_stack) or self.C
        self.S = self.C // self.cps
        self.P, self.pdepth, self.e_sense, self.e_prog = float(power_w), float(power_depth), float(e_sense), float(e_prog)
        self.ncdc, self.cbw = int(coredie), float(coredie_bw)
        self.X, self.t_prog, self.d_max, self.retry_p = float(refresh_x), float(t_prog), float(d_max), float(retry_p)
        self.plane_scope = scope == "plane"
        self.host_refresh = refresh_policy == "host"
        self.dinc = self.ppb / self.X if self.X > 0 else 0.
        self.thr = 2. * self.d_max if self.host_refresh else self.d_max
        self.rng = SplitMix(seed)
        self.heap, self.seq, self.now = [], 0, 0.
        # requests
        self.r_tag, self.r_ch, self.r_cu, self.r_page, self.r_bytes, self.r_batch, self.r_buf = [], [], [], [], [], [], []
        self.r_submit = []
        # channels
        C, CU = self.C, self.C * self.U
        self.inflight, self.waiting = [0] * C, [deque() for _ in range(C)]
        self.hold, self.timer_gen = [[] for _ in range(C)], [0] * C
        self.timer_on = [False] * C
        self.link_busy, self.link_free, self.xq = [False] * C, [0.] * C, [deque() for _ in range(C)]
        # units: two page buffers each (page, valid, lock count, stamp)
        self.queue = [deque() for _ in range(CU)]
        self.ustate, self.blocked = [IDLE] * CU, [False] * CU
        self.buf_page = [[-1, -1] for _ in range(CU)]
        self.buf_valid = [[False, False] for _ in range(CU)]
        self.buf_lock = [[0, 0] for _ in range(CU)]
        self.buf_stamp = [[0., 0.] for _ in range(CU)]
        self.sbuf = [0] * CU  # the buffer of the sense in progress
        self.pend_r, self.pend_v, self.pend_retry, self.pend_since = [0] * CU, [0] * CU, [0] * CU, [0.] * CU
        # refresh: debt per unit, operations in slots (per die for scope die, per unit for scope plane)
        self.debt, self.oret, self.res = [0.] * CU, [0.] * CU, [False] * CU
        self.oslot, self.opages = [NONE] * CU, [0.] * CU
        self.want, self.dmax = [0] * (C * self.dies), [0.] * (C * self.dies)
        # power budget, one per stack
        self.pL, self.ptL, self.pq, self.ppending = [self.pdepth] * self.S, [0.] * self.S, [deque() for _ in range(self.S)], [False] * self.S
        # core-die channels
        K = C * self.ncdc
        self.cq, self.cbusy, self.ctime = [deque() for _ in range(K)], [False] * K, [0.] * K
        self.hbm_free = 0.
        self.outstanding = 0
        self.open = {}  # request -> tag, from submission until handed to the host
        self.done = []  # (time, request) not yet handed to the host
        # statistics
        self.unit_senses, self.unit_busy = [0] * CU, [0.] * CU
        self.hits, self.senses = 0, 0
        self.link_time, self.link_bytes = [0.] * C, [0] * C
        self.block_reads = {}
        self.sram_bytes, self.hbm_bytes = 0, 0
        self.retries, self.ref_ops, self.ref_pages, self.ref_unit_us = 0, 0, 0., 0.
        self.energy, self.power_wait = 0., 0.

    # ---- address mapping (OCP §4.6): global page -> channel by the host's interleave, local page -> die/bank ----
    def locate(self, page):
        g, C = self.g, self.C
        channel = (page // g) % C
        local = g * (page // (g * C)) + page % g
        return channel, local

    def mapping(self, page):
        """(channel, die, bank) of a global page, what a host reads from the configuration registers."""
        channel, local = self.locate(page)
        return channel, (local // self.banks) % self.dies, local % self.banks

    # ---- host side ----
    def submit(self, t, pages, offsets, nbytes, tags, batch):
        """Requests issued by the host at time t (t never goes back). Returns the request ids."""
        if t < self.now - 1e-9:
            raise RuntimeError("device time went backwards: %r < %r" % (t, self.now))
        ids = []
        for page, off, n, tag in zip(pages, offsets, nbytes, tags):
            if n <= 0 or n % ALIGN or off % ALIGN or off < 0 or off + n > PAGE or page < 0:
                raise DeviceError("a read is 64 B to 4 KiB, 64 B aligned, inside one 4 KiB page (offset %d, %d bytes)" % (off, n))
            channel, local = self.locate(page)
            r = len(self.r_tag)
            self.r_tag.append(tag)
            self.r_ch.append(channel)
            self.r_cu.append(channel * self.U + local % self.U)
            self.r_page.append(local)
            self.r_bytes.append(n)
            self.r_batch.append(bool(batch))
            self.r_buf.append(-1)
            self.r_submit.append(t)
            self.outstanding += 1
            self.open[r] = tag
            self._push(t, ARRIVE, r)
            ids.append(r)
        return ids

    def refresh(self, t, die, n):
        """The host asks for n refresh operations on a die (global die id: channel * dies_per_channel + die) at time t."""
        if t < self.now - 1e-9:
            raise RuntimeError("device time went backwards: %r < %r" % (t, self.now))
        if not (0 <= die < self.C * self.dies) or n < 0:
            raise DeviceError("refresh: die must be in [0, %d) and n >= 0" % (self.C * self.dies))
        self.run_until(t)
        self.want[die] += n
        self._die_kick(die, t)

    def debt_per_die(self, t):
        """Pages of refresh work each die owes at time t: the most any of its planes owes (the read counts a host can query)."""
        self.run_until(t)
        return list(self.dmax)

    def settle(self):
        """Time to pay what is still owed (dies and planes pay in parallel): added to a run's completion time."""
        return max(self.debt) * (self.tR + self.t_prog) if self.X > 0 and self.debt else 0.

    def hbm_read(self, t, nbytes):
        """A synchronous HBM read issued at t; returns when its data is in."""
        if nbytes <= 0:
            return t
        if self.hbm_bw <= 0:
            raise DeviceError("no HBM in this configuration")
        begin = max(t, self.hbm_free)
        self.hbm_free = begin + nbytes / self.hbm_bw
        self.hbm_bytes += nbytes
        return self.hbm_free + self.hbm_latency

    def sram_read(self, t, per_channel):
        """A synchronous scratchpad read at t: {channel: bytes}; it takes each channel link ahead of queued transfers."""
        self.run_until(t)
        end = t
        for c, n in sorted(per_channel.items()):
            if n <= 0:
                continue
            begin = max(t, self.link_free[c])
            self.link_free[c] = begin + n / self.bw
            self.link_time[c] += n / self.bw
            self.sram_bytes += n
            end = max(end, self.link_free[c])
        return end + self.sram_latency if end > t else t

    def run_until(self, t):
        heap = self.heap
        while heap and heap[0][0] <= t:
            time, _, kind, a, b = heapq.heappop(heap)
            self.now = time
            if kind == ARRIVE:
                self._arrive(a, time)
            elif kind == SENSED:
                self._sensed(a, b, time)
            elif kind == XFERRED:
                self._xferred(a, b, time)
            elif kind == CDCDONE:
                self._cdc_done(a, b, time)
            elif kind == POWER:
                self._power_event(a, time)
            elif kind == REFDONE:
                self._refdone(a, time)
            elif kind == HITREADY:
                self._enqueue(a, time)
            elif b == self.timer_gen[a]:
                self._flush(a, time)
        if t > self.now:
            self.now = t

    def next_wake(self, cursor, count=1, timeout=math.inf, extra=0):
        """The time the host wakes after going idle at `cursor`: when `count` events (undelivered completions plus `extra`
        other events) are waiting, when `timeout` has passed, or when nothing is in flight."""
        self.run_until(cursor)
        deadline = cursor + timeout
        if len(self.done) + extra >= count or self.outstanding == 0:
            return cursor
        while self.heap:
            t = self.heap[0][0]
            if t > deadline:
                self.run_until(deadline)
                return deadline
            self.run_until(t)
            if len(self.done) + extra >= count or self.outstanding == 0:
                return t
        raise RuntimeError("device stalled with %d requests outstanding" % self.outstanding)

    def take_done(self):
        out, self.done = self.done, []
        for _, r in out:
            del self.open[r]
        return out

    def take_done_arrays(self):
        """The completions as numpy arrays: times, request ids, tags (what the runner uses)."""
        import numpy as np
        out = self.take_done()
        return (np.array([t for t, _ in out], dtype=np.float64), np.array([r for _, r in out], dtype=np.int64),
                np.array([self.r_tag[r] for _, r in out], dtype=np.int64))

    def done_count(self):
        return len(self.done)

    def pending_tags(self):
        """Tags of the requests the host has issued and not yet seen complete, in issue order."""
        return list(self.open.values())

    # ---- internals: requests ----
    def _push(self, t, kind, a, b=0):
        self.seq += 1
        heapq.heappush(self.heap, (t, self.seq, kind, a, b))

    def _arrive(self, r, t):
        c = self.r_ch[r]
        if self.inflight[c] >= self.mocs:
            self.waiting[c].append(r)
        else:
            self._admit(r, t)

    def _admit(self, r, t):
        c = self.r_ch[r]
        self.inflight[c] += 1
        if self.r_batch[r] or self.tmr <= 0:
            self.hold[c].append(r)
            self._flush(c, t)
        else:
            self.hold[c].append(r)
            if not self.timer_on[c]:
                self.timer_on[c] = True
                self._push(t + self.tmr, TIMER, c, self.timer_gen[c])

    def _flush(self, c, t):
        held, self.hold[c] = self.hold[c], []
        self.timer_on[c] = False
        self.timer_gen[c] += 1
        for r in held:
            self._dispatch(r, t)

    def _find(self, cu, page):
        if not self.hits_on:
            return -1
        pages, valid = self.buf_page[cu], self.buf_valid[cu]
        if pages[0] == page and valid[0]:
            return 0
        if self.nbuf > 1 and pages[1] == page and valid[1]:
            return 1
        return -1

    def _dispatch(self, r, t):
        cu = self.r_cu[r]
        v = self._find(cu, self.r_page[r])
        if v >= 0:  # a hit returns at once (OCP §5.3: subject only to ordering, which we do not model)
            self._hit(cu, v, r, t)
        else:
            self.queue[cu].append(r)
            self._unit_next(cu, t)

    def _hit(self, cu, v, r, t):
        self.hits += 1
        if self.policy == "lru":
            self.buf_stamp[cu][v] = t
        if self.hit_us > 0:  # the base-die buffer's own access time; the buffer is held meanwhile
            self.buf_lock[cu][v] += 1
            self.r_buf[r] = v
            self._push(t + self.hit_us, HITREADY, r, 0)
        else:
            self._ready(r, cu, v, t)

    def _victim(self, cu):
        pages, valid, lock, stamp = self.buf_page[cu], self.buf_valid[cu], self.buf_lock[cu], self.buf_stamp[cu]
        free = [v for v in range(self.nbuf) if lock[v] == 0 and (valid[v] or pages[v] == -1)]
        if not free:
            return -1
        empty = [v for v in free if pages[v] == -1]
        if empty:
            return empty[0]
        if len(free) == 1:
            return free[0]
        if self.policy == "random":
            return self.rng.bit()
        return 0 if stamp[0] <= stamp[1] else 1

    def _unit_next(self, cu, t):
        if self.ustate[cu] != IDLE or self.res[cu]:
            return
        q = self.queue[cu]
        while q:
            r = q[0]
            v = self._find(cu, self.r_page[r])
            if v >= 0:
                q.popleft()
                self._hit(cu, v, r, t)
                continue
            v = self._victim(cu)
            if v < 0:
                self.blocked[cu] = True
                return
            q.popleft()
            self.buf_page[cu][v], self.buf_valid[cu][v] = self.r_page[r], False
            retry = 0
            if self.retry_p > 0 and self.debt[cu] > self.d_max:
                retry = 1 if self.rng.random() < self.retry_p else 0
            if self._power_request(cu // self.U // self.cps, 0, cu, self.e_sense * (1 + retry), t):
                self._start_sense(cu, r, v, retry, t)
            else:
                self.ustate[cu] = POWER_WAIT
                self.pend_r[cu], self.pend_v[cu], self.pend_retry[cu], self.pend_since[cu] = r, v, retry, t
            return

    def _start_sense(self, cu, r, v, retry, t):
        duration = self.tR * (1. + self.jitter * (2. * self.rng.random() - 1.)) if self.jitter else self.tR
        if retry:
            duration += self.tR * (1. + self.jitter * (2. * self.rng.random() - 1.)) if self.jitter else self.tR
        duration += self.disturb
        self.ustate[cu] = SENSING
        self.sbuf[cu] = v
        self.unit_senses[cu] += 1 + retry
        self.unit_busy[cu] += duration
        self.senses += 1 + retry
        self.retries += retry
        block = (cu, self.r_page[r] // self.U // self.ppb)
        self.block_reads[block] = self.block_reads.get(block, 0) + 1
        if self.X > 0:
            self.debt[cu] += (1 + retry) * self.dinc
            gd = cu // self.banks
            if self.debt[cu] > self.dmax[gd]:
                self.dmax[gd] = self.debt[cu]
            if self.debt[cu] > self.thr:
                self._urgent(cu, t)
        self._push(t + duration, SENSED, r, v)

    def _sensed(self, r, v, t):
        cu = self.r_cu[r]
        if self.ncdc:
            self.ustate[cu] = CDC
            k = (cu // self.U) * self.ncdc + (cu % self.U) // self.banks * self.ncdc // self.dies
            self.cq[k].append(r)
            self._cdc_next(k, t)
            return
        self._land(cu, r, v, t)

    def _cdc_next(self, k, t):
        if self.cbusy[k] or not self.cq[k]:
            return
        r = self.cq[k].popleft()
        duration = PAGE / self.cbw
        self.cbusy[k] = True
        self.ctime[k] += duration
        self._push(t + duration, CDCDONE, k, r)

    def _cdc_done(self, k, r, t):
        self.cbusy[k] = False
        cu = self.r_cu[r]
        self._land(cu, r, self.sbuf[cu], t)
        self._cdc_next(k, t)

    def _land(self, cu, r, v, t):
        """The sensed page is in the base-die buffer: the unit is free, the request goes to the link."""
        self.buf_valid[cu][v] = True
        self.buf_stamp[cu][v] = t
        self.ustate[cu] = IDLE
        self._ready(r, cu, v, t)
        self._unit_next(cu, t)
        self._settled(cu, t)

    def _ready(self, r, cu, v, t):
        self.buf_lock[cu][v] += 1
        self.r_buf[r] = v
        self._enqueue(r, t)

    def _enqueue(self, r, t):
        c = self.r_ch[r]
        self.xq[c].append(r)
        self._link_next(c, t)

    def _link_next(self, c, t):
        if self.link_busy[c] or not self.xq[c]:
            return
        r = self.xq[c].popleft()
        begin = max(t, self.link_free[c])
        duration = self.r_bytes[r] / self.bw
        self.link_free[c] = begin + duration
        self.link_busy[c] = True
        self.link_time[c] += duration
        self.link_bytes[c] += self.r_bytes[r]
        self._push(begin + duration, XFERRED, c, r)

    def _xferred(self, c, r, t):
        self.link_busy[c] = False
        self.done.append((t, r))
        self.outstanding -= 1
        cu, v = self.r_cu[r], self.r_buf[r]
        self.r_buf[r] = -3  # finished
        self.buf_lock[cu][v] -= 1
        if self.blocked[cu]:
            self.blocked[cu] = False
            self._unit_next(cu, t)
            self._settled(cu, t)
        self.inflight[c] -= 1
        if self.waiting[c]:
            self._admit(self.waiting[c].popleft(), t)
        self._link_next(c, t)

    # ---- internals: power budget (one token bucket per stack) ----
    def _power_request(self, s, kind, ident, e, t):
        """kind 0: the pending sense of unit `ident`; kind 1: refresh operation slot `ident`. True if granted now."""
        if self.P <= 0:
            self.energy += e
            return True
        q = self.pq[s]
        if not q and not self.ppending[s]:
            level = min(self.pdepth, self.pL[s] + self.P * (t - self.ptL[s]))
            if level >= e:
                self.pL[s], self.ptL[s] = level - e, t
                self.energy += e
                return True
            q.append((kind, ident, e))
            self.ppending[s] = True
            self._push(t + (e - level) / self.P, POWER, s)
            return False
        q.append((kind, ident, e))
        return False

    def _power_event(self, s, t):
        q, first = self.pq[s], True
        while q:
            kind, ident, e = q[0]
            level = min(self.pdepth, self.pL[s] + self.P * (t - self.ptL[s]))
            if not first and level < e:
                self._push(t + (e - level) / self.P, POWER, s)
                return
            first = False  # the head was due now: grant it whatever rounding left
            q.popleft()
            self.pL[s], self.ptL[s] = level - e, t
            self.energy += e
            if kind == 0:
                self.power_wait += t - self.pend_since[ident]
                self._start_sense(ident, self.pend_r[ident], self.pend_v[ident], self.pend_retry[ident], t)
            else:
                self._op_start(ident, t)
        self.ppending[s] = False

    # ---- internals: refresh ----
    def _urgent(self, cu, t):
        """A plane owes more than the threshold: pay now (the plane, or its die, starts nothing new until paid)."""
        if self.plane_scope:
            if self.oslot[cu] == NONE:
                self._op_new(cu, t)
        else:
            gd = cu // self.banks
            if self.oslot[gd] == NONE and self.want[gd] == 0:
                self.want[gd] += 1
                self._die_kick(gd, t)

    def _die_kick(self, gd, t):
        """Start the operations asked for on a die, as far as its slots allow."""
        if not self.plane_scope:
            if self.oslot[gd] == NONE and self.want[gd] > 0:
                self.want[gd] -= 1
                self.oslot[gd] = PENDING
                for u in range(gd * self.banks, (gd + 1) * self.banks):
                    self.res[u] = True
                self._op_try(gd, t)
            return
        while self.want[gd] > 0:
            best = -1
            for u in range(gd * self.banks, (gd + 1) * self.banks):
                if self.oslot[u] == NONE and (best < 0 or self.debt[u] > self.debt[best]):
                    best = u
            if best < 0:
                return
            self.want[gd] -= 1
            self._op_new(best, t)

    def _op_new(self, cu, t):
        self.oslot[cu] = PENDING
        self.res[cu] = True
        self._op_try(cu, t)

    def _op_units(self, o):
        return range(o, o + 1) if self.plane_scope else range(o * self.banks, (o + 1) * self.banks)

    def _op_try(self, o, t):
        """An operation waiting for its units: once all are idle, draw its energy and start."""
        if self.oslot[o] != PENDING:
            return
        units = self._op_units(o)
        for u in units:
            if self.ustate[u] != IDLE:
                return
        pages = 0.
        for u in units:
            self.oret[u] = min(1., self.debt[u])
            pages += self.oret[u]
        self.oslot[o] = PWAIT
        self.opages[o] = pages
        if self._power_request(units[0] // self.U // self.cps, 1, o, (self.e_sense + self.e_prog) * pages, t):
            self._op_start(o, t)

    def _op_start(self, o, t):
        self.oslot[o] = RUNNING
        units = self._op_units(o)
        for u in units:
            self.ustate[u] = REFRESH
        duration = self.tR + self.t_prog
        self.ref_ops += 1
        self.ref_pages += self.opages[o]
        self.ref_unit_us += duration * len(units)
        self._push(t + duration, REFDONE, o)

    def _refdone(self, o, t):
        units = self._op_units(o)
        for u in units:
            self.debt[u] -= self.oret[u]
            self.oret[u] = 0.
            self.ustate[u] = IDLE
            self.res[u] = False
        self.oslot[o] = NONE
        gd = units[0] // self.banks
        top = 0.
        for u in range(gd * self.banks, (gd + 1) * self.banks):
            if self.debt[u] > top:
                top = self.debt[u]
        self.dmax[gd] = top
        if self.plane_scope:
            if self.debt[o] > self.thr:
                self._op_new(o, t)
        elif top > self.thr and self.want[gd] == 0:
            self.want[gd] += 1
        self._die_kick(gd, t)
        for u in units:
            self._unit_next(u, t)
        for u in units:
            self._settled(u, t)

    def _settled(self, cu, t):
        """A unit may have gone idle: an operation waiting for it may start; the default policy pays in idle time."""
        if self.res[cu]:
            self._op_try(cu if self.plane_scope else cu // self.banks, t)
            return
        if self.X <= 0 or self.host_refresh or self.ustate[cu] != IDLE or self.queue[cu]:
            return
        if self.plane_scope:
            if self.oslot[cu] == NONE and self.debt[cu] >= 1.:
                self._op_new(cu, t)
            return
        gd = cu // self.banks
        if self.oslot[gd] != NONE or self.want[gd] or self.dmax[gd] < 1.:
            return
        for u in range(gd * self.banks, (gd + 1) * self.banks):
            if self.ustate[u] != IDLE or self.queue[u]:
                return
        self.want[gd] += 1
        self._die_kick(gd, t)

    # ---- diagnostics ----
    def stats(self, span):
        CU = self.C * self.U
        loads = self.unit_busy
        mean = sum(loads) / CU if CU else 0.
        busiest = max(loads) if loads else 0.
        link_mean = sum(self.link_time) / self.C if self.C else 0.
        K = len(self.ctime)
        cdc_mean = sum(self.ctime) / K if K else 0.
        return {"senses": self.senses, "hits": self.hits, "requests": len(self.r_tag),
                "bytes": sum(self.link_bytes), "sram_bytes": self.sram_bytes, "hbm_bytes": self.hbm_bytes,
                "unit_rho": busiest / mean if mean > 0 else 0., "busiest_unit_util": busiest / span if span > 0 else 0.,
                "unit_util": mean / span if span > 0 else 0.,
                "channel_rho": max(self.link_time) / link_mean if link_mean > 0 else 0.,
                "channel_util": link_mean / span if span > 0 else 0.,
                "hottest_block_reads": max(self.block_reads.values()) if self.block_reads else 0,
                "coredie_rho": max(self.ctime) / cdc_mean if cdc_mean > 0 else 0.,
                "coredie_util": cdc_mean / span if span > 0 else 0.,
                "retries": self.retries, "refresh_ops": self.ref_ops, "refresh_pages": self.ref_pages,
                "refresh_unit_share": self.ref_unit_us / (CU * span) if span > 0 and CU else 0.,
                "max_debt": max(self.debt) if self.debt else 0., "energy_uj": self.energy,
                "power_wait_us": self.power_wait, "settle_us": self.settle()}
