"""Known-answer tests of the device model: every expected time is worked out by hand.

Every test runs on both implementations: the Python reference (device.py) and the C model the runner uses (device_c.c,
classes with the suffix C). Equivalence checks that the two give identical traces on random workloads.
"""
import math
import random
import unittest

from hbf_eval import fastdevice
from hbf_eval.device import POLICIES, Device, DeviceError

TR, BW = 10., 1024.  # tR 10 us; channel link 1024 B/us, so 64 B take 1/16 us and a page 4 us
X64, X4K = 64 / BW, 4096 / BW


IMPL = {"Device": Device}


def dev(**kw):
    args = dict(channels=1, units=16, tR=TR, channel_bw=BW)
    args.update(kw)
    return IMPL["Device"](**args)


def run(d, reads, t=0., batch=False):
    """Submit [(page, nbytes)] at t, run to the end, return completion times in request order."""
    ids = d.submit(t, [p for p, _ in reads], [0] * len(reads), [n for _, n in reads], list(range(len(reads))), batch)
    d.next_wake(t, count=10 ** 9)
    done = dict((r, time) for time, r in d.take_done())
    return [done[r] for r in ids]


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


class SingleReads(unittest.TestCase):
    def test_one_read_is_tR_plus_transfer(self):
        self.assertTrue(close(run(dev(), [(0, 4096)])[0], TR + X4K))
        self.assertTrue(close(run(dev(), [(5, 64)])[0], TR + X64))

    def test_a_64_byte_read_still_senses_a_whole_page(self):
        d = dev()
        run(d, [(0, 64)])
        self.assertEqual(d.senses, 1)
        self.assertTrue(close(d.unit_busy[0], TR))

    def test_reads_must_be_aligned_and_inside_a_page(self):
        for off, n in ((32, 64), (0, 100), (4032, 128), (0, 0), (0, 8192)):
            with self.assertRaises(DeviceError):
                dev().submit(0., [0], [off], [n], [0], False)


class Parallelism(unittest.TestCase):
    def test_same_bank_different_pages_is_strictly_serial(self):
        # pages 0, 16, 32, ... all map to bank 0 of die 0 (16 units = 1 die x 16 banks)
        k = 5
        times = run(dev(), [(16 * i, 64) for i in range(k)])
        for i, t in enumerate(times):
            self.assertTrue(close(t, (i + 1) * TR + X64), (i, t))

    def test_different_banks_overlap_their_senses_and_share_the_link(self):
        k = 8
        times = run(dev(), [(i, 64) for i in range(k)])
        for i, t in enumerate(times):
            self.assertTrue(close(t, TR + (i + 1) * X64), (i, t))
        self.assertTrue(close(max(times), TR + k * X64))

    def test_different_channels_overlap_everything(self):
        k = 4
        times = run(dev(channels=k), [(i, 4096) for i in range(k)])  # interleave of one page: page i -> channel i
        for t in times:
            self.assertTrue(close(t, TR + X4K))

    def test_address_mapping_follows_the_default_striping(self):
        d = dev(channels=4, units=64, interleave=2)
        # interleave 2 pages: pages 0,1 -> channel 0; 2,3 -> channel 1; ...; 8,9 -> channel 0 again (local pages 2,3)
        self.assertEqual(d.locate(0), (0, 0))
        self.assertEqual(d.locate(3), (1, 1))
        self.assertEqual(d.locate(9), (0, 3))
        # local page -> bank = local mod 16, die = (local / 16) mod 4
        self.assertEqual(d.mapping(8 * 17), (0, 2, 2))  # global 136 -> channel 0, local 34 -> die 2, bank 2


class Buffers(unittest.TestCase):
    def test_repeated_reads_of_one_page_sense_once(self):
        d = dev()
        times = run(d, [(0, 64), (0, 64), (0, 128)])
        self.assertEqual(d.senses, 1)
        self.assertEqual(d.hits, 2)
        self.assertTrue(close(times[0], TR + X64))
        self.assertTrue(close(times[1], TR + 2 * X64))
        self.assertTrue(close(times[2], TR + 2 * X64 + 128 / BW))

    def sequence(self, policy, seed=0):
        """A, B, A, C, A on one bank, each issued after the previous completes; returns the number of senses."""
        d, t = dev(policy=policy, seed=seed), 0.
        for page in (0, 16, 0, 32, 0):
            t = run(d, [(page, 64)], t)[0]
        return d.senses

    def test_lru_keeps_the_page_used_last(self):
        self.assertEqual(self.sequence("lru"), 3)  # C evicts B, A hits

    def test_fifo_evicts_the_page_loaded_first(self):
        self.assertEqual(self.sequence("fifo"), 4)  # C evicts A, A misses

    def test_random_is_seeded(self):
        self.assertEqual({self.sequence("random", 1) for _ in range(3)}, {self.sequence("random", 1)})
        self.assertTrue({self.sequence("random", s) for s in range(20)} <= {3, 4})

    def test_a_buffer_with_a_pending_transfer_is_not_evicted(self):
        # link 1000x slower: page 0's transfer (4 KiB) runs to 10 + 4000; bank 0 senses 16 into the other buffer at 10..20,
        # then page 32 needs a buffer: both are locked until their transfers end, so its sense starts at 10 + 4000.
        slow = BW / 1000
        d = dev(channel_bw=slow)
        times = run(d, [(0, 4096), (16, 4096), (32, 4096)])
        self.assertTrue(close(times[0], TR + 4096 / slow))
        self.assertTrue(close(times[1], TR + 2 * 4096 / slow))
        # page 32's sense starts when page 0's transfer ends, and its transfer waits for page 16's
        self.assertTrue(close(times[2], TR + 3 * 4096 / slow))
        self.assertEqual(d.senses, 3)


    # layer-2 attribution switches (off by default): hits=False, buffers never hit; buffers=1, one buffer per bank
    def test_without_hits_every_read_senses(self):
        d = dev(hits=False)
        times = run(d, [(0, 64), (0, 64), (0, 128)])
        self.assertEqual((d.senses, d.hits), (3, 0))
        # each sense takes a free buffer; the first buffer is free again (its 64 B across at 10 + 1/16) by 20
        self.assertTrue(close(times[0], TR + X64))
        self.assertTrue(close(times[1], 2 * TR + X64))
        self.assertTrue(close(times[2], 3 * TR + 128 / BW))

    def test_one_buffer_still_hits_its_page(self):
        d = dev(buffers=1)
        run(d, [(0, 64), (0, 64), (0, 128)])
        self.assertEqual((d.senses, d.hits), (1, 2))

    def test_one_buffer_keeps_only_the_last_page(self):
        d, t = dev(buffers=1), 0.
        for page in (0, 16, 0, 32, 0):
            t = run(d, [(page, 64)], t)[0]
        self.assertEqual(d.senses, 5)

    def test_one_buffer_serialises_sense_and_transfer(self):
        # the only buffer is locked until its page is across, so the next sense waits for the transfer
        slow = BW / 1000
        d = dev(channel_bw=slow, buffers=1)
        times = run(d, [(0, 4096), (16, 4096), (32, 4096)])
        for i in range(3):
            self.assertTrue(close(times[i], (i + 1) * (TR + 4096 / slow)))

    def test_a_hit_waits_the_buffer_access_time(self):
        # hit_us = 1: both later reads hit when page 0 lands at 10, and reach the link at 11
        d = dev(hit_us=1.)
        times = run(d, [(0, 64), (0, 64), (0, 128)])
        self.assertEqual((d.senses, d.hits), (1, 2))
        self.assertTrue(close(times[0], TR + X64))
        self.assertTrue(close(times[1], TR + 1. + X64))
        self.assertTrue(close(times[2], TR + 1. + X64 + 128 / BW))

    def test_a_buffer_being_hit_is_not_evicted(self):
        # page 0 is hit at 10 + 1/16 (after its first read is across) with hit_us = 100; pages 16 and 32 on the same bank
        # must sense into the other buffer one after the other, not evict page 0 while its hit is pending: page 32 waits
        # for page 16's transfer to free that buffer
        d, t = dev(hit_us=100.), 0.
        t = run(d, [(0, 64)], t)[0]
        times = run(d, [(0, 64), (16, 64), (32, 64)], t)
        self.assertEqual(d.senses, 3)
        self.assertTrue(close(times[0], t + 100. + X64))
        self.assertTrue(close(times[1], t + TR + X64))
        self.assertTrue(close(times[2], t + 2 * TR + 2 * X64))

    def test_buffers_must_be_one_or_two(self):
        with self.assertRaises(ValueError):
            dev(buffers=3)


class Scheduling(unittest.TestCase):
    def test_in_flight_cap_holds_requests_on_the_host(self):
        times = run(dev(mocs=2), [(i, 64) for i in range(4)])
        expected = [TR + X64, TR + 2 * X64, TR + X64 + TR + X64, TR + 2 * X64 + TR + X64]
        for t, e in zip(times, expected):
            self.assertTrue(close(t, e), (t, e))

    def test_normal_reads_wait_for_the_base_die_timer(self):
        self.assertTrue(close(run(dev(tmr=2.), [(0, 64)])[0], 2. + TR + X64))

    def test_batch_read_dispatches_at_once(self):
        self.assertTrue(close(run(dev(tmr=2.), [(0, 64)], batch=True)[0], TR + X64))

    def test_batch_read_flushes_the_held_normal_reads(self):
        d = dev(tmr=2.)
        a = d.submit(0., [0], [0], [64], [0], False)[0]
        b = d.submit(.5, [1], [0], [64], [1], True)[0]
        d.next_wake(.5, count=10 ** 9)
        done = dict((r, t) for t, r in d.take_done())
        self.assertTrue(close(done[a], .5 + TR + X64))
        self.assertTrue(close(done[b], .5 + TR + 2 * X64))

    def test_timer_holds_reads_arriving_during_its_period(self):
        d = dev(tmr=2.)
        a = d.submit(0., [0], [0], [64], [0], False)[0]
        b = d.submit(1., [1], [0], [64], [1], False)[0]
        d.next_wake(1., count=10 ** 9)
        done = dict((r, t) for t, r in d.take_done())
        self.assertTrue(close(done[a], 2. + TR + X64))
        self.assertTrue(close(done[b], 2. + TR + 2 * X64))

    def test_sram_reads_go_ahead_of_queued_transfers(self):
        d = dev()
        ids = d.submit(0., [0, 1, 2], [0] * 3, [64] * 3, [0, 1, 2], False)
        d.run_until(TR)  # all three sensed; the first transfer is on the link, two are queued
        end = d.sram_read(TR, {0: 256})
        self.assertTrue(close(end, TR + X64 + 256 / BW))
        d.next_wake(TR, count=10 ** 9)
        done = dict((r, t) for t, r in d.take_done())
        self.assertTrue(close(done[ids[0]], TR + X64))
        self.assertTrue(close(done[ids[1]], TR + 2 * X64 + 256 / BW))
        self.assertTrue(close(done[ids[2]], TR + 3 * X64 + 256 / BW))


class Roofs(unittest.TestCase):
    def test_link_bound_reads_reach_the_channel_bandwidth(self):
        n = 1000
        times = run(dev(units=128, tR=1.), [(i, 4096) for i in range(n)])
        self.assertTrue(close(max(times), 1. + n * X4K))
        self.assertTrue(close(n * 4096 / (max(times) - 1.), BW))

    def test_sense_bound_reads_reach_units_over_tR(self):
        # 16 units, 64 B reads (link time negligible), 16 x 10 pages: 10 waves of tR
        times = run(dev(channel_bw=1e9), [(i, 64) for i in range(160)])
        self.assertTrue(math.isclose(max(times), 10 * TR, rel_tol=1e-6))

    def test_read_disturb_is_charged_to_the_bank(self):
        d = dev(disturb_us=.5)
        self.assertTrue(close(run(d, [(0, 64)])[0], TR + .5 + X64))


class Determinism(unittest.TestCase):
    def trace(self, seed):
        d = dev(units=32, jitter=.1, policy="random", tmr=1., mocs=8, seed=seed)
        pages = [(7 * i * i + 3 * i) % 200 for i in range(300)]
        return run(d, [(p, 64 * (1 + i % 4)) for i, p in enumerate(pages)])

    def test_one_seed_one_trace(self):
        self.assertEqual(self.trace(5), self.trace(5))
        self.assertNotEqual(self.trace(5), self.trace(6))

    def test_jitter_stays_within_its_band(self):
        d = dev(jitter=.1, seed=3)
        for i in range(50):
            t0 = 100. * i
            t = run(d, [(16 * i, 64)], t0)[0] - t0
            self.assertTrue(.9 * TR + X64 - 1e-9 <= t <= 1.1 * TR + X64 + 1e-9)


class Waking(unittest.TestCase):
    def test_wakes_on_count_timeout_or_idle(self):
        d = dev()
        d.submit(0., [0, 1, 2], [0] * 3, [64] * 3, [0, 1, 2], False)
        self.assertTrue(close(d.next_wake(0., count=2), TR + 2 * X64))
        self.assertEqual(len(d.take_done()), 2)
        self.assertTrue(close(d.next_wake(TR + 2 * X64, count=5, timeout=.01), TR + 2 * X64 + .01))
        self.assertEqual(d.take_done(), [])
        self.assertTrue(close(d.next_wake(TR + 2 * X64 + .01, count=5), TR + 3 * X64))  # nothing left in flight
        self.assertEqual([d.r_tag[r] for _, r in d.take_done()], [2])

    def test_pending_lists_what_the_host_has_not_seen(self):
        d = dev()
        d.submit(0., [0, 1], [0, 0], [64, 64], [7, 8], False)
        d.next_wake(0., count=1)
        self.assertEqual(d.pending_tags(), [7, 8])  # complete but not handed over yet
        d.take_done()
        self.assertEqual(d.pending_tags(), [8])


def workload(make, seed):
    """A random workload touching every mechanism; returns everything the host can observe, step by step."""
    rng = random.Random(seed)
    C = rng.choice([1, 2, 3])
    d = make(channels=C, units=rng.choice([16, 32]), tR=rng.uniform(1., 20.), channel_bw=rng.choice([64., 1024., 1e5]),
             policy=rng.choice(POLICIES), jitter=rng.choice([0., .1]), tmr=rng.choice([0., 1.5]), mocs=rng.choice([2, 8, 1 << 30]),
             interleave=rng.choice([1, 2, 5]), disturb_us=rng.choice([0., .3]), pages_per_block=4, hbm_bw=100., hbm_latency=.1,
             sram_latency=.05, seed=seed)
    t, trace = 0., []
    for step in range(60):
        k = rng.randrange(0, 20)
        d.submit(t, [rng.randrange(0, 300) for _ in range(k)], [64 * rng.randrange(0, 8) for _ in range(k)],
                 [64 * rng.randrange(1, 8) for _ in range(k)], [step * 100 + i for i in range(k)], rng.random() < .5)
        if rng.random() < .2:
            t = d.hbm_read(t, rng.randrange(1, 5000))
        if rng.random() < .2:
            t = d.sram_read(t, {rng.randrange(C): 64 * rng.randrange(1, 10)})
        t = d.next_wake(t, count=rng.randrange(1, 10), timeout=rng.choice([math.inf, 3., .5]))
        trace.append((t, d.take_done(), d.pending_tags(), d.outstanding))
        t += rng.uniform(0., 2.)
    t = d.next_wake(t, count=10 ** 9)
    trace.append((t, d.take_done(), d.stats(t)))
    return trace


def workload_v2(make, seed):
    """Like workload, with the power budget, core-die channels, refresh debt, retries and host refresh calls."""
    rng = random.Random(1000 + seed)
    C, banks = rng.choice([1, 2, 4]), rng.choice([8, 16])
    dies = rng.choice([1, 2, 4])
    units = banks * dies
    tR = rng.uniform(1., 20.)
    d = make(channels=C, units=units, tR=tR, channel_bw=rng.choice([64., 1024., 1e5]), policy=rng.choice(POLICIES),
             jitter=rng.choice([0., .1]), tmr=rng.choice([0., 1.5]), mocs=rng.choice([2, 8, 1 << 30]),
             interleave=rng.choice([1, 2, 5]), pages_per_block=rng.choice([4, 64]), hbm_bw=100., hbm_latency=.1,
             sram_latency=.05, seed=seed, banks_per_die=banks, channels_per_stack=rng.choice([0, 1] + ([2] if C % 2 == 0 else [])),
             power_w=rng.choice([0., .5, 3.]), power_depth=rng.choice([0., 2., 40.]), e_sense=rng.choice([.2, 1.]),
             e_prog=rng.choice([0., 3.3]), coredie=rng.choice([0] + [k for k in (1, 2) if dies % k == 0]),
             coredie_bw=rng.choice([100., 2000.]), refresh_x=rng.choice([0., 8., 200.]), t_prog=rng.choice([20., 75.]),
             scope=rng.choice(["die", "plane"]), d_max=rng.choice([math.inf, 1., 3.]), retry_p=rng.choice([0., .3, 1.]),
             refresh_policy=rng.choice(["default", "host"]))
    t, trace = 0., []
    for step in range(60):
        k = rng.randrange(0, 24)
        d.submit(t, [rng.randrange(0, 400) for _ in range(k)], [64 * rng.randrange(0, 8) for _ in range(k)],
                 [64 * rng.randrange(1, 8) for _ in range(k)], [step * 100 + i for i in range(k)], rng.random() < .5)
        if rng.random() < .2:
            t = d.hbm_read(t, rng.randrange(1, 5000))
        if rng.random() < .2:
            t = d.sram_read(t, {rng.randrange(C): 64 * rng.randrange(1, 10)})
        if rng.random() < .3:
            d.refresh(t, rng.randrange(C * dies), rng.randrange(0, 3))
        if rng.random() < .3:
            trace.append(("debt", d.debt_per_die(t), d.settle()))
        t = d.next_wake(t, count=rng.randrange(1, 10), timeout=rng.choice([math.inf, 3., .5]))
        trace.append((t, d.take_done(), d.pending_tags(), d.outstanding))
        t += rng.uniform(0., 2.)
    t = d.next_wake(t, count=10 ** 9)
    trace.append((t, d.take_done(), d.stats(t), d.settle()))
    return trace


class Power(unittest.TestCase):
    """Power budget: a token bucket per stack, refilled at P, depth D; a sense draws E_sense first."""

    def test_senses_beyond_the_burst_start_e_over_p_apart(self):
        # P 1 W = 1 uJ/us, E_sense 2 uJ, depth 4 uJ (full at t = 0): two senses at 0, then one every 2 us
        d = dev(power_w=1., power_depth=4., e_sense=2.)
        times = run(d, [(i, 64) for i in range(4)], batch=True)
        expected = [TR + X64, TR + 2 * X64, 2. + TR + X64, 4. + TR + X64]
        for t, e in zip(times, expected):
            self.assertTrue(close(t, e), (t, e))
        st = d.stats(max(times))
        self.assertTrue(close(st["energy_uj"], 8.))
        self.assertTrue(close(st["power_wait_us"], 2. + 4.))

    def test_each_stack_has_its_own_budget(self):
        # one sense's worth of tokens per stack: pages 0 and 1 are on channels 0 and 1
        apart = run(dev(channels=2, channels_per_stack=1, power_w=1., power_depth=2., e_sense=2.), [(0, 64), (1, 64)], batch=True)
        shared = run(dev(channels=2, channels_per_stack=2, power_w=1., power_depth=2., e_sense=2.), [(0, 64), (1, 64)], batch=True)
        self.assertTrue(close(apart[0], TR + X64) and close(apart[1], TR + X64))
        self.assertTrue(close(shared[0], TR + X64) and close(shared[1], 2. + TR + X64))

    def test_hits_draw_no_power(self):
        d = dev(power_w=1., power_depth=2., e_sense=2.)
        times = run(d, [(0, 64), (0, 64)], batch=True)
        self.assertTrue(close(times[1], TR + 2 * X64))
        self.assertTrue(close(d.stats(times[1])["energy_uj"], 2.))

    def test_sustained_rate_is_p_over_e_sense(self):
        # 64 units, tR 4: the structure allows 16 senses/us; P 2 W at 0.25 uJ allows 8/us, so power sets the rate
        d = dev(units=64, tR=4., channel_bw=1e9, power_w=2., power_depth=80., e_sense=.25)
        n = 6400
        times = sorted(run(d, [(i, 64) for i in range(n)], batch=True))
        burst = int(80. / .25)
        rate = (n - 2 * burst) / (times[-1] - times[2 * burst])
        self.assertTrue(math.isclose(rate, 2. / .25, rel_tol=.01), rate)


class CoreDie(unittest.TestCase):
    """Core-die channels (OCP §4.6): every sense moves the whole 4 KiB page into the base die."""

    def test_two_dies_on_one_channel_queue_their_pages(self):
        # coredie_bw 1024 B/us: a page takes 4 us. Banks 0 and 1 of die 0 sense at 0..10, cross at 10..14 and 14..18
        times = run(dev(coredie=1, coredie_bw=BW), [(0, 64), (1, 64)])
        self.assertTrue(close(times[0], TR + 4. + X64))
        self.assertTrue(close(times[1], TR + 8. + X64))

    def test_the_unit_holds_its_page_until_it_is_across(self):
        # page 17 is bank 1 again: its sense starts when page 1 has crossed (18), crosses at 28..32
        times = run(dev(coredie=1, coredie_bw=BW), [(0, 64), (1, 64), (17, 64)])
        self.assertTrue(close(times[2], 18. + TR + 4. + X64), times)

    def test_dies_on_separate_channels_cross_in_parallel(self):
        # 2 dies (32 units); page 16 is on die 1
        apart = run(dev(units=32, coredie=2, coredie_bw=BW), [(0, 64), (16, 64)])
        shared = run(dev(units=32, coredie=1, coredie_bw=BW), [(0, 64), (16, 64)])
        self.assertTrue(close(apart[0], TR + 4. + X64) and close(apart[1], TR + 4. + 2 * X64))
        self.assertTrue(close(shared[0], TR + 4. + X64) and close(shared[1], TR + 8. + X64))

    def test_a_64_byte_read_moves_a_whole_page_inside(self):
        d = dev(coredie=1, coredie_bw=BW)
        t = run(d, [(0, 64)])[0]
        self.assertTrue(close(d.stats(t)["coredie_util"] * t, 4.))


# refresh tests: pages_per_block 4 and X 4, so every sense adds one page of debt; t_prog 100, so an operation takes 110
RF = dict(pages_per_block=4, refresh_x=4., t_prog=100.)
OP = TR + 100.


def run_at(d, reads):
    """[(t, page)] submitted at their times (64 B, batch), run to the end; completion times in order."""
    ids = []
    for t, page in reads:
        ids += d.submit(t, [page], [0], [64], [len(ids)], True)
    d.next_wake(reads[-1][0], count=10 ** 9)
    done = dict((r, time) for time, r in d.take_done())
    return [done[r] for r in ids]


class Refresh(unittest.TestCase):
    """Host-managed refresh: debt per plane, operations hold the die (or plane) for tR + t_prog."""

    def test_default_policy_pays_when_the_die_is_idle_and_holds_the_die(self):
        d = dev(**RF)
        # page 0 senses at 0..10 (debt 1); the die is idle at 10, so an operation holds it until 10 + 110 = 120;
        # page 1 (same die, another bank) arrives at 20 and senses at 120..130
        times = run_at(d, [(0., 0), (20., 1)])
        self.assertTrue(close(times[0], TR + X64))
        self.assertTrue(close(times[1], TR + OP + TR + X64), times)
        self.assertEqual(d.stats(times[1])["refresh_ops"], 2)  # page 1's debt: the idle die pays it at 130..240
        self.assertTrue(close(d.settle(), OP))  # an operation retires its pages when it ends: one page still owed
        self.assertEqual(d.debt_per_die(239.), [1.])
        self.assertEqual(d.debt_per_die(240.), [0.])
        self.assertEqual(d.settle(), 0.)

    def test_other_dies_are_not_held(self):
        times = run_at(dev(units=32, **RF), [(0., 0), (20., 16)])  # page 16 is on die 1
        self.assertTrue(close(times[1], 20. + TR + X64))

    def test_plane_scope_holds_only_the_plane(self):
        d = dev(scope="plane", **RF)
        times = run_at(d, [(0., 0), (20., 1), (20., 16)])  # page 1: bank 1, free; page 16: bank 0, held until 120
        self.assertTrue(close(times[1], 20. + TR + X64))
        self.assertTrue(close(times[2], TR + OP + TR + X64), times)

    def test_host_policy_pays_only_when_asked_and_settles_the_rest(self):
        d = dev(refresh_policy="host", **RF)
        times = run_at(d, [(0., 0), (20., 1)])
        self.assertTrue(close(times[1], 20. + TR + X64))
        self.assertEqual(d.debt_per_die(40.), [1.])
        self.assertTrue(close(d.settle(), OP))
        d.refresh(40., 0, 1)  # one multi-plane operation retires a page on both banks: 40..150
        later = run_at(d, [(50., 2)])
        self.assertTrue(close(later[0], 40. + OP + TR + X64), later)
        self.assertEqual(d.debt_per_die(later[0]), [1.])  # page 2's own sense
        st = d.stats(later[0])
        self.assertEqual(st["refresh_ops"], 1)
        self.assertTrue(close(st["refresh_pages"], 2.))

    def test_debt_beyond_d_max_is_paid_at_once(self):
        # three reads queued on bank 0; d_max 1.5. The second sense (10..20) takes the debt to 2: an operation waits
        # for it and runs 20..130 (debt 1); the third sense runs 130..140 (debt 2), another operation 140..250 (debt 1),
        # then the idle die pays the last page 250..360.
        d = dev(d_max=1.5, **RF)
        times = run(d, [(0, 64), (16, 64), (32, 64)], batch=True)
        self.assertTrue(close(times[0], TR + X64))
        self.assertTrue(close(times[1], 2 * TR + X64))
        self.assertTrue(close(times[2], 2 * TR + OP + TR + X64), times)
        d.run_until(1e6)
        st = d.stats(1e6)
        self.assertEqual(st["refresh_ops"], 3)
        self.assertEqual(st["max_debt"], 0.)

    def test_host_policy_is_forced_beyond_twice_d_max(self):
        d = dev(refresh_policy="host", d_max=1., **RF)  # forced above 2
        times = run(d, [(0, 64), (16, 64), (32, 64), (48, 64)], batch=True)
        # senses 0..10 (1), 10..20 (2), 20..30 (3 > 2: forced, 30..140, debt 2), 140..150 (3: forced again)
        self.assertTrue(close(times[2], 3 * TR + X64))
        self.assertTrue(close(times[3], 3 * TR + OP + TR + X64), times)

    def test_settle_is_the_largest_debt_times_an_operation(self):
        d = dev(refresh_policy="host", **RF)
        run(d, [(0, 64), (16, 64), (32, 64)], batch=True)
        self.assertTrue(close(d.settle(), 3 * OP))
        self.assertTrue(close(d.stats(1.)["settle_us"], 3 * OP))

    def test_refresh_draws_read_and_program_energy(self):
        d = dev(power_w=1., power_depth=1e3, e_sense=2., e_prog=30., **RF)
        times = run_at(d, [(0., 0)])
        d.run_until(1e6)
        self.assertTrue(close(d.stats(1e6)["energy_uj"], 2. + 32.))

    def test_bad_refresh_requests_are_refused(self):
        d = dev(refresh_policy="host", **RF)
        for die, n in ((1, 1), (-1, 1), (0, -1)):
            with self.assertRaises(DeviceError):
                d.refresh(0., die, n)


class Retry(unittest.TestCase):
    """Read retry: beyond d_max a sense is repeated with probability retry_p."""

    def test_a_retry_is_one_more_sense(self):
        # X such that a sense adds 0.75 pages; d_max 0.5, host policy (forced above 1). The second read on bank 0
        # finds 0.75 > 0.5 and is retried: it senses 20..40.
        d = dev(refresh_policy="host", d_max=.5, retry_p=1., pages_per_block=3, refresh_x=4., t_prog=100.)
        times = run_at(d, [(0., 0), (20., 16)])
        self.assertTrue(close(times[1], 20. + 2 * TR + X64), times)
        st = d.stats(times[1])
        self.assertEqual((st["retries"], st["senses"]), (1, 3))

    def test_retry_rate_is_p(self):
        # one retry decision per bank: 64 channels x 16 banks; each bank's second sense starts with debt 0.51 > 0.5
        # (plane scope: a forced payment on one bank leaves the other banks' debt alone)
        d = dev(channels=64, scope="plane", refresh_policy="host", d_max=.5, retry_p=.5, pages_per_block=51, refresh_x=100.,
                t_prog=1., seed=11)
        pages = [i for i in range(64 * 16)] + [64 * 16 + i for i in range(64 * 16)]
        run(d, [(p, 64) for p in pages], batch=True)
        rate = d.stats(1.)["retries"] / (64 * 16)
        self.assertTrue(abs(rate - .5) < .06, rate)


@unittest.skipIf(fastdevice.load() is None, "no C compiler")
class Equivalence(unittest.TestCase):
    def test_c_and_python_give_identical_traces(self):
        for seed in range(40):
            self.assertEqual(workload(fastdevice.CDevice, seed), workload(Device, seed), seed)

    def test_c_and_python_agree_without_hits_and_with_one_buffer(self):
        import functools
        for seed in range(40):
            kw = {"hits": seed % 2 == 0, "buffers": 1 + seed % 3 // 2, "hit_us": (0., .05, 2.)[seed % 5 % 3]}
            self.assertEqual(workload(functools.partial(fastdevice.CDevice, **kw), seed),
                             workload(functools.partial(Device, **kw), seed), seed)

    def test_c_and_python_agree_with_every_v2_mechanism_on(self):
        for seed in range(60):
            self.assertEqual(workload_v2(fastdevice.CDevice, seed), workload_v2(Device, seed), seed)


def c_variant(cls):
    def setUp(self):
        IMPL["Device"] = fastdevice.CDevice

    def tearDown(self):
        IMPL["Device"] = Device

    return unittest.skipIf(fastdevice.load() is None, "no C compiler")(type(cls.__name__ + "C", (cls,), {"setUp": setUp, "tearDown": tearDown}))


for _cls in (SingleReads, Parallelism, Buffers, Scheduling, Roofs, Determinism, Waking, Power, CoreDie, Refresh, Retry):
    globals()[_cls.__name__ + "C"] = c_variant(_cls)


if __name__ == "__main__":
    unittest.main()
