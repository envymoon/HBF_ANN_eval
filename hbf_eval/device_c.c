/* Device model in C: the same model as device.py (the reference), event for event and bit for bit.
 *
 * Every rule, every event and every floating-point operation follows device.py in the same order, so the two give
 * identical results (tests/test_device.py runs the known-answer tests on both and compares random workloads trace by
 * trace). Compile without fast-math and without FP contraction (fastdevice.py passes -ffp-contract=off).
 * Time in microseconds, sizes in bytes, bandwidth in bytes per microsecond, energy in microjoules.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { ARRIVE = 0, TIMER = 1, SENSED = 2, XFERRED = 3, CDCDONE = 4, POWER = 5, REFDONE = 6, HITREADY = 7 };
enum { IDLE = 0, POWER_WAIT = 1, SENSING = 2, CDC = 3, REFRESH = 4 };
enum { NONE = 0, PENDING = 1, PWAIT = 2, RUNNING = 3 };
enum { LRU = 0, FIFO = 1, RANDOM = 2 };
#define PAGE 4096
#define ALIGN 64

typedef struct { int64_t *d; int64_t head, len, cap; } Ring;
typedef struct { int64_t *d; int64_t len, cap; } Vec;
typedef struct { double t; int64_t seq; int64_t a, b; int32_t kind; } Event;

typedef struct {
    int64_t C, U, g, mocs, ppb, policy, banks, dies, cps, S, ncdc, K, plane_scope, host_refresh, hits_on, nbuf;
    double tR, bw, jitter, tmr, disturb, hbm_bw, hbm_latency, sram_latency;
    double P, pdepth, e_sense, e_prog, cbw, X, t_prog, d_max, retry_p, dinc, thr, hit_us;
    uint64_t rng;
    Event *heap; int64_t nheap, capheap, seq;
    double now;
    /* requests */
    int64_t nreq, capreq, first_open;
    int32_t *r_ch, *r_cu, *r_bytes; int64_t *r_page, *r_tag; uint8_t *r_batch, *r_open; int8_t *r_buf;
    /* channels */
    int64_t *inflight, *timer_gen; uint8_t *timer_on, *link_busy; double *link_free;
    Ring *waiting, *xq; Vec *hold; Vec scratch;
    /* units */
    Ring *queue; int8_t *ustate; uint8_t *blocked;
    int64_t *buf_page; uint8_t *buf_valid; int32_t *buf_lock; double *buf_stamp;
    int8_t *sbuf, *pend_v, *pend_retry; int64_t *pend_r; double *pend_since;
    /* refresh */
    double *debt, *oret, *opages; uint8_t *res; int8_t *oslot; int64_t *want; double *dmax;
    /* power, per stack (queue entries: 2 * ident + kind) */
    double *pL, *ptL; Ring *pq; uint8_t *ppending;
    /* core-die channels */
    Ring *cq; uint8_t *cbusy; double *ctime;
    double hbm_free;
    int64_t outstanding;
    /* completions not yet handed over */
    double *done_t; int64_t *done_r; int64_t ndone, capdone;
    /* statistics */
    int64_t *unit_senses; double *unit_busy; int64_t hits, senses;
    double *link_time; int64_t *link_bytes;
    int32_t *block_reads; int64_t nblocks;
    int64_t sram_bytes, hbm_bytes, retries, ref_ops;
    double ref_pages, ref_unit_us, energy, power_wait;
    int failed; /* out of memory */
} Dev;

static void *grow(void *p, int64_t n, size_t size, int *failed) {
    void *q = realloc(p, (size_t)n * size);
    if (!q && n) *failed = 1;
    return q ? q : p;
}

/* ---- splitmix64, the same as device.SplitMix ---- */
static uint64_t rng_next(Dev *d) {
    uint64_t z = (d->rng += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double rng_random(Dev *d) { return (double)(rng_next(d) >> 11) * (1.0 / 9007199254740992.0); }

/* ---- containers ---- */
static void ring_push(Dev *d, Ring *q, int64_t v) {
    if (q->len == q->cap) {
        int64_t cap = q->cap ? 2 * q->cap : 8;
        int64_t *n = malloc((size_t)cap * sizeof(int64_t));
        if (!n) { d->failed = 1; return; }
        for (int64_t i = 0; i < q->len; i++) n[i] = q->d[(q->head + i) % q->cap];
        free(q->d); q->d = n; q->head = 0; q->cap = cap;
    }
    q->d[(q->head + q->len) % q->cap] = v;
    q->len++;
}
static int64_t ring_pop(Ring *q) { int64_t v = q->d[q->head]; q->head = (q->head + 1) % q->cap; q->len--; return v; }
static int64_t ring_front(Ring *q) { return q->d[q->head]; }
static void vec_push(Dev *d, Vec *v, int64_t x) {
    if (v->len == v->cap) { v->cap = v->cap ? 2 * v->cap : 8; v->d = grow(v->d, v->cap, sizeof(int64_t), &d->failed); }
    if (!d->failed) v->d[v->len++] = x;
}

static int ev_less(const Event *a, const Event *b) { return a->t < b->t || (a->t == b->t && a->seq < b->seq); }
static void push(Dev *d, double t, int32_t kind, int64_t a, int64_t b) {
    if (d->nheap == d->capheap) {
        d->capheap = d->capheap ? 2 * d->capheap : 64;
        d->heap = grow(d->heap, d->capheap, sizeof(Event), &d->failed);
        if (d->failed) return;
    }
    d->seq++;
    Event e = {t, d->seq, a, b, kind};
    int64_t i = d->nheap++;
    while (i > 0) {
        int64_t p = (i - 1) / 2;
        if (!ev_less(&e, &d->heap[p])) break;
        d->heap[i] = d->heap[p];
        i = p;
    }
    d->heap[i] = e;
}
static Event pop(Dev *d) {
    Event top = d->heap[0], last = d->heap[--d->nheap];
    int64_t i = 0, n = d->nheap;
    for (;;) {
        int64_t l = 2 * i + 1, r = l + 1, m = i;
        const Event *best = &last;
        if (l < n && ev_less(&d->heap[l], best)) { m = l; best = &d->heap[l]; }
        if (r < n && ev_less(&d->heap[r], best)) { m = r; }
        if (m == i) break;
        d->heap[i] = d->heap[m];
        i = m;
    }
    if (n) d->heap[i] = last;
    return top;
}

/* ---- the model (device.py, method by method) ---- */
static void link_next(Dev *d, int64_t c, double t);
static void unit_next(Dev *d, int64_t cu, double t);
static void settled(Dev *d, int64_t cu, double t);
static void die_kick(Dev *d, int64_t gd, double t);
static void op_try(Dev *d, int64_t o, double t);
static void op_start(Dev *d, int64_t o, double t);

static int find(Dev *d, int64_t cu, int64_t page) {
    if (!d->hits_on) return -1;
    if (d->buf_page[2 * cu] == page && d->buf_valid[2 * cu]) return 0;
    if (d->nbuf > 1 && d->buf_page[2 * cu + 1] == page && d->buf_valid[2 * cu + 1]) return 1;
    return -1;
}

static void enqueue(Dev *d, int64_t r, double t) {
    int64_t c = d->r_ch[r];
    ring_push(d, &d->xq[c], r);
    link_next(d, c, t);
}

static void ready(Dev *d, int64_t r, int64_t cu, int v, double t) {
    d->buf_lock[2 * cu + v] += 1;
    d->r_buf[r] = (int8_t)v;
    enqueue(d, r, t);
}

static void hit(Dev *d, int64_t cu, int v, int64_t r, double t) {
    d->hits += 1;
    if (d->policy == LRU) d->buf_stamp[2 * cu + v] = t;
    if (d->hit_us > 0) {  /* the base-die buffer's own access time; the buffer is held meanwhile */
        d->buf_lock[2 * cu + v] += 1;
        d->r_buf[r] = (int8_t)v;
        push(d, t + d->hit_us, HITREADY, r, 0);
    } else {
        ready(d, r, cu, v, t);
    }
}

static int victim(Dev *d, int64_t cu) {
    int free_[2], nfree = 0;
    for (int v = 0; v < d->nbuf; v++)
        if (d->buf_lock[2 * cu + v] == 0 && (d->buf_valid[2 * cu + v] || d->buf_page[2 * cu + v] == -1)) free_[nfree++] = v;
    if (!nfree) return -1;
    for (int i = 0; i < nfree; i++)
        if (d->buf_page[2 * cu + free_[i]] == -1) return free_[i];
    if (nfree == 1) return free_[0];
    if (d->policy == RANDOM) return (int)(rng_next(d) >> 63);
    return d->buf_stamp[2 * cu] <= d->buf_stamp[2 * cu + 1] ? 0 : 1;
}

static void count_block(Dev *d, int64_t cu, int64_t block) {
    if (block >= d->nblocks) {
        int64_t nb = 2 * block + 2, CU = d->C * d->U;
        int32_t *n = calloc((size_t)(CU * nb), sizeof(int32_t));
        if (!n) { d->failed = 1; return; }
        for (int64_t u = 0; u < CU; u++)
            for (int64_t b = 0; b < d->nblocks; b++) n[u * nb + b] = d->block_reads[u * d->nblocks + b];
        free(d->block_reads); d->block_reads = n; d->nblocks = nb;
    }
    d->block_reads[cu * d->nblocks + block] += 1;
}

/* ---- power budget (device.Device._power_request / _power_event) ---- */
static double power_need(Dev *d, int64_t entry) {
    int64_t ident = entry >> 1;
    if ((entry & 1) == 0) return d->e_sense * (double)(1 + d->pend_retry[ident]);
    return (d->e_sense + d->e_prog) * d->opages[ident];
}

static int power_request(Dev *d, int64_t s, int kind, int64_t ident, double e, double t) {
    if (d->P <= 0) { d->energy += e; return 1; }
    Ring *q = &d->pq[s];
    if (!q->len && !d->ppending[s]) {
        double a = d->pL[s] + d->P * (t - d->ptL[s]);
        double level = a < d->pdepth ? a : d->pdepth;
        if (level >= e) { d->pL[s] = level - e; d->ptL[s] = t; d->energy += e; return 1; }
        ring_push(d, q, 2 * ident + kind);
        d->ppending[s] = 1;
        push(d, t + (e - level) / d->P, POWER, s, 0);
        return 0;
    }
    ring_push(d, q, 2 * ident + kind);
    return 0;
}

static void start_sense(Dev *d, int64_t cu, int64_t r, int v, int retry, double t);

static void power_event(Dev *d, int64_t s, double t) {
    Ring *q = &d->pq[s];
    int first = 1;
    while (q->len) {
        int64_t entry = ring_front(q);
        double e = power_need(d, entry);
        double a = d->pL[s] + d->P * (t - d->ptL[s]);
        double level = a < d->pdepth ? a : d->pdepth;
        if (!first && level < e) { push(d, t + (e - level) / d->P, POWER, s, 0); return; }
        first = 0;
        ring_pop(q);
        d->pL[s] = level - e; d->ptL[s] = t;
        d->energy += e;
        int64_t ident = entry >> 1;
        if ((entry & 1) == 0) {
            d->power_wait += t - d->pend_since[ident];
            start_sense(d, ident, d->pend_r[ident], d->pend_v[ident], d->pend_retry[ident], t);
        } else {
            op_start(d, ident, t);
        }
    }
    d->ppending[s] = 0;
}

/* ---- refresh ---- */
static void op_new(Dev *d, int64_t cu, double t) {
    d->oslot[cu] = PENDING;
    d->res[cu] = 1;
    op_try(d, cu, t);
}

static void urgent(Dev *d, int64_t cu, double t) {
    if (d->plane_scope) {
        if (d->oslot[cu] == NONE) op_new(d, cu, t);
    } else {
        int64_t gd = cu / d->banks;
        if (d->oslot[gd] == NONE && d->want[gd] == 0) { d->want[gd] += 1; die_kick(d, gd, t); }
    }
}

static void die_kick(Dev *d, int64_t gd, double t) {
    if (!d->plane_scope) {
        if (d->oslot[gd] == NONE && d->want[gd] > 0) {
            d->want[gd] -= 1;
            d->oslot[gd] = PENDING;
            for (int64_t u = gd * d->banks; u < (gd + 1) * d->banks; u++) d->res[u] = 1;
            op_try(d, gd, t);
        }
        return;
    }
    while (d->want[gd] > 0) {
        int64_t best = -1;
        for (int64_t u = gd * d->banks; u < (gd + 1) * d->banks; u++)
            if (d->oslot[u] == NONE && (best < 0 || d->debt[u] > d->debt[best])) best = u;
        if (best < 0) return;
        d->want[gd] -= 1;
        op_new(d, best, t);
    }
}

static void op_range(Dev *d, int64_t o, int64_t *lo, int64_t *hi) {
    if (d->plane_scope) { *lo = o; *hi = o + 1; }
    else { *lo = o * d->banks; *hi = (o + 1) * d->banks; }
}

static void op_try(Dev *d, int64_t o, double t) {
    if (d->oslot[o] != PENDING) return;
    int64_t lo, hi;
    op_range(d, o, &lo, &hi);
    for (int64_t u = lo; u < hi; u++) if (d->ustate[u] != IDLE) return;
    double pages = 0.;
    for (int64_t u = lo; u < hi; u++) {
        d->oret[u] = d->debt[u] < 1. ? d->debt[u] : 1.;
        pages += d->oret[u];
    }
    d->oslot[o] = PWAIT;
    d->opages[o] = pages;
    if (power_request(d, lo / d->U / d->cps, 1, o, (d->e_sense + d->e_prog) * pages, t)) op_start(d, o, t);
}

static void op_start(Dev *d, int64_t o, double t) {
    d->oslot[o] = RUNNING;
    int64_t lo, hi;
    op_range(d, o, &lo, &hi);
    for (int64_t u = lo; u < hi; u++) d->ustate[u] = REFRESH;
    double duration = d->tR + d->t_prog;
    d->ref_ops += 1;
    d->ref_pages += d->opages[o];
    d->ref_unit_us += duration * (double)(hi - lo);
    push(d, t + duration, REFDONE, o, 0);
}

static void refdone(Dev *d, int64_t o, double t) {
    int64_t lo, hi;
    op_range(d, o, &lo, &hi);
    for (int64_t u = lo; u < hi; u++) {
        d->debt[u] -= d->oret[u];
        d->oret[u] = 0.;
        d->ustate[u] = IDLE;
        d->res[u] = 0;
    }
    d->oslot[o] = NONE;
    int64_t gd = lo / d->banks;
    double top = 0.;
    for (int64_t u = gd * d->banks; u < (gd + 1) * d->banks; u++) if (d->debt[u] > top) top = d->debt[u];
    d->dmax[gd] = top;
    if (d->plane_scope) {
        if (d->debt[o] > d->thr) op_new(d, o, t);
    } else if (top > d->thr && d->want[gd] == 0) {
        d->want[gd] += 1;
    }
    die_kick(d, gd, t);
    for (int64_t u = lo; u < hi; u++) unit_next(d, u, t);
    for (int64_t u = lo; u < hi; u++) settled(d, u, t);
}

static void settled(Dev *d, int64_t cu, double t) {
    if (d->res[cu]) { op_try(d, d->plane_scope ? cu : cu / d->banks, t); return; }
    if (d->X <= 0 || d->host_refresh || d->ustate[cu] != IDLE || d->queue[cu].len) return;
    if (d->plane_scope) {
        if (d->oslot[cu] == NONE && d->debt[cu] >= 1.) op_new(d, cu, t);
        return;
    }
    int64_t gd = cu / d->banks;
    if (d->oslot[gd] != NONE || d->want[gd] || d->dmax[gd] < 1.) return;
    for (int64_t u = gd * d->banks; u < (gd + 1) * d->banks; u++)
        if (d->ustate[u] != IDLE || d->queue[u].len) return;
    d->want[gd] += 1;
    die_kick(d, gd, t);
}

/* ---- senses ---- */
static void start_sense(Dev *d, int64_t cu, int64_t r, int v, int retry, double t) {
    double duration = d->jitter != 0. ? d->tR * (1. + d->jitter * (2. * rng_random(d) - 1.)) : d->tR;
    if (retry) duration += d->jitter != 0. ? d->tR * (1. + d->jitter * (2. * rng_random(d) - 1.)) : d->tR;
    duration += d->disturb;
    d->ustate[cu] = SENSING;
    d->sbuf[cu] = (int8_t)v;
    d->unit_senses[cu] += 1 + retry;
    d->unit_busy[cu] += duration;
    d->senses += 1 + retry;
    d->retries += retry;
    count_block(d, cu, d->r_page[r] / d->U / d->ppb);
    if (d->X > 0) {
        d->debt[cu] += (double)(1 + retry) * d->dinc;
        int64_t gd = cu / d->banks;
        if (d->debt[cu] > d->dmax[gd]) d->dmax[gd] = d->debt[cu];
        if (d->debt[cu] > d->thr) urgent(d, cu, t);
    }
    push(d, t + duration, SENSED, r, v);
}

static void unit_next(Dev *d, int64_t cu, double t) {
    if (d->ustate[cu] != IDLE || d->res[cu]) return;
    Ring *q = &d->queue[cu];
    while (q->len) {
        int64_t r = ring_front(q);
        int v = find(d, cu, d->r_page[r]);
        if (v >= 0) { ring_pop(q); hit(d, cu, v, r, t); continue; }
        v = victim(d, cu);
        if (v < 0) { d->blocked[cu] = 1; return; }
        ring_pop(q);
        d->buf_page[2 * cu + v] = d->r_page[r];
        d->buf_valid[2 * cu + v] = 0;
        int retry = 0;
        if (d->retry_p > 0 && d->debt[cu] > d->d_max) retry = rng_random(d) < d->retry_p ? 1 : 0;
        if (power_request(d, cu / d->U / d->cps, 0, cu, d->e_sense * (double)(1 + retry), t)) {
            start_sense(d, cu, r, v, retry, t);
        } else {
            d->ustate[cu] = POWER_WAIT;
            d->pend_r[cu] = r; d->pend_v[cu] = (int8_t)v; d->pend_retry[cu] = (int8_t)retry; d->pend_since[cu] = t;
        }
        return;
    }
}

static void dispatch(Dev *d, int64_t r, double t) {
    int64_t cu = d->r_cu[r];
    int v = find(d, cu, d->r_page[r]);
    if (v >= 0) { hit(d, cu, v, r, t); return; }
    ring_push(d, &d->queue[cu], r);
    unit_next(d, cu, t);
}

static void flush(Dev *d, int64_t c, double t) {
    Vec *h = &d->hold[c];
    d->scratch.len = 0;
    for (int64_t i = 0; i < h->len; i++) vec_push(d, &d->scratch, h->d[i]);
    h->len = 0;
    d->timer_on[c] = 0;
    d->timer_gen[c] += 1;
    for (int64_t i = 0; i < d->scratch.len; i++) dispatch(d, d->scratch.d[i], t);
}

static void admit(Dev *d, int64_t r, double t) {
    int64_t c = d->r_ch[r];
    d->inflight[c] += 1;
    vec_push(d, &d->hold[c], r);
    if (d->r_batch[r] || d->tmr <= 0) {
        flush(d, c, t);
    } else if (!d->timer_on[c]) {
        d->timer_on[c] = 1;
        push(d, t + d->tmr, TIMER, c, d->timer_gen[c]);
    }
}

static void arrive(Dev *d, int64_t r, double t) {
    int64_t c = d->r_ch[r];
    if (d->inflight[c] >= d->mocs) ring_push(d, &d->waiting[c], r);
    else admit(d, r, t);
}

static void land(Dev *d, int64_t cu, int64_t r, int v, double t) {
    d->buf_valid[2 * cu + v] = 1;
    d->buf_stamp[2 * cu + v] = t;
    d->ustate[cu] = IDLE;
    ready(d, r, cu, v, t);
    unit_next(d, cu, t);
    settled(d, cu, t);
}

static void cdc_next(Dev *d, int64_t k, double t) {
    if (d->cbusy[k] || !d->cq[k].len) return;
    int64_t r = ring_pop(&d->cq[k]);
    double duration = 4096. / d->cbw;
    d->cbusy[k] = 1;
    d->ctime[k] += duration;
    push(d, t + duration, CDCDONE, k, r);
}

static void sensed(Dev *d, int64_t r, int v, double t) {
    int64_t cu = d->r_cu[r];
    if (d->ncdc) {
        d->ustate[cu] = CDC;
        int64_t k = (cu / d->U) * d->ncdc + (cu % d->U) / d->banks * d->ncdc / d->dies;
        ring_push(d, &d->cq[k], r);
        cdc_next(d, k, t);
        return;
    }
    land(d, cu, r, v, t);
}

static void cdc_done(Dev *d, int64_t k, int64_t r, double t) {
    d->cbusy[k] = 0;
    int64_t cu = d->r_cu[r];
    land(d, cu, r, d->sbuf[cu], t);
    cdc_next(d, k, t);
}

static void link_next(Dev *d, int64_t c, double t) {
    if (d->link_busy[c] || !d->xq[c].len) return;
    int64_t r = ring_pop(&d->xq[c]);
    double begin = t > d->link_free[c] ? t : d->link_free[c];
    double duration = (double)d->r_bytes[r] / d->bw;
    d->link_free[c] = begin + duration;
    d->link_busy[c] = 1;
    d->link_time[c] += duration;
    d->link_bytes[c] += d->r_bytes[r];
    push(d, begin + duration, XFERRED, c, r);
}

static void xferred(Dev *d, int64_t c, int64_t r, double t) {
    d->link_busy[c] = 0;
    if (d->ndone == d->capdone) {
        d->capdone = d->capdone ? 2 * d->capdone : 256;
        d->done_t = grow(d->done_t, d->capdone, sizeof(double), &d->failed);
        d->done_r = grow(d->done_r, d->capdone, sizeof(int64_t), &d->failed);
        if (d->failed) return;
    }
    d->done_t[d->ndone] = t;
    d->done_r[d->ndone++] = r;
    d->outstanding -= 1;
    int64_t cu = d->r_cu[r];
    int v = d->r_buf[r];
    d->r_buf[r] = -3;
    d->buf_lock[2 * cu + v] -= 1;
    if (d->blocked[cu]) { d->blocked[cu] = 0; unit_next(d, cu, t); settled(d, cu, t); }
    d->inflight[c] -= 1;
    if (d->waiting[c].len) admit(d, ring_pop(&d->waiting[c]), t);
    link_next(d, c, t);
}

/* ---- the interface (fastdevice.py) ---- */
void dev_free(Dev *d);

/* ip: C, U, policy, mocs, interleave, pages_per_block, banks_per_die, channels_per_stack (0 = all), coredie,
 *     plane_scope, host_refresh, hits (0 = buffers never hit), buffers (1 or 2)
 * dp: tR, channel_bw, jitter, tmr, disturb_us, hbm_bw, hbm_latency, sram_latency, power_w, power_depth, e_sense, e_prog,
 *     coredie_bw, refresh_x, t_prog, d_max, retry_p, hit_us */
Dev *dev_new(const int64_t *ip, const double *dp, uint64_t seed) {
    Dev *d = calloc(1, sizeof(Dev));
    if (!d) return NULL;
    d->C = ip[0]; d->U = ip[1]; d->policy = ip[2]; d->mocs = ip[3]; d->g = ip[4]; d->ppb = ip[5]; d->banks = ip[6];
    d->cps = ip[7] ? ip[7] : ip[0]; d->ncdc = ip[8]; d->plane_scope = ip[9]; d->host_refresh = ip[10];
    d->hits_on = ip[11]; d->nbuf = ip[12];
    d->tR = dp[0]; d->bw = dp[1]; d->jitter = dp[2]; d->tmr = dp[3]; d->disturb = dp[4]; d->hbm_bw = dp[5];
    d->hbm_latency = dp[6]; d->sram_latency = dp[7]; d->P = dp[8]; d->pdepth = dp[9]; d->e_sense = dp[10];
    d->e_prog = dp[11]; d->cbw = dp[12]; d->X = dp[13]; d->t_prog = dp[14]; d->d_max = dp[15]; d->retry_p = dp[16];
    d->hit_us = dp[17];
    d->rng = seed;
    d->dies = d->U / d->banks;
    d->S = d->C / d->cps;
    d->K = d->C * d->ncdc;
    d->dinc = d->X > 0 ? (double)d->ppb / d->X : 0.;
    d->thr = d->host_refresh ? 2. * d->d_max : d->d_max;
    int64_t C = d->C, CU = C * d->U, GD = C * d->dies, S = d->S, K = d->K;
    d->inflight = calloc(C, 8); d->timer_gen = calloc(C, 8); d->timer_on = calloc(C, 1); d->link_busy = calloc(C, 1);
    d->link_free = calloc(C, 8); d->waiting = calloc(C, sizeof(Ring)); d->xq = calloc(C, sizeof(Ring));
    d->hold = calloc(C, sizeof(Vec)); d->link_time = calloc(C, 8); d->link_bytes = calloc(C, 8);
    d->queue = calloc(CU, sizeof(Ring)); d->ustate = calloc(CU, 1); d->blocked = calloc(CU, 1);
    d->buf_page = malloc(16 * CU); d->buf_valid = calloc(2 * CU, 1); d->buf_lock = calloc(2 * CU, 4);
    d->buf_stamp = calloc(2 * CU, 8); d->unit_senses = calloc(CU, 8); d->unit_busy = calloc(CU, 8);
    d->sbuf = calloc(CU, 1); d->pend_v = calloc(CU, 1); d->pend_retry = calloc(CU, 1); d->pend_r = calloc(CU, 8);
    d->pend_since = calloc(CU, 8); d->debt = calloc(CU, 8); d->oret = calloc(CU, 8); d->opages = calloc(CU, 8);
    d->res = calloc(CU, 1); d->oslot = calloc(CU, 1); d->want = calloc(GD, 8); d->dmax = calloc(GD, 8);
    d->pL = calloc(S, 8); d->ptL = calloc(S, 8); d->pq = calloc(S, sizeof(Ring)); d->ppending = calloc(S, 1);
    d->cq = calloc(K ? K : 1, sizeof(Ring)); d->cbusy = calloc(K ? K : 1, 1); d->ctime = calloc(K ? K : 1, 8);
    d->nblocks = 1; d->block_reads = calloc(CU, 4);
    if (!d->inflight || !d->timer_gen || !d->timer_on || !d->link_busy || !d->link_free || !d->waiting || !d->xq || !d->hold ||
        !d->link_time || !d->link_bytes || !d->queue || !d->ustate || !d->blocked || !d->buf_page || !d->buf_valid ||
        !d->buf_lock || !d->buf_stamp || !d->unit_senses || !d->unit_busy || !d->block_reads || !d->sbuf || !d->pend_v ||
        !d->pend_retry || !d->pend_r || !d->pend_since || !d->debt || !d->oret || !d->opages || !d->res || !d->oslot ||
        !d->want || !d->dmax || !d->pL || !d->ptL || !d->pq || !d->ppending || !d->cq || !d->cbusy || !d->ctime) {
        dev_free(d); return NULL;
    }
    for (int64_t i = 0; i < 2 * CU; i++) d->buf_page[i] = -1;
    for (int64_t s = 0; s < S; s++) d->pL[s] = d->pdepth;
    return d;
}

void dev_free(Dev *d) {
    if (!d) return;
    int64_t CU = d->C * d->U;
    if (d->waiting) for (int64_t c = 0; c < d->C; c++) free(d->waiting[c].d);
    if (d->xq) for (int64_t c = 0; c < d->C; c++) free(d->xq[c].d);
    if (d->hold) for (int64_t c = 0; c < d->C; c++) free(d->hold[c].d);
    if (d->queue) for (int64_t u = 0; u < CU; u++) free(d->queue[u].d);
    if (d->pq) for (int64_t s = 0; s < d->S; s++) free(d->pq[s].d);
    if (d->cq) for (int64_t k = 0; k < d->K; k++) free(d->cq[k].d);
    free(d->heap); free(d->r_ch); free(d->r_cu); free(d->r_bytes); free(d->r_page); free(d->r_tag); free(d->r_batch);
    free(d->r_open); free(d->r_buf); free(d->inflight); free(d->timer_gen); free(d->timer_on); free(d->link_busy);
    free(d->link_free); free(d->waiting); free(d->xq); free(d->hold); free(d->scratch.d); free(d->queue); free(d->ustate);
    free(d->blocked); free(d->buf_page); free(d->buf_valid); free(d->buf_lock); free(d->buf_stamp); free(d->done_t);
    free(d->done_r); free(d->unit_senses); free(d->unit_busy); free(d->link_time); free(d->link_bytes);
    free(d->block_reads); free(d->sbuf); free(d->pend_v); free(d->pend_retry); free(d->pend_r); free(d->pend_since);
    free(d->debt); free(d->oret); free(d->opages); free(d->res); free(d->oslot); free(d->want); free(d->dmax);
    free(d->pL); free(d->ptL); free(d->pq); free(d->ppending); free(d->cq); free(d->cbusy); free(d->ctime);
    free(d);
}

/* Returns the first request id, -1 for a bad read (the scheme's fault), -2 if time went backwards, -3 out of memory. */
int64_t dev_submit(Dev *d, double t, int64_t n, const int64_t *pages, const int64_t *offs, const int64_t *nbytes,
                   const int64_t *tags, int64_t batch) {
    if (t < d->now - 1e-9) return -2;
    for (int64_t i = 0; i < n; i++) {
        int64_t off = offs[i], nb = nbytes[i];
        if (nb <= 0 || nb % ALIGN || off % ALIGN || off < 0 || off + nb > PAGE || pages[i] < 0) return -1;
    }
    if (d->nreq + n > d->capreq) {
        int64_t cap = d->capreq ? d->capreq : 1024;
        while (cap < d->nreq + n) cap *= 2;
        d->r_ch = grow(d->r_ch, cap, 4, &d->failed); d->r_cu = grow(d->r_cu, cap, 4, &d->failed);
        d->r_bytes = grow(d->r_bytes, cap, 4, &d->failed); d->r_page = grow(d->r_page, cap, 8, &d->failed);
        d->r_tag = grow(d->r_tag, cap, 8, &d->failed); d->r_batch = grow(d->r_batch, cap, 1, &d->failed);
        d->r_open = grow(d->r_open, cap, 1, &d->failed); d->r_buf = grow(d->r_buf, cap, 1, &d->failed);
        if (d->failed) return -3;
        d->capreq = cap;
    }
    int64_t first = d->nreq, g = d->g, C = d->C;
    for (int64_t i = 0; i < n; i++) {
        int64_t page = pages[i], channel = (page / g) % C, local = g * (page / (g * C)) + page % g, r = d->nreq++;
        d->r_tag[r] = tags[i]; d->r_ch[r] = (int32_t)channel; d->r_cu[r] = (int32_t)(channel * d->U + local % d->U);
        d->r_page[r] = local; d->r_bytes[r] = (int32_t)nbytes[i]; d->r_batch[r] = batch != 0; d->r_buf[r] = -1;
        d->r_open[r] = 1;
        d->outstanding += 1;
        push(d, t, ARRIVE, r, 0);
    }
    return d->failed ? -3 : first;
}

void dev_run_until(Dev *d, double t) {
    while (d->nheap && d->heap[0].t <= t && !d->failed) {
        Event e = pop(d);
        d->now = e.t;
        switch (e.kind) {
            case ARRIVE: arrive(d, e.a, e.t); break;
            case SENSED: sensed(d, e.a, (int)e.b, e.t); break;
            case XFERRED: xferred(d, e.a, e.b, e.t); break;
            case CDCDONE: cdc_done(d, e.a, e.b, e.t); break;
            case POWER: power_event(d, e.a, e.t); break;
            case REFDONE: refdone(d, e.a, e.t); break;
            case HITREADY: enqueue(d, e.a, e.t); break;
            default: if (e.b == d->timer_gen[e.a]) flush(d, e.a, e.t);
        }
    }
    if (t > d->now) d->now = t;
}

/* 0 on success, -1 for a bad die or count, -2 if time went backwards. */
int64_t dev_refresh(Dev *d, double t, int64_t die, int64_t n) {
    if (t < d->now - 1e-9) return -2;
    if (die < 0 || die >= d->C * d->dies || n < 0) return -1;
    dev_run_until(d, t);
    d->want[die] += n;
    die_kick(d, die, t);
    return 0;
}

/* Pages owed per die at t (C * dies_per_channel entries). */
void dev_debt(Dev *d, double t, double *out) {
    dev_run_until(d, t);
    memcpy(out, d->dmax, (size_t)(d->C * d->dies) * 8);
}

double dev_settle(Dev *d) {
    int64_t CU = d->C * d->U;
    if (!(d->X > 0) || !CU) return 0.;
    double top = d->debt[0];
    for (int64_t u = 1; u < CU; u++) if (d->debt[u] > top) top = d->debt[u];
    return top * (d->tR + d->t_prog);
}

/* The wake time (device.Device.next_wake); NaN if the device stalled with requests outstanding. */
double dev_next_wake(Dev *d, double cursor, int64_t count, double timeout, int64_t extra) {
    dev_run_until(d, cursor);
    double deadline = cursor + timeout;
    if (d->ndone + extra >= count || d->outstanding == 0) return cursor;
    while (d->nheap && !d->failed) {
        double t = d->heap[0].t;
        if (t > deadline) { dev_run_until(d, deadline); return deadline; }
        dev_run_until(d, t);
        if (d->ndone + extra >= count || d->outstanding == 0) return t;
    }
    return NAN;
}

/* NaN when there is no HBM. */
double dev_hbm_read(Dev *d, double t, int64_t nbytes) {
    if (nbytes <= 0) return t;
    if (d->hbm_bw <= 0) return NAN;
    double begin = t > d->hbm_free ? t : d->hbm_free;
    d->hbm_free = begin + (double)nbytes / d->hbm_bw;
    d->hbm_bytes += nbytes;
    return d->hbm_free + d->hbm_latency;
}

/* channels ascending */
double dev_sram_read(Dev *d, double t, int64_t n, const int64_t *channels, const int64_t *nbytes) {
    dev_run_until(d, t);
    double end = t;
    for (int64_t i = 0; i < n; i++) {
        int64_t c = channels[i], b = nbytes[i];
        if (b <= 0) continue;
        double begin = t > d->link_free[c] ? t : d->link_free[c];
        d->link_free[c] = begin + (double)b / d->bw;
        d->link_time[c] += (double)b / d->bw;
        d->sram_bytes += b;
        end = end > d->link_free[c] ? end : d->link_free[c];
    }
    return end > t ? end + d->sram_latency : t;
}

int64_t dev_ndone(Dev *d) { return d->ndone; }
int64_t dev_outstanding(Dev *d) { return d->outstanding; }
int64_t dev_nreq(Dev *d) { return d->nreq; }
int64_t dev_ncdc(Dev *d) { return d->K; }
int dev_failed(Dev *d) { return d->failed; }

/* Hand the completions over: times, request ids and tags (each array holds dev_ndone entries). */
int64_t dev_take_done(Dev *d, double *t, int64_t *r, int64_t *tag) {
    int64_t n = d->ndone;
    for (int64_t i = 0; i < n; i++) {
        t[i] = d->done_t[i];
        r[i] = d->done_r[i];
        tag[i] = d->r_tag[d->done_r[i]];
        d->r_open[d->done_r[i]] = 0;
    }
    d->ndone = 0;
    while (d->first_open < d->nreq && !d->r_open[d->first_open]) d->first_open++;
    return n;
}

/* Tags of the open requests in issue order; returns how many (writes at most cap). */
int64_t dev_pending(Dev *d, int64_t *out, int64_t cap) {
    int64_t k = 0;
    for (int64_t r = d->first_open; r < d->nreq; r++)
        if (d->r_open[r]) { if (k < cap) out[k] = d->r_tag[r]; k++; }
    return k;
}

int64_t dev_tag(Dev *d, int64_t r) { return d->r_tag[r]; }

/* Counters: hits, senses, sram_bytes, hbm_bytes, hottest block reads, retries, refresh operations.
 * Floats: refresh pages, refresh unit-us, max debt, energy, power wait, settle.
 * Arrays: unit_busy, unit_senses (C*U), link_time, link_bytes (C), core-die busy time (K). */
void dev_counters(Dev *d, int64_t *out) {
    int64_t hottest = 0, n = d->C * d->U * d->nblocks;
    for (int64_t i = 0; i < n; i++) if (d->block_reads[i] > hottest) hottest = d->block_reads[i];
    out[0] = d->hits; out[1] = d->senses; out[2] = d->sram_bytes; out[3] = d->hbm_bytes; out[4] = hottest;
    out[5] = d->retries; out[6] = d->ref_ops;
}
void dev_floats(Dev *d, double *out) {
    int64_t CU = d->C * d->U;
    double top = CU ? d->debt[0] : 0.;
    for (int64_t u = 1; u < CU; u++) if (d->debt[u] > top) top = d->debt[u];
    out[0] = d->ref_pages; out[1] = d->ref_unit_us; out[2] = top; out[3] = d->energy; out[4] = d->power_wait;
    out[5] = dev_settle(d);
}
void dev_arrays(Dev *d, double *unit_busy, int64_t *unit_senses, double *link_time, int64_t *link_bytes, double *ctime) {
    memcpy(unit_busy, d->unit_busy, (size_t)(d->C * d->U) * 8);
    memcpy(unit_senses, d->unit_senses, (size_t)(d->C * d->U) * 8);
    memcpy(link_time, d->link_time, (size_t)d->C * 8);
    memcpy(link_bytes, d->link_bytes, (size_t)d->C * 8);
    if (d->K) memcpy(ctime, d->ctime, (size_t)d->K * 8);
}
