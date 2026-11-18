"""Tooth numbering utilities: arch-order dynamic-programming decoder (valid FDI chart, no duplicate numbers,
missing and extra teeth allowed) and box helpers used by the numbering metrics."""
import numpy as np

UP_PERM = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LO_PERM = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]


def primary_of(fdi):
    q, t = divmod(fdi, 10)
    return (q + 4) * 10 + t if t <= 5 else None


JAWS = {'U': UP_PERM, 'L': LO_PERM}


def fdi_jaw(fdi):
    q = fdi // 10
    return 'U' if q in (1, 2, 5, 6) else 'L'


class ArchDecoder:
    def __init__(self, classes, extra_cost=6.0, side_agnostic=False):
        self.classes = list(classes)                 # list of FDI ints, column order of prob vectors
        self.col = {f: i for i, f in enumerate(self.classes)}
        self.extra_cost = extra_cost
        # side_agnostic: merge each class with its left/right mirror (11+21, 31+41, ...)
        # so the classifier only supplies tooth TYPE and the arch geometry decides SIDE.
        swap = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7}
        self.mirror_col = [self.col.get(swap[f // 10] * 10 + f % 10, i)
                           for i, f in enumerate(self.classes)]
        self.side_agnostic = side_agnostic

    def _jaw_slots(self, jaw):
        # list of positions; each position -> (perm_col, primary_col or None)
        out = []
        for f in JAWS[jaw]:
            p = primary_of(f)
            out.append((self.col.get(f), self.col.get(p) if p else None, f, p))
        return out

    def decode(self, centers, probs):
        """centers: (N,2); probs: (N,C) rows sum to 1. Returns list of FDI or None."""
        centers = np.asarray(centers, float); probs = np.asarray(probs, float)
        n = len(centers)
        labels = [None] * n
        if n == 0:
            return labels
        if self.side_agnostic:
            probs = probs + probs[:, self.mirror_col]   # p(type) duplicated on both sides
            probs = probs / np.clip(probs.sum(1, keepdims=True), 1e-9, None) * 2.0
        logp = np.log(np.clip(probs, 1e-9, 1.0))
        # jaw split by probability mass (robust to curved arch)
        up_cols = [self.col[f] for f in self.classes if fdi_jaw(f) == 'U']
        pu = probs[:, up_cols].sum(1) / np.clip(probs.sum(1), 1e-9, None)
        for jaw, sel in (('U', pu >= 0.5), ('L', pu < 0.5)):
            idx = np.where(sel)[0]
            if len(idx) == 0:
                continue
            order = idx[np.argsort(centers[idx, 0])]
            lab = self._dp(logp[order], self._jaw_slots(jaw))
            for i, l in zip(order, lab):
                labels[i] = l
        return labels

    def _dp(self, lp, slots):
        # state: (position p in 0..16, mask m of sub-slots used at position p: bit0 perm, bit1 primary)
        n, P = len(lp), len(slots)
        NEG = -1e18
        # score[i][p][m]: best score after processing i instances, current position p with mask m
        score = np.full((n + 1, P + 1, 4), NEG)
        back = {}
        score[0][0][0] = 0.0          # position 0 = "before first", treated as pos index with empty mask
        def cands(p):
            pc, qc, pf, qf = slots[p]
            c = []
            if pc is not None: c.append((1, pc, pf))
            if qc is not None: c.append((2, qc, qf))
            return c
        for i in range(n):
            for p in range(P + 1):
                for m in range(4):
                    s = score[i][p][m]
                    if s <= NEG / 2:
                        continue
                    # option A: instance i is extra
                    t = s - self.extra_cost
                    if t > score[i + 1][p][m]:
                        score[i + 1][p][m] = t; back[(i + 1, p, m)] = (p, m, None)
                    # option B: assign to a sub-slot at current position p (p>=1) not yet used
                    if p >= 1:
                        for bit, col, f in cands(p - 1):
                            if m & bit:
                                continue
                            t = s + lp[i, col]
                            if t > score[i + 1][p][m | bit]:
                                score[i + 1][p][m | bit] = t; back[(i + 1, p, m | bit)] = (p, m, f)
                    # option C: advance to a later position q>p (skipping = missing teeth) and assign there
                    for q in range(p + 1, P + 1):
                        for bit, col, f in cands(q - 1):
                            t = s + lp[i, col]
                            if t > score[i + 1][q][bit]:
                                score[i + 1][q][bit] = t; back[(i + 1, q, bit)] = (p, m, f)
        # best end state
        p, m = np.unravel_index(np.argmax(score[n]), score[n].shape)
        out = [None] * n
        i = n
        while i > 0:
            pp, pm, f = back[(i, p, m)]
            out[i - 1] = f
            p, m, i = pp, pm, i - 1
        return out


def argmax_decode(classes, probs):
    return [classes[int(k)] for k in np.argmax(probs, 1)]


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def nms_agnostic(boxes, scores, thr=0.5):
    order = np.argsort(-scores); keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1:
            break
        o = iou(boxes[[i]], boxes[order[1:]])[0]
        order = order[1:][o < thr]
    return np.array(keep, int)
