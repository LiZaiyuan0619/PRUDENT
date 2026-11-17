"""Rules that turn IF-Prune's importance map and a second ("observer") map into the large model's kept set.
Pure torch, no model code.

Every rule returns (score, keep_ratio): the large model keeps the top int(n * keep_ratio) tokens of `score`
(internlm2 / qwen2 modeling in internvl_kl), so a rule builds a float32 score whose top-k is its kept set.
float32 on purpose: the scores are ranks up to n, which bf16 would round into ties.

Notation: a = IF-Prune's map, b = the observer's, n = visual tokens, k = int(n * ratio).
The guide of PRUDENT uses wrank with w = 0.5 (rankavg in internvl_fuse): the average of the two ranks.
"""
import torch


def ranks(x):
    """0 = most important; ties broken by position (stable sort), so the result is deterministic."""
    order = torch.sort(x, descending=True, stable=True).indices
    r = torch.empty(len(x), dtype=torch.long, device=x.device)
    r[order] = torch.arange(len(x), device=x.device)
    return r


def from_order(order, n, device):
    """A score whose descending order is `order` (a permutation of range(n))."""
    s = torch.empty(n, dtype=torch.float32, device=device)
    s[order] = torch.arange(n, 0, -1, dtype=torch.float32, device=device)
    return s


def topk_set(x, k):
    return set(torch.sort(x, descending=True, stable=True).indices[:k].tolist())


def overlap(a, b, ratio):
    k = max(int(len(a) * ratio), 1)
    return len(topk_set(a, k) & topk_set(b, k)) / k


def single(b, ratio):
    """The observer alone (as float32, ties broken by position)."""
    return from_order(torch.sort(b, descending=True, stable=True).indices, len(b), b.device), ratio


def interleave(a, b, ratio):
    """Alternate a's and b's rankings (a first), skipping tokens already taken."""
    n = len(a)
    la = torch.sort(a, descending=True, stable=True).indices.tolist()
    lb = torch.sort(b, descending=True, stable=True).indices.tolist()
    pos, seen, order = [0, 0], set(), []
    lists = (la, lb)
    while len(order) < n:
        for s in (0, 1):
            while pos[s] < n and lists[s][pos[s]] in seen:
                pos[s] += 1
            if pos[s] < n and len(order) < n:
                order.append(lists[s][pos[s]])
                seen.add(lists[s][pos[s]])
                pos[s] += 1
    return from_order(torch.tensor(order, device=a.device), n, a.device), ratio


def wrank(a, b, ratio, w=0.5):
    """Weighted rank fusion: keep the best (1 - w) * rank_a + w * rank_b; w = trust in the observer.
    w = 0.5 is rank averaging. Ties broken by a's rank, then position."""
    n = len(a)
    ra, rb = ranks(a).double(), ranks(b).double()
    key = (1 - w) * ra + w * rb
    order = torch.from_numpy(__import__("numpy").lexsort((ra.cpu().numpy(), key.cpu().numpy())))
    return from_order(order.to(a.device), n, a.device), ratio


def tiebreak(a, b, ratio):
    """a's order, with ties in a (bf16 maps tie a lot) broken by b instead of by position."""
    n = len(a)
    np = __import__("numpy")
    order = np.lexsort((ranks(b).cpu().numpy(), -a.double().cpu().numpy()))
    return from_order(torch.from_numpy(order).to(a.device), n, a.device), ratio


def union(a, b, ratio):
    """Keep the union of each map's top-k (k..2k tokens: more where the maps disagree); inside the
    union and beyond it, a's order decides (only the set matters to the large model)."""
    n = len(a)
    k = int(n * ratio)
    u = sorted(topk_set(a, k) | topk_set(b, k))
    s = from_order(torch.sort(a, descending=True, stable=True).indices, n, a.device)
    s[torch.tensor(u, device=a.device)] += float(n)
    return s, (len(u) + 0.5) / n


def top_share(x, ratio=0.05):
    """Share of a (non-negative) map's mass in its own top `ratio`: how peaked the map is."""
    x = x.double().clamp_min(0)
    k = max(int(len(x) * ratio), 1)
    tot = float(x.sum())
    return float(torch.sort(x, descending=True).values[:k].sum()) / tot if tot > 0 else 0.0


def entw(a, b, ratio):
    """Per-sample trust: weight each map's rank by how peaked it is (no parameter):
    w = peak_b / (peak_a + peak_b). IF-Prune's map is a KL value, b an attention sum; both >= 0."""
    pa, pb = top_share(a), top_share(b)
    w = pb / (pa + pb) if pa + pb > 0 else 0.5
    return wrank(a, b, ratio, w)[0], ratio


def thumb_quota(a, ratio, num_patches, share):
    """a's top-k, with a fixed share of k taken from the thumbnail (the last tile when num_patches > 1)
    by a's order inside the thumbnail, the rest from the local tiles by a's order."""
    n = len(a)
    k = int(n * ratio)
    if num_patches <= 1:
        return single(a, ratio)
    per = n // num_patches
    is_t = torch.zeros(n, dtype=torch.bool, device=a.device)
    is_t[(num_patches - 1) * per:] = True
    order = torch.sort(a, descending=True, stable=True).indices
    kt = min(int(round(share * k)), int(is_t.sum()))
    t_first = order[is_t[order]][:kt]
    rest = order[~torch.isin(order, t_first)]
    rest_local = rest[~is_t[rest]][:k - kt]
    head = torch.cat([t_first, rest_local])
    tail = order[~torch.isin(order, head)]
    return from_order(torch.cat([head, tail]), n, a.device), ratio


def tie_noise(a, ratio, mode, seed=0):
    """a's order with its ties broken on purpose: 'asc' (lower position first), 'desc', or 'rand<seed>'."""
    n = len(a)
    ad = a.double().cpu().numpy()
    np = __import__("numpy")
    if mode == "asc":
        tb = np.arange(n)
    elif mode == "desc":
        tb = -np.arange(n)
    else:
        tb = np.random.default_rng(seed).permutation(n)
    order = np.lexsort((tb, -ad))
    return from_order(torch.from_numpy(order).to(a.device), n, a.device), ratio


def union_k(a, b, k_ratio):
    """Union of each map's top int(n * k_ratio): the disagreement-driven budget (agreeing samples keep
    ~k, disagreeing ones up to 2k); k_ratio is set per task so that the mean keep matches a target."""
    return union(a, b, k_ratio)


def merge_plan(score, keep, a, b, feat, mode, wide=3.0):
    """Partial trust for the dropped tokens: instead of dropping them, average some of them into the most
    similar kept token (cosine on the large model's visual features). The kept count and positions do not change.
      none      nothing merged (the plain rule)
      disagree  merge the dropped tokens that either map ranks in its own top int(wide * k): one estimator thinks they
                matter, the selection did not keep them
      all       merge every dropped token (PruMerge-style, the non-Disagree control)
    -> (merged features [n, C], number of tokens merged)."""
    n = len(score)
    k = int(n * keep)
    kept = torch.sort(score, descending=True, stable=True).indices[:k]
    if mode == "none" or k == 0:
        return feat, 0
    is_kept = torch.zeros(n, dtype=torch.bool, device=feat.device)
    is_kept[kept] = True
    if mode == "all":
        src = (~is_kept).nonzero().view(-1)
    elif mode == "disagree":
        kw = min(n, int(wide * k))
        cand = torch.zeros(n, dtype=torch.bool, device=feat.device)
        cand[torch.sort(a, descending=True, stable=True).indices[:kw]] = True
        cand[torch.sort(b, descending=True, stable=True).indices[:kw]] = True
        src = (cand & ~is_kept).nonzero().view(-1)
    else:
        raise ValueError(mode)
    if len(src) == 0:
        return feat, 0
    f = torch.nn.functional.normalize(feat.float(), dim=-1)
    to = kept[(f[src] @ f[kept].T).argmax(1)]  # each source -> its most similar kept token
    out = feat.float().clone()
    cnt = torch.ones(n, device=feat.device)
    out.index_add_(0, to, feat[src].float())
    cnt.index_add_(0, to, torch.ones(len(src), device=feat.device))
    out[kept] = out[kept] / cnt[kept, None]
    return out.to(feat.dtype), int(len(src))
