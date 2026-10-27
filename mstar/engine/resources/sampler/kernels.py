"""GPU sampling kernels, imported only by GPU execution paths."""
import torch
import triton
import triton.language as tl

_MAX_THREADS_PER_BLOCK = 1024

def _launchable(configs, device):
    warp_size = torch.cuda.get_device_properties(device).warp_size
    return [c for c in configs if c.num_warps * warp_size <= _MAX_THREADS_PER_BLOCK]

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192},  num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 32768}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 32768}, num_warps=32, num_stages=2),
    ],
    key=["V", "APPLY_PENALTY", "INCLUDE_GREEDY", "WINDOW", "NUM_STOP"],
    prune_configs_by={"early_config_prune": lambda configs, nargs, **kw: _launchable(
        configs, nargs["logits_ptr"].device,
    )},
)
@triton.jit
def _fused_sampling_prep_kernel(
    logits_ptr,        # [B, V] input
    temperature_ptr,   # [B]
    penalty_ptr,       # [B] (only read when APPLY_PENALTY=True)
    seen_mask_ptr,     # [B, V] bool (only read when APPLY_PENALTY=True)
    hist_ptr, hist_stride, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,  # see _history_adjust
    probs_ptr,         # [B, V] float32 output
    V,
    stride_b, stride_v,
    out_stride_b, out_stride_v,
    mask_stride_b, mask_stride_v,
    BLOCK_SIZE: tl.constexpr,
    APPLY_PENALTY: tl.constexpr,
    INCLUDE_GREEDY: tl.constexpr,
    WINDOW: tl.constexpr,
    NUM_STOP: tl.constexpr,
):
    """Fused (optional rep penalty) + (logits/temperature) + softmax.

    When INCLUDE_GREEDY is True and a row's temperature == 0, the kernel
    emits a one-hot distribution at the argmax instead of a temperature-scaled
    softmax — so a downstream multinomial sampler deterministically returns
    the argmax token (replaces the separate torch.argmax + torch.where pair).

    Both constexprs specialize at compile time; the unused branches compile out.
    """
    row = tl.program_id(0)
    temp = tl.load(temperature_ptr + row)
    if INCLUDE_GREEDY:
        is_greedy = temp == 0
        # Safe inv_temp so the softmax branch doesn't produce NaN for greedy
        # rows (their output is overwritten by the one-hot anyway).
        inv_temp = tl.where(is_greedy, 1.0, 1.0 / tl.maximum(temp, 1e-30))
    else:
        inv_temp = 1.0 / temp

    if APPLY_PENALTY:
        penalty = tl.load(penalty_ptr + row)

    # Pass 1: scan over V, compute max of raw vals (post-penalty) + argmax.
    # argmax is only used by the greedy one-hot path; still tracked when
    # INCLUDE_GREEDY is True regardless of per-row temp.
    max_raw = -float("inf")
    max_idx = tl.zeros([], dtype=tl.int32)
    for v_start in range(0, V, BLOCK_SIZE):
        offs = v_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < V
        vals = tl.load(
            logits_ptr + row * stride_b + offs * stride_v,
            mask=mask, other=-float("inf"),
        )
        if APPLY_PENALTY:
            seen = tl.load(
                seen_mask_ptr + row * mask_stride_b + offs * mask_stride_v,
                mask=mask, other=0,
            ).to(tl.int1)
            penalized = tl.where(vals > 0, vals / penalty, vals * penalty)
            vals = tl.where(seen, penalized, vals)
        vals = _history_adjust(
            vals, offs, row, temp, hist_ptr, hist_stride, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
            WINDOW, NUM_STOP,
        )
        masked_vals = tl.where(mask, vals, -float("inf"))
        block_max = tl.max(masked_vals)
        if INCLUDE_GREEDY:
            block_argmax = tl.argmax(masked_vals, axis=0)
            is_new = block_max > max_raw
            max_idx = tl.where(is_new, v_start + block_argmax.to(tl.int32), max_idx)
        max_raw = tl.maximum(max_raw, block_max)

    max_scaled = max_raw * inv_temp

    # Pass 2: exp(scaled - max_scaled), accumulate sum
    sum_exp = tl.zeros([], dtype=tl.float32)
    for v_start in range(0, V, BLOCK_SIZE):
        offs = v_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < V
        vals = tl.load(
            logits_ptr + row * stride_b + offs * stride_v,
            mask=mask, other=0.0,
        )
        if APPLY_PENALTY:
            seen = tl.load(
                seen_mask_ptr + row * mask_stride_b + offs * mask_stride_v,
                mask=mask, other=0,
            ).to(tl.int1)
            penalized = tl.where(vals > 0, vals / penalty, vals * penalty)
            vals = tl.where(seen, penalized, vals)
        vals = _history_adjust(
            vals, offs, row, temp, hist_ptr, hist_stride, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
            WINDOW, NUM_STOP,
        )
        scaled = vals * inv_temp
        exp_val = tl.exp(scaled - max_scaled)
        exp_val = tl.where(mask, exp_val, 0.0)
        sum_exp += tl.sum(exp_val)

    # Avoid div-by-zero in the greedy rows (their output is overwritten).
    inv_sum = 1.0 / tl.maximum(sum_exp, 1e-30)

    # Pass 3: write the output — softmax probs for non-greedy rows,
    # one-hot at argmax for greedy rows.
    for v_start in range(0, V, BLOCK_SIZE):
        offs = v_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < V
        vals = tl.load(
            logits_ptr + row * stride_b + offs * stride_v,
            mask=mask, other=0.0,
        )
        if APPLY_PENALTY:
            seen = tl.load(
                seen_mask_ptr + row * mask_stride_b + offs * mask_stride_v,
                mask=mask, other=0,
            ).to(tl.int1)
            penalized = tl.where(vals > 0, vals / penalty, vals * penalty)
            vals = tl.where(seen, penalized, vals)
        vals = _history_adjust(
            vals, offs, row, temp, hist_ptr, hist_stride, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
            WINDOW, NUM_STOP,
        )
        scaled = vals * inv_temp
        softmax_val = tl.exp(scaled - max_scaled) * inv_sum
        if INCLUDE_GREEDY:
            is_max = offs == max_idx
            one_hot = tl.where(is_max, 1.0, 0.0)
            probs = tl.where(is_greedy, one_hot, softmax_val)
        else:
            probs = softmax_val
        tl.store(
            probs_ptr + row * out_stride_b + offs * out_stride_v,
            probs, mask=mask,
        )


@triton.jit
def _history_adjust(
    vals, offs, row, temp, hist_ptr, hist_stride, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
    WINDOW: tl.constexpr, NUM_STOP: tl.constexpr,
):
    """The ``HistoryRows`` processors on one tile of a row's logits: the windowed
    frequency penalty (``penalty ** n``, n the token's count among the row's
    ring of recent tokens, sign-aware like the presence penalty), and the
    stop-id floor for greedy rows (sampled rows are floored after the filter,
    in ``apply_min_tokens_floor``). Both compile out when off."""
    if WINDOW > 0:
        pen = tl.load(wpen_ptr + row)
        factor = tl.zeros_like(vals) + 1.0
        for w in tl.static_range(WINDOW):
            tok = tl.load(hist_ptr + row * hist_stride + w)
            factor = tl.where(offs == tok, factor * pen, factor)
        vals = tl.where(vals > 0, vals / factor, vals * factor)
    if NUM_STOP > 0:
        floor = (temp == 0) & (tl.load(count_ptr + row) < tl.load(min_tok_ptr + row))
        for s in tl.static_range(NUM_STOP):
            vals = tl.where(floor & (offs == tl.load(stop_ptr + s)), -float("inf"), vals)
    return vals


@triton.jit
def _penalize(vals, seen, penalty):
    """Repetition penalty: divide positive logits, multiply negative ones."""
    return tl.where(seen, tl.where(vals > 0, vals / penalty, vals * penalty), vals)


@triton.jit
def _split_partials_kernel(
    logits_ptr, temperature_ptr, penalty_ptr, seen_mask_ptr,
    hist_ptr, hist_stride, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
    chunk_max_ptr, chunk_sum_ptr, chunk_arg_ptr,
    V, CHUNK,
    stride_b, stride_v, mask_stride_b, mask_stride_v, part_stride_b,
    BLOCK_SIZE: tl.constexpr,
    APPLY_PENALTY: tl.constexpr,
    INCLUDE_GREEDY: tl.constexpr,
    WINDOW: tl.constexpr,
    NUM_STOP: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    temp = tl.load(temperature_ptr + row)
    if INCLUDE_GREEDY:
        inv_temp = tl.where(temp == 0, 1.0, 1.0 / tl.maximum(temp, 1e-30))
    else:
        inv_temp = 1.0 / temp
    if APPLY_PENALTY:
        penalty = tl.load(penalty_ptr + row)

    lo = split * CHUNK
    hi = tl.minimum(lo + CHUNK, V)

    # Online softmax over this chunk: one read, running max and sum.
    run_max = -float("inf")
    run_sum = tl.zeros([], dtype=tl.float32)
    arg = tl.zeros([], dtype=tl.int32)
    for v_start in range(lo, hi, BLOCK_SIZE):
        offs = v_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hi
        vals = tl.load(
            logits_ptr + row * stride_b + offs * stride_v,
            mask=mask, other=-float("inf"),
        ).to(tl.float32)
        if APPLY_PENALTY:
            seen = tl.load(
                seen_mask_ptr + row * mask_stride_b + offs * mask_stride_v,
                mask=mask, other=0,
            ).to(tl.int1)
            vals = _penalize(vals, seen, penalty)
        vals = _history_adjust(
            vals, offs, row, temp, hist_ptr, hist_stride, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
            WINDOW, NUM_STOP,
        )
        scaled = tl.where(mask, vals * inv_temp, -float("inf"))
        block_max = tl.max(scaled)
        if INCLUDE_GREEDY:
            is_new = block_max > run_max
            arg = tl.where(
                is_new, v_start + tl.argmax(scaled, axis=0).to(tl.int32), arg,
            )
        new_max = tl.maximum(run_max, block_max)
        # rescale what we had, then fold this block in. Guard both terms while
        # no finite logit has been seen: `-inf - -inf` is NaN, and an all -inf
        # leading block would carry a NaN sum into the combine step.
        rescale = tl.where(run_max == -float("inf"), 0.0, tl.exp(run_max - new_max))
        terms = tl.where(
            mask & (scaled > -float("inf")), tl.exp(scaled - new_max), 0.0
        )
        run_sum = run_sum * rescale + tl.sum(terms)
        run_max = new_max

    base = row * part_stride_b + split
    tl.store(chunk_max_ptr + base, run_max)
    tl.store(chunk_sum_ptr + base, run_sum)
    if INCLUDE_GREEDY:
        tl.store(chunk_arg_ptr + base, arg)


@triton.jit
def _split_combine_kernel(
    chunk_max_ptr, chunk_sum_ptr, chunk_arg_ptr,
    row_max_ptr, row_inv_sum_ptr, row_arg_ptr,
    NSPLIT, part_stride_b,
    BLOCK_N: tl.constexpr,
    INCLUDE_GREEDY: tl.constexpr,
):
    """Fold the per-chunk partials into one max/sum/argmax per row."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < NSPLIT
    base = row * part_stride_b + offs
    cmax = tl.load(chunk_max_ptr + base, mask=mask, other=-float("inf"))
    csum = tl.load(chunk_sum_ptr + base, mask=mask, other=0.0)
    gmax = tl.max(cmax)
    # each chunk's sum is relative to its own max, so rescale before adding
    gsum = tl.sum(tl.where(mask, csum * tl.exp(cmax - gmax), 0.0))
    tl.store(row_max_ptr + row, gmax)
    tl.store(row_inv_sum_ptr + row, 1.0 / tl.maximum(gsum, 1e-30))
    if INCLUDE_GREEDY:
        carg = tl.load(chunk_arg_ptr + base, mask=mask, other=0)
        winner = tl.argmax(tl.where(mask, cmax, -float("inf")), axis=0)
        tl.store(row_arg_ptr + row, tl.sum(tl.where(offs == winner, carg, 0)))


@triton.jit
def _split_write_kernel(
    logits_ptr, temperature_ptr, penalty_ptr, seen_mask_ptr,
    hist_ptr, hist_stride, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
    row_max_ptr, row_inv_sum_ptr, row_arg_ptr, probs_ptr,
    V, CHUNK,
    stride_b, stride_v, out_stride_b, out_stride_v,
    mask_stride_b, mask_stride_v,
    BLOCK_SIZE: tl.constexpr,
    APPLY_PENALTY: tl.constexpr,
    INCLUDE_GREEDY: tl.constexpr,
    WINDOW: tl.constexpr,
    NUM_STOP: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    temp = tl.load(temperature_ptr + row)
    if INCLUDE_GREEDY:
        is_greedy = temp == 0
        inv_temp = tl.where(is_greedy, 1.0, 1.0 / tl.maximum(temp, 1e-30))
        arg = tl.load(row_arg_ptr + row)
    else:
        inv_temp = 1.0 / temp
    if APPLY_PENALTY:
        penalty = tl.load(penalty_ptr + row)
    gmax = tl.load(row_max_ptr + row)
    inv_sum = tl.load(row_inv_sum_ptr + row)

    lo = split * CHUNK
    hi = tl.minimum(lo + CHUNK, V)
    for v_start in range(lo, hi, BLOCK_SIZE):
        offs = v_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hi
        vals = tl.load(
            logits_ptr + row * stride_b + offs * stride_v,
            mask=mask, other=0.0,
        ).to(tl.float32)
        if APPLY_PENALTY:
            seen = tl.load(
                seen_mask_ptr + row * mask_stride_b + offs * mask_stride_v,
                mask=mask, other=0,
            ).to(tl.int1)
            vals = _penalize(vals, seen, penalty)
        vals = _history_adjust(
            vals, offs, row, temp, hist_ptr, hist_stride, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
            WINDOW, NUM_STOP,
        )
        out = tl.exp(vals * inv_temp - gmax) * inv_sum
        if INCLUDE_GREEDY:
            out = tl.where(is_greedy, tl.where(offs == arg, 1.0, 0.0), out)
        tl.store(
            probs_ptr + row * out_stride_b + offs * out_stride_v,
            out, mask=mask,
        )


@triton.jit
def _count_ge(x_bits, t):
    return tl.sum((x_bits >= t).to(tl.int32), axis=0)


@triton.jit
def _kth_largest_bits(x_bits, k):
    """The largest threshold ``t`` (float bits) with at least ``k`` entries >= t:
    the k-th largest value's bits. Non-negative floats order like their bits."""
    lo = 0
    hi = 0x7F800000
    for _ in range(31):
        mid = lo + (hi - lo + 1) // 2
        ok = _count_ge(x_bits, mid) >= k
        lo = tl.where(ok, mid, lo)
        hi = tl.where(ok, hi, mid - 1)
    return lo


@triton.jit
def _fused_filter_kernel(
    probs_ptr, out_ptr, top_k_ptr, top_p_ptr, first_k_ptr, min_keep_ptr,
    V, stride_in, stride_out,
    HAS_MIN_KEEP: tl.constexpr, BLOCK: tl.constexpr,
):
    """``filter_top_k_top_p`` for one row, in registers: the top-p count of the
    (top-k-first rows: renormalised top-k) distribution by a bitwise search for
    its cutoff, raised to ``min_keep``, clipped to ``k``; then the top-``keep``
    renorm. No sort and one launch for the batch."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    valid = offs < V
    p = tl.load(probs_ptr + row * stride_in + offs, mask=valid, other=0.0)
    p = tl.maximum(p, 0.0)
    bits = p.to(tl.int32, bitcast=True)
    k = tl.load(top_k_ptr + row)
    first_k = tl.load(first_k_ptr + row)
    top_p = tl.load(top_p_ptr + row)

    # the set top-p runs over: the top-`first_k` (all of it for top-p-first rows)
    t_first = 0
    if first_k < V:  # a branch, not tl.where: skip the search when nothing is cut
        t_first = _kth_largest_bits(bits, first_k)
    in_first = bits >= t_first
    mass_first = tl.sum(tl.where(in_first, p, 0.0), axis=0)
    # top-p keeps token i iff the mass strictly above it is < p * mass. That mass
    # falls as i's value rises, so the kept tokens are those >= the smallest
    # threshold t with mass(> t) < p * mass: search for it on the bits
    target = top_p * mass_first
    lo = 0
    hi = 0x7F800000
    for _ in range(31):
        mid = lo + (hi - lo) // 2
        above = tl.sum(tl.where(in_first & (bits > mid), p, 0.0), axis=0)
        ok = above < target
        hi = tl.where(ok, mid, hi)
        lo = tl.where(ok, lo, mid + 1)
    n = _count_ge(bits, tl.maximum(lo, t_first))
    # p = 1 removes nothing; the search can't say so, since "the mass above" the
    # smallest tokens rounds to the whole mass in float32
    n = tl.where(top_p >= 1.0, _count_ge(bits, t_first), n)
    if HAS_MIN_KEEP:
        n = tl.maximum(n, tl.load(min_keep_ptr + row))
    keep = tl.minimum(n, k)
    t_keep = 0
    if keep < V:
        t_keep = _kth_largest_bits(bits, keep)
    out = tl.where(valid & (bits >= t_keep), p, 0.0)
    total = tl.sum(out, axis=0)
    tl.store(out_ptr + row * stride_out + offs, out / tl.maximum(total, 1e-30), mask=valid)


