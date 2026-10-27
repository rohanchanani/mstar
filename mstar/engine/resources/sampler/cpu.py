"""CPU realization of the sampler owner's policy; no persistent sampling state."""

import torch

from mstar.engine.resources.sampler.utils import (
    FilterOrder,
    apply_min_tokens_floor,
    filter_top_k_top_p,
)


def _uniform(seed, offset):
    # Preserve Pi's torch-cpu-generator-prefix-f64-cdf-v1 stream exactly.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(offset + 1, generator=generator, dtype=torch.float64)[-1]


def can_use_native_greedy(plan):
    """Whether a producer's plain argmax already implements this invocation."""
    return all(
        c.temperature == 0
        and (not plan.apply_penalty or c.repetition_penalty == 1)
        and c.window_penalty == 1
        and c.min_tokens == 0
        for c in plan.configs
    )


def probabilities(logits, plan):
    """The upstream processor order, using CPU tensors and owner-supplied state."""
    if logits.device.type != "cpu" or logits.ndim != 2 or not logits.shape[1]:
        raise ValueError("CPU sampling requires nonempty [batch, vocabulary] CPU logits")
    if logits.shape[0] != len(plan.configs):
        raise ValueError("sampling plan and logits must have the same rows")
    values = logits.to(torch.float64)
    if values.isnan().any() or values.isposinf().any() or not values.isfinite().any(dim=1).all():
        raise ValueError("CPU logits require a finite choice per row and optional -inf masks")
    out = []
    for i, c in enumerate(plan.configs):
        x = values[i]
        if plan.apply_penalty and c.presence_penalty != 1 and plan.seen_tokens[i] is not None:
            x = torch.where(plan.seen_tokens[i], torch.where(x > 0, x / c.presence_penalty, x * c.presence_penalty), x)
        history = plan.history
        if history is not None and c.repetition_window:
            recent = history.tokens[i, : c.repetition_window]
            counts = torch.bincount(recent[recent >= 0], minlength=x.numel())
            alpha = torch.pow(torch.tensor(c.window_penalty, dtype=x.dtype), counts)
            x = torch.where(x > 0, x / alpha, x * alpha)
        if c.temperature == 0:
            # Upstream applies the floor before argmax on greedy rows.
            if history is not None and c.min_tokens and history.count[i] < c.min_tokens:
                x = x.clone()
                x[history.stop_ids] = -float("inf")
            p = torch.nn.functional.one_hot(x.argmax(), x.numel()).to(x.dtype)
        else:
            # Same operation sequence as the original Pi categorical consumer.
            p = torch.softmax((x - x.max()) / c.temperature, dim=0)
            if c.min_p:
                p = p.masked_fill(p < c.min_p * p.max(), 0)
                p = p / p.sum()
        out.append(p)
    probs = torch.stack(out)
    if plan.apply_filters and any(c.top_k or c.top_p < 1 or c.uses_filter_order for c in plan.configs):
        probs = filter_top_k_top_p(
            probs,
            torch.tensor([c.top_k for c in plan.configs]),
            torch.tensor([c.top_p for c in plan.configs], dtype=probs.dtype),
            FilterOrder(
                torch.tensor([c.top_p_first for c in plan.configs]),
                torch.tensor([c.top_p_min_keep for c in plan.configs]),
            ),
        )
    if plan.history is not None and plan.history.stop_ids is not None:
        probs = apply_min_tokens_floor(probs, plan.history)
    return probs


def sample(logits, plan):
    """Return tokens; only SamplerResource commits RNG/history effects."""
    probs = probabilities(logits, plan)
    result = []
    for i, c in enumerate(plan.configs):
        if plan.offsets[i] < 0:
            raise ValueError("sampling offset must be nonnegative")
        if c.temperature == 0:
            result.append(probs[i].argmax())
            continue
        cdf = probs[i].cumsum(0)
        uniform = _uniform(c.seed, plan.offsets[i])
        total = cdf[-1]
        target = torch.minimum(uniform * total, torch.nextafter(total, torch.zeros_like(total)))
        result.append(torch.searchsorted(cdf, target, right=True))
    return torch.stack(result)


def _plain_categorical_token(logits, plan):
    """Read a singleton plan; return owned I64 storage without committing it.

    CPU algorithm: ``torch-cpu-generator-prefix-f64-cdf-v1``. This is independent
    of the native GPU Philox/Gumbel mapping. Only the resource owner advances
    offsets or retains sampled history. Model suppression is already in logits.
    """
    if logits.device.type != "cpu" or logits.ndim != 1 or not logits.numel():
        raise ValueError("CPU categorical readout requires one nonempty CPU logit row")
    if len(plan.configs) != 1 or len(plan.offsets) != 1 or plan.offsets[0] < 0:
        raise ValueError("CPU categorical readout requires one owner seed/offset row")
    temperature = plan.configs[0].temperature
    if not 0 < temperature < float("inf"):
        raise ValueError("CPU categorical readout requires positive finite temperature")
    values = logits.to(torch.float64)
    if values.isnan().any() or values.isposinf().any() or not values.isfinite().any():
        raise ValueError("CPU categorical logits require a finite choice and optional -inf masks")
    probabilities = torch.softmax((values - values.max()) / temperature, dim=0)
    cdf = probabilities.cumsum(0)
    uniform = _uniform(plan.configs[0].seed, plan.offsets[0])
    total = cdf[-1]
    # Keep the rounded target below the total. right=True skips zero-probability
    # prefix/plateau entries even for an exactly zero generator result.
    target = torch.minimum(uniform * total, torch.nextafter(total, torch.zeros_like(total)))
    return torch.searchsorted(cdf, target, right=True).reshape(1)


def categorical_token(logits, plan):
    """Singleton CPU consumer, retaining the original unfiltered fast path."""
    c = plan.configs[0]
    if (
        c.temperature > 0
        and (not plan.apply_penalty or c.repetition_penalty == 1)
        and c.window_penalty == 1
        and c.min_p == 0
        and c.min_tokens == 0
        and (not plan.apply_filters or (c.top_k == 0 and c.top_p == 1))
    ):
        return _plain_categorical_token(logits, plan)
    if logits.ndim != 1:
        raise ValueError("CPU categorical readout requires one logit row")
    return sample(logits.unsqueeze(0), plan)
