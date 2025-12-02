# Technical Design: Importance Sampling Observability Metrics

## Problem Statement

ART computes importance sampling ratios internally for PPO/GRPO training but does not expose these metrics for monitoring. Users have no visibility into:

1. Whether logprobs are being extracted correctly from trajectories
2. Whether importance sampling is actually active (vs. falling back to REINFORCE)
3. How often PPO clipping is triggered

This makes it difficult to debug training issues and verify that the importance sampling pipeline is working correctly.

### Background: How Importance Sampling Works in ART

```
Rollout Phase
    │
    ▼
Trajectories with logprobs attached to messages
    │
    ▼
Tokenization Phase (tokenize.py)
    │
    ├─► Dict messages: extract logprobs if present, else NaN
    └─► Choice objects: extract logprobs if present
    │
    ▼
Training Phase (train.py)
    │
    ├─► If logprobs are NaN: set old_logprobs = new_logprobs.detach()
    │   └─► prob_ratio = exp(0) = 1.0 (NO importance sampling)
    │
    └─► If logprobs are real: compute prob_ratio = exp(new - old)
        └─► PPO clipping applied when ratio outside [1-ε, 1+ε]
```

When all logprobs are NaN, ART silently falls back to vanilla REINFORCE (advantage-weighted policy gradient with no off-policy correction). This is valid but may not be what users expect.

## Solution

Add three new metrics to ART's training loop that are logged to wandb:

### 1. `frac_old_logprobs_valid`

**What it measures:** Fraction of `old_logprobs` values that are NOT NaN at training time.

**Implementation:**
```python
old_logprobs_nan_mask = torch.isnan(old_logprobs)
frac_old_logprobs_valid = 1.0 - (
    old_logprobs_nan_mask.float().sum() / (old_logprobs.numel() + 1e-6)
).item()
```

**Interpretation:**
| Value | Meaning |
|-------|---------|
| 0.0 | All logprobs are NaN - importance sampling NOT active |
| ~0.3-0.5 | Partial logprobs - some tokens have valid logprobs |
| ~0.8-1.0 | Most logprobs valid - importance sampling fully active |

**Why not exactly 1.0?** System messages, tool calls, and prompt tokens don't have logprobs - only assistant response tokens do.

### 2. `mean_importance_ratio`

**What it measures:** Mean importance sampling ratio π_new(a|s) / π_old(a|s) across assistant tokens.

**Implementation:**
```python
mean_importance_ratio = (prob_ratio * assistant_mask).sum() / (assistant_mask.sum() + 1e-6)
```

**Interpretation:**
| Value | Meaning |
|-------|---------|
| Exactly 1.0 | No distribution shift (or all NaN logprobs) |
| 0.8 - 1.2 | Healthy training - policy evolving gradually |
| < 0.5 or > 2.0 | Large distribution shift - may indicate issues |

### 3. `clip_fraction`

**What it measures:** Fraction of assistant tokens where PPO clipping was triggered.

**Implementation:**
```python
clipped_ratio = torch.clip(prob_ratio, 1 - epsilon, 1 + epsilon_high)
is_clipped = (prob_ratio < 1 - epsilon) | (prob_ratio > 1 + epsilon_high)
clip_fraction = (is_clipped.float() * assistant_mask).sum() / (assistant_mask.sum() + 1e-6)
```

**Interpretation:**
| Value | Meaning |
|-------|---------|
| 0.0 | No clipping - either on-policy or no importance sampling |
| 0.01 - 0.1 | Healthy - some off-policy correction happening |
| > 0.3 | High clipping - policy has diverged significantly from rollout policy |

## Implementation Details

### Files Modified

**`src/art/unsloth/train.py`**

1. Compute `frac_old_logprobs_valid` before the NaN replacement:
```python
old_logprobs_nan_mask = torch.isnan(old_logprobs)
frac_old_logprobs_valid = 1.0 - (
    old_logprobs_nan_mask.float().sum() / (old_logprobs.numel() + 1e-6)
).item()
old_logprobs = torch.where(
    old_logprobs_nan_mask,  # reuse mask
    new_logprobs.detach(),
    old_logprobs,
)
```

2. Compute clip metrics after prob_ratio calculation:
```python
clipped_ratio = torch.clip(prob_ratio, 1 - epsilon, 1 + epsilon_high)
is_clipped = (prob_ratio < 1 - epsilon) | (prob_ratio > 1 + epsilon_high)
clip_fraction = (is_clipped.float() * assistant_mask).sum() / (assistant_mask.sum() + 1e-6)
mean_importance_ratio = (prob_ratio * assistant_mask).sum() / (assistant_mask.sum() + 1e-6)
```

3. Log the new metrics:
```python
trainer._metrics["train"]["frac_old_logprobs_valid"].append(frac_old_logprobs_valid)
trainer._metrics["train"]["mean_importance_ratio"].append(mean_importance_ratio.item())
trainer._metrics["train"]["clip_fraction"].append(clip_fraction.item())
```

### Performance Impact

- **Memory:** Negligible - reuses existing tensors, only adds scalar computations
- **Compute:** Negligible - O(n) operations on existing tensors
- **Logging overhead:** 3 additional floats per training step

## Use Cases

### 1. Debugging Missing Logprobs

If `frac_old_logprobs_valid = 0`:
- Check that rollout is requesting logprobs from the model
- Check that logprobs are being attached to trajectory messages
- Check tokenization is extracting logprobs correctly (especially for dict messages)

### 2. Monitoring Training Health

Healthy training should show:
- `frac_old_logprobs_valid` stable and > 0
- `mean_importance_ratio` fluctuating around 1.0
- `clip_fraction` low but non-zero

### 3. Detecting Distribution Drift

If `clip_fraction` suddenly increases:
- Policy may have diverged too far from rollout policy
- Consider reducing learning rate or increasing rollout frequency

## Backwards Compatibility

These changes are additive - existing code continues to work. The new metrics appear in wandb logs automatically if wandb is configured.

## Testing

Manual verification:
1. Run training with valid logprobs → `frac_old_logprobs_valid > 0`
2. Run training with `allow_training_without_logprobs=True` and no logprobs → `frac_old_logprobs_valid = 0`
3. Verify `mean_importance_ratio` deviates from 1.0 over training steps

## Related Work

- PPO paper (Schulman et al., 2017) discusses importance sampling and clipping
- TRL's `PPOTrainer` logs similar metrics (`clipfrac`, `ratio`)
- This brings ART's observability closer to standard PPO implementations
