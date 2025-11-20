# How Does Batch Normalization Help Optimization?
> [1805.11604](https://arxiv.org/abs/1805.11604)<br>
<div align=center><img src="/figures/1805.11604.01.png" style="height: 250px; width: auto;"/></div>

## Summary 
Batch Normalization (BatchNorm) helps training, `not by reducing ICS`(internal covariate shift); instead, `by smoothing the optimization landscape` (i.e., BN reduces local Lipschitz constants of the loss and the gradient (β-smoothness)).

---

## Motivation 
Clarify why BatchNorm actually helps optimization: is it ICS or something else?

## Experiment
### 1. Does BatchNorm’s performance stem from controlling internal covariate shift?
- `BatchNorm helps` even when one `manually injects ICS` ("Noisy" BatchNorm).
<div align=center><img src="/figures/1805.11604.02.png" style="height: 250px; width: auto;"/></div>

- The $L_{p}$ normalization techniques lead to `larger distributional covariate shift` compared to normal networks, `yet yield improved optimization performance`.
<div align=center><img src="/figures/1805.11604.14.png" style="height: 120px; width: auto;"/></div>

### 2. Is BatchNorm reducing internal covariate shift?
- BatchNorm didn't guarantee smaller ICS; sometimes it induces worse ICS (in DLN).
- Metrics for ICS
    - L2-difference:<br>
      The L2 norm of the difference between the gradients before/after layer updates.<br>
      If small, that would indicate a small ICS.
    - Cosine similarity:<br>
      The cosine angle between the gradients before/after layer updates.<br>
      If the gradient direction is very similar, cosine ≈ 1 indicates a small ICS.
<div align=center><img src="/figures/1805.11604.03.png" style="height: 250px; width: auto;"/></div>

### 3. Why does BatchNorm work?
- BatchNorm has `smaller` and `smoother` Lipschitzness.
- Metrics
    - Loss Landscape<br>
        1. At every training step, compute the gradient at that point, then move along that gradient (in both positive and negative directions) and plot how the loss changes as a function of the distance moved.
        2. The shaded region in the plot shows the range of loss values when perturbing parameters in the gradient direction, for both the non-BN and BN networks.
    - Gradient Predictiveness<br>
        1. At every training step, they move along the gradient direction and, at each perturbed point, they recompute the gradient. Then they measure the ℓ₂ distance between the original gradient and the new gradient they got after moving.
        2. This measures how predictable the gradient is: if small moves in parameter space drastically change the gradient, that's bad (high curvature, unstable training).
    - “Effective” β-smoothness<br>
      They define an effective β-smoothness as the maximum ℓ₂ change in the gradient over the distance moved in the gradient direction (up to some cap on the step distance).
<div align=center><img src="/figures/1805.11604.04.png" style="height: 250px; width: auto;"/></div>
