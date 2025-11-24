# Root Mean Square Layer Normalization
> [1910.07467](https://arxiv.org/abs/1910.07467)<br>
> RMS Layer Norm

## Summary 
1. RMSNorm has the `same performance` as LayerNorm, but `cost less`.
   - Comparing to LayerNorm, RMSNorm `speedups` wall-clock time by neglecting operations on mean statistics.<br>
      Because author believes that `re-centering` is less important.
     
2. `pRMSNorm` futher reduce compute cost by `estimating` the RMS term over the first p%⋅n entries.<br>
   Which is emprically proven useful by only set `p to 6.25%` in Figure 2.
   <div align=center><img src="/figures/1910.07467.02.png" style="height: 150px; width: auto;"/></div>
   
3. `RMSNorm` normalize the output distribution to a `√n-scaled unit sphere`.
    - `L2-Norm` normalize the output distribution to a `unit sphere`.<br>
      This paper tested that **L2-Norm** does **NOT** work for layer normalization in Figure 2 above. &rarr; `scaling the sphere by √n` is importnat.
    - Why `scaling the sphere by √n` (i.e. with the size of the input vector) is importnat?<br>
      Author hypothesize that because it makes the normalization more robust `across vectors of different size`.

---

## Motivation 
1. LayerNorm improves convergence in many architectures, but it adds `extra per-step computation` (mean subtraction, variance computation) that can significantly `slow down wall-clock training time`.
2. Community generally believed `recentering is important` to prevent bias/shift, but is this necessary?<br>
   Prior works also tried to approximate or replace `variance computations` (e.g. L1-norm [Wu et al. (2018)](https://arxiv.org/abs/1802.09769)) to reduce nonlinear ops, but most still keep the `re-centering invariance` which requires compute on mean statistic .
