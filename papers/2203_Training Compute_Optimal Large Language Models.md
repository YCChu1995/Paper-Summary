# Training Compute-Optimal Large Language Models
> [2203.15556](https://arxiv.org/abs/2203.15556)<br>

<div align=center><img src="/figures/2203.15556.01.png" style="height: 250px; width: auto;"/> <img src="/figures/2203.15556.04.png" style="height: 250px; width: auto;"/></div>

## Summary 
1. The `model size (N)` and the `number of training tokens (D)` should be `scaled equally`.<br>
   (Comparison to **Kaplan et al. (2020)** in Figure A4.)

$$N_{opt} \propto C^{~0.5}, D_{opt} \propto C^{~0.5}$$

<div align=center><img src="/figures/2203.15556.T3.png" style="height: 120px; width: auto;"/> <img src="/figures/2203.15556.A4.png" style="height: 200px; width: auto;"/></div>

2. Why the results differ from ones in **Kaplan et al. (2020)**
    - Different Learning Rate Scheduele<br>
      Kaplan et al. (2020): A `fixed learning-rate schedule` across different model sizes.<br>
      Hoffmann et al. (2022): `Optimal` cosine annealing learning rates
    - Different Experimental Regime
        - Data Size:<br>
          Kaplan et al. (2020) ~ 1.5 B<br>
          Hoffmann et al. (2022) ~ 23 B
        - Parameter Counting Methods<br>
          Kaplan et al. (2020) doesn't count embedding which makes parameter scaling exponents skewed. (Especially for smaller models.)<br>
          According to **Reconciling Kaplan and Chinchilla Scaling Laws** (Pearce & Song, 2024), `this counting choice is one of the main reasons for Kaplan’s “overestimation” of the exponent a`.


## Tech Insights 
1. `Overestimating` the number of training step with setting `cosine cycle length` leads to `performance drops`.
<div align=center><img src="/figures/2203.15556.A1.png" style="height: 250px; width: auto;"/></div>

2. `Optimal cosine cycle length`: they assume that
    1. The cosine cycle `length` correctly calibrated to the `maximum number of steps`.<br>
       (As in Figure A1 above.)
    2. `10x learning rate decay` in line with Rae et al. (2021) `instead of decaying to 0`.<br>
       Quote from the note in p.22:<br>
       We find the difference between decaying by 10× and decaying to 0.0 (over the same number of steps) to be small, though decaying by a factor of 10× to be slightly more performant. Decaying by less (5×) is clearly worse.

3. `AdamW` **silghtly outperformance** Adam in both `language modelling loss` and the `downstream task performance after finetuning`.
<div align=center><img src="/figures/2203.15556.A7.png" style="height: 250px; width: auto;"/></div>

---

## Motivation 
The authors disagree with the larger model size trend; they believe that the model size and the number of training tokens should be scaled equally.
1. Prior work [(Kaplan et al., 2001.08361)](https://github.com/YCChu1995/Paper-Summary/blob/main/2001_Scaling%20Laws%20for%20Neural%20Language%20Models.md) suggests that one should increase 5.4x on N and 1.8x on D while C is increased on 10x, which leads to the larger model size trend.
2. Gathering extremely large high‐quality datasets (≥ 1 trillion tokens) seemed plausible but expensive, which also leads to the larger model size trend.
3. Many recent large LMs have grown primarily by increasing model size (e.g., GPT‑3 175B, Gopher 280B, Megatron‑Turing NLG 530B) while holding the number of training tokens roughly constant (~300 B tokens) despite increasing compute budgets.
4. Such undertrained over-parameterized models cost a lot during fine-tuning and inference.

## Experiment
### Approach 1 : Fix model sizes and vary number of training tokens
<div align=center><img src="/figures/2203.15556.02.png" style="height: 250px; width: auto;"/></div>

### Approach 2 : IsoFLOP profiles
- Lines in **Figure 3 - Left** is fitted in parabola.
<div align=center><img src="/figures/2203.15556.03.png" style="height: 250px; width: auto;"/></div>

### Approach 3: Fitting a parametric loss function
<div align=center><img src="/figures/2203.15556.04.png" style="height: 250px; width: auto;"/></div>
