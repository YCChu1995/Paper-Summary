# An Empirical Model of Large-Batch Training
> [1812.06162](https://arxiv.org/abs/1812.06162)<br>
> Gradient Noise Scale
<div align=center><img src="/figures/1812.06162.01.png" style="height: 200px; width: auto;"/></div>

## Summary 
### 1. Batch size
- The `ideal batch size` over training<br>
  The ideal batch size at time $t$, with `trade off examples processed vs. optimizer steps`, scales as

$$B^{*}(t) \propto \sqrt{B_{noise}(t)}$$

- `Gradient noise scale` estimation<br>
  The critical batch size $B_{crit}$ characterizing `cost/time tradeoffs` can be predicted at the order of magnitude level by measuring the gradient noise scale, most easily in the `simplified gradient noise scale` $B_{simple}$.

$$B_{crit}\sim B_{noise}\sim B_{simple}，B_{simple}=\frac{tr(\Sigma)}{G^TG}$$

### 2. Learning rate
- `Optimal learning rate` across runs<br>
  To maintain the training dynamics, one should keep the temperature as a constant, i.e., `the learning rate should scale as the batch size does`.

$$T\equiv \frac{\eta}{B}$$

## Tech Insights 
1. Calculating back propagation in batch is actually `ESTIMATE` the gradient with only a batch of data.
   - Smaller batch-size results in `higher` estimation noise. (Step-to-step variance dominates.)<br>
     Further increasing the batch size can `help the gradient estimation` and `reduce the noise`,<br>
     &rarr; `cut steps-to-convergence`.
   - Larger batch-size estimates well with `little` noise. (Two random batches produce nearly identical gradients.)<br>
     So doubling the batch size barely helps.<br>
     Further increasing the batch size would only `increase the compute budget` with `no noise left to decrease`,<br>
     &rarr; `cutting no steps-to-convergence`.
2. Measuring the `simplified noise scale` early during training, one can help determine the useful upper limit for `batch size estimation`.

---

## Motivation 
1. `There should be a predictive, task-agnostic rule for choosing a batch size`, which would deliver near-linear parallel speedups instead of becoming compute-wasteful. 
   > Different domains reported very different **max useful** batch sizes (e.g., ~10^3–10^4 for ImageNet vs. ~10^6 in RL/Dota) 

## Chain of Thoughts
### 1. The Conceptual Setup: “Signal” vs. “Noise” in Gradients
- In SGD, each mini-batch produces a `gradient estimate` that contains both a `signal` (the true full-batch gradient) and `noise` (due to sampling). 
  - When the batch size is small, noise dominates.
  - When the batch size is large, noise is reduced.

### 2. Gradients, Batches, and the Gradient Noise Scale
- Optimal `per-step loss improvement` ($\Delta L_{opt}$) and the corresponding `learning rate` ($\epsilon_{opt}$) of SGD can be described as a function of `batch size`,
  > Details in equation (2.6) ~ (2.8)<br>
  > Scaling behavior is confirmed in the experiment, [Figure 5](#1-optimal-learning-rate-scaling-behavior) for `learning rate`.

$$\epsilon_{opt}(B)\propto \frac{1}{1+B_{noise}/B}，\Delta L_{opt}(B)\propto \frac{1}{1+B_{noise}/B}$$

- $B_{noise}$ is defined as the `gradient noise scale` to capture the `noise-to-signal ratio`,
  - `Noise` information is captured by $\Sigma$, a covariance matrix defined in equation (2.3).
  - `Singal` information is captured by $G$, the true gradient matrix.
    
$$B_{noise}=\frac{tr(H\Sigma )}{G^THG}$$

### 3. Gradient Noise Scale
- Quantitative analysis
  - When B $\ll$ $B_{noise}$ , both $\epsilon_{opt}(B)$ and $\Delta L_{opt}(B)\approx \frac{B}{B_{noise}}$. (nearly linear speed-up)
  - When B $\gg$ $B_{noise}$ , both $\epsilon_{opt}(B)$ and $\Delta L_{opt}(B)\approx 1$. (plateaus—minimal gains)
  - $B_{noise}$ is a key insight for `batch-size choice`, because it is the `turning point` between these two regimes.
<div align=center><img src="/figures/1812.06162.02.png" style="height: 180px; width: auto;"/></div>

- Critical batch size $B_crit$
  `Critical batch size should be $B_{noise}$`, where speed-efficiency trade-off hits ~50%.
  > Visable in ([Figure 6](#2-time-vs-compute-efficiency))

- Dependent on
  - Signal ($\left\| G\right\|$)
    - Gradient noise scale `growth over successful training`.
      > Due to `gradient "signal" drop` (since reached lower loss regime) while `gradient variance remaining the same`.<br>
      > Training successfully &rarr; Reaching low loss regime &rarr; $\left\| G\right\| \downarrow，\Delta \Sigma \sim 0 $ &rarr; $B_{noise} \uparrow$ ([Experimental Evidence](#4-gradient-noise-scale-is-independent-to-model-size-and-increasing-during-success-training))
  - Noise ($\sigma$)
    - `Difficult tasks` have higher gradient noise scale.
      > Difficult tasks &rarr; diverse data, long horizons, sparse rewards &rarr; increased gradient variance (Empirical evidence in [Table 1](#3-simple-gradient-noise-scale-is-good-enough))<br>
      > This reflects `increased gradient variance` from less correlated examples (e.g., complex RL environments like Dota).

- Independent of
  - `Full dataset size`, so it `generalizes across domains` (supervised, RL, generative). 
    > [Experimental Evidence](#5-gradient-noise-scale-works-among-different-task)
  - `Model size` has `NO directly affect` on $B_{noise}$.<br>
    The indirect effect: When `larger model size is trained`, and reaching the `lower loss regime`, which causes `gradient "signal" drop`. Then leads to `noise scale growth`.
    > Larger model is well-trained &rarr; Lower loss regime  &rarr; Gradient "signal" drop  &rarr; Gradient noise scale growth ([Experimental Evidence](#4-gradient-noise-scale-is-independent-to-model-size-and-increasing-during-success-training))

- Sensitive to learning rate ("temperature")<br>
  If the learning rate is too small, the measured noise scale can become artificially inflated.<br>
  Hence, measurements should use well-tuned learning rates for meaningful interpretation.
  
- Practical “simple” version
  > The presence of the Hessian H requires some overhead to compute.<br>
  > `$B_{simple}$ is good enough`, and in the same magnitude as $B_{crit}$. (Supported by [Table 1](#3-simple-gradient-noise-scale-is-good-enough))<br>
  > `$B_{simple}$ and $B_{noise}$ typically differ only by a small constant multiplicative factor.` ([Experimental Evidence](#3-simple-gradient-noise-scale-is-good-enough))<br>

$$B_{simple}=\frac{tr(\Sigma )}{G^TG}$$

### 4. Predictions for Data/Time Efficiency Tradeoffs
- An `inverse-proportion relation` between `number of steps` ($S$) and `number of examples` ($E$) taken to reach a specified level
of performance.
  > Derived in Appendix D.<br>
  > Fit well with the Pareto curve in [Figure 7](#3-simple-gradient-noise-scale-is-good-enough).
  
$$\frac{S}{S_{min}}-1=\left (\frac{E}{E_{min}}-1\right )^{-1}$$

- Since $E_{tot} = B\cdot S_{tot}$, we defined the critical batch size $B_{crit}$ as,
  > $B_{crit}$ is defined as the point on the Pareto curve where time and compute efficiency begin to sharply degrade, typically where efficiency drops by around 50%, meaning you're using approximately twice the minimal examples or steps.

$$B_{crit} = \frac{E_{min}}{S_{min}}$$

- $B_{crit}$ and $B_{noise}$ are in the same `magnitude`.<br>
  Their model (Appendix D) predicts that (where $B_{noise}$ is appropriately averaged over training),
  > The correlation is numerically supported by Table 1 in [Figure 7](#3-simple-gradient-noise-scale-is-good-enough).
  
$$B_{crit} \approx B_{noise}$$

### 5. Caveats
- Short-horizon bias<br>
  The analysis in earlier sections is local; it considers only the immediate step in loss reduction.<br>
  But greedy improvements (making the `best one-step update`) might NOT yield the `best long-term trajectory`, especially in non-convex landscapes, bad local minima, or ill-conditioned regions.
- Poor conditioning<br>
  If the optimization problem is `ill-conditioned` (where different dimensions scale very differently), the `quadratic approximation breaks down`.
- Generalization gap<br>
  The model `only addresses training loss`, we don’t make claims about `generalization performance` (i.e., test loss).

## Experiments
### 1. Optimal learning rate scaling behavior
- For optimizers like SGD or momentum, the learning rate follows [theoretical derivation](#2-gradients-batches-and-the-gradient-noise-scale) (linear scaling with batch size up to a point).
- For Adam/RMSProp, the rate scales sub-linearly.
   
$$\epsilon_{opt, SGD}(B)=\frac{\epsilon_{max}}{1+B_{noise}/B}；\epsilon_{opt, Adam}(B)\propto B^{\alpha}，\alpha =0.5\sim 1$$ 
<div align=center><img src="/figures/1812.06162.03.png" style="height: 250px; width: auto;"/></div>

### 2. Time vs compute efficiency
- `Optimization steps` and `examples processed` are viewed as the indicators for `time-efficiency` and `compute-efficiency`.<br>
  > They assume run time per step is batch size irrelevant.
- Experiment shows that a larger batch has higher time-efficiency and lower compute-efficiency.
- Considering the run time (is batch size related) reality, finding a sweet spot is important.<br>
  And the turning point in the graph is around B = 64 , aligning with theoretical predictions.
<div align=center><img src="/figures/1812.06162.04.png" style="height: 250px; width: auto;"/></div>

### 3. Simple gradient noise scale is `good enough`
<div align=center><img src="/figures/1812.06162.05.png" style="height: 200px; width: auto;"/> <img src="/figures/1812.06162.06.png" style="height: 200px; width: auto;"/></div>

### 4. Gradient noise scale is independent of model size, and increases during Success training
- `Gradient noise scale is directly independent of model size`. (but indirectly dependent on the model size)
  > Larger model size &rarr; lower loss regime (low perplexity) &rarr; higher gradient noise scale
- `Gradient noise scale increases as training proceeds`.
<div align=center><img src="/figures/1812.06162.07.png" style="height: 250px; width: auto;"/></div>

### 5. Gradient noise scale works across different tasks
<div align=center><img src="/figures/1812.06162.08.png" style="height: 250px; width: auto;"/></div>

## Appendix
### 1. Temperature and the Noise Scale
- Empirical proportionality<br>
  In experiments, they observe that `across runs, the gradient noise scale is proportional to the temperature`, which they defined as
  
$$B_{noise}\propto \frac{1}{T}，T\equiv \frac{\eta}{B} $$

- Implication - Hyperparameter scaling<br>
  If you maintain the temperature $\frac{\eta}{B}$, you can `move to much larger batch sizes without retuning requirement`.

### 2. Dynamically Varying the Batch Size
- The ideal batch size $B^{*}(t)$ at time $t$, assuming you want to `trade off examples processed vs. optimizer steps`, scales as

$$B^{*}(t) \propto \sqrt{B_{noise}(t)}$$
