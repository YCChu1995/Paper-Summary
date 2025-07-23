# On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines
> [2006.04884](https://arxiv.org/abs/2006.04884)<br>
<div align=center><img src="/figures/2006.04884.1.png" style="height: 250px; width: auto;"/></div>

## Motivation 
1. At the moment, there is no good explanation for the BERT fine‑tuning instability.<br>
  Prior work pointed to two main culprits: `catastrophic forgetting` ([1909.11299](https://github.com/YCChu1995/Paper-Summary/blob/main/1909___Mixout-Effective%20Regularization%20to%20Finetune%20Large-scale%20Pretrained%20Language%20Models.md)) and `small dataset sizes` ([1810.04805](https://github.com/YCChu1995/Paper-Summary/blob/main/1810_BERT%20-%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding.md)).<br>
  Yet these explanations don’t fully account for why some runs fail while others succeed.
2. Current solutions improve outcomes empirically, but lack a rigorous mechanistic explanation
    - Intermediate Task Training - STILTS (1811.01088)
    - catastrophic forgetting - Mixout ([1909.11299](https://github.com/YCChu1995/Paper-Summary/blob/main/1909___Mixout-Effective%20Regularization%20to%20Finetune%20Large-scale%20Pretrained%20Language%20Models.md))
3. This training brittleness undermines reproducibility and fair comparisons across models and hyperparameters.

## Experiment
### 1. Catastrophic Forgetting Analysis 
- `Catastrophic Forgetting is NOT the root cause of BERT fine‑tuning instability`, but the byproduct of failing to optimize. 
- In the figure below, both failed (a) and successful (b) runs show degraded MLM perplexity when using outputs from the fine-tuned model versus pretrained layers.<br>
  &rarr; `Both failed (a) and successful (b) runs have knowledge forgetting.`
- In failed runs (a), resetting just ~10 of the top 24 layers recovers most MLM performance.<br>
  &rarr; `Catastrophic forgetting is localized in the upper layers`, not the entire model.
- The failed runs (c) never actually learn the downstream task (0.5 accuracy): their training loss remains at chance-level, and dev accuracy hovers at majority-class results.<br>
  &rarr; `The reasons for fail runs are optimization issues, not excessive forgetting.`
<div align=center><img src="/figures/2006.04884.2.png" style="height: 250px; width: auto;"/></div>

### 2. Dataset Size Analysis
- Small data size w/ same iteration number &rarr; fewer total training steps &rarr; failed runs.<br>
  Small data size w/ more iteration number &rarr; same total training steps &rarr; successful runs.<br>
  &rarr; When iteration count is restored, even a small sample yields stable fine-tuning comparable to using the full dataset.
  &rarr; It is the `reduced total training steps` (due to fewer examples per epoch) that lead to instability.
- `Dataset Size is NOT the root cause of BERT fine‑tuning instability.`
<div align=center><img src="/figures/2006.04884.3.png" style="height: 250px; width: auto;"/></div>

### 3. Vanishing Gradient
- Even with the same hyperparameters and initialization, lack of warm-up and bias correction leads to optimization failures.
  > Failed runs have increasing vanishing gradients over training iterations.<br>
  > Successful runs have decreasing vanishing gradients over training iterations.<br>
  
  &rarr; `Vanishing gradients, preventing effective learning, and then causing trials to fail.`
<div align=center><img src="/figures/2006.04884.4.png" style="height: 250px; width: auto;"/></div>

### 4. Generalization Variance (in accuracy among successful runs)
- Even after escaping early-phase failures, each run follows its own generalization path.<br>
  Extended fine-tuning still yields notable accuracy variance across seeds, underscoring that `optimization isn't the only instability source; generalization differs too`.<br>
  &rarr; During training in successful runs, `accuracy doesn’t plateau early, and fluctuates significantly even late into training, oscillating between ~50% and ~75%`.
- `Low training loss doesn’t guarantee high performance on unseen data.`
<div align=center><img src="/figures/2006.04884.5.png" style="height: 250px; width: auto;"/></div>

## Summary 
1. Debunk popular explanations: `Catastrophic forgetting and small data size are not the reason for fine-tuning instability.`
    - Catastrophic forgetting is the byproduct of optimization failure.
    - Dataset Size is NOT the root cause of BERT fine‑tuning instability, but the reduced total training steps is.
2. `Fine-tuning instability results from optimization failure` with 2 core issues with empirical evidence.
    - `Vanishing gradients` (optimization failures) in early-training (causing most fail runs):<br>
      Solution: **Adam with bias correction** / **learning rate warm up** / **gradient clipping** / ...
    - `Generalization Varience` (low training loss doesn’t guarantee high performance on validation data) in late-training (causing performance variance in success runs):<br>
      Solution: **Train longer** (more iterations/epochs) / **pick or average over multiple random seed runs** / ...
3. Proposed Solution
    1. Optimizer: Use **Adam with bias correction enabled**, countering vanishing gradients at the start.
    2. Learning Rate & Schedule: Keep lr small (~2×10⁻⁵), **linearly warm up** for ~10% of steps, then decay to zero
    3. Longer Training: Instead of the usual 3 epochs, train for ~20 epochs—ensuring convergence to zero training loss and broader exploration of parameter space.

## Tech Insights 
1. `Low training loss doesn’t guarantee high performance on validation data.`<br>
   > This high generalization variance is also found in T5 and GPT. ([2301.09820](https://arxiv.org/abs/2301.09820), [2302.07778](https://arxiv.org/abs/2302.07778))
   > 
   &rarr; `Fine-tuning BERT has high generalization variance.`
2. `Fine-tuning instability results from optimization failure.`
   > **Vanishing gradients** in early-training (causing most fail runs)<br>
   > **Generalization variance** in late-training (causing performance variance in success runs)
