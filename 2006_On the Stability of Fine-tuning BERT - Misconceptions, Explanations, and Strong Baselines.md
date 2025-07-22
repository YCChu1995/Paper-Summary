# On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines
> [2006.04884](https://arxiv.org/abs/2006.04884)<br>

## Motivation 
1. At the moment, there is no good explanation for the BERT fine‑tuning instability.<br>
  Prior work pointed to two main culprits: `catastrophic forgetting` ([1909.11299](https://github.com/YCChu1995/Paper-Summary/blob/main/1909___Mixout-Effective%20Regularization%20to%20Finetune%20Large-scale%20Pretrained%20Language%20Models.md)) and `small dataset sizes` ([1810.04805](https://github.com/YCChu1995/Paper-Summary/blob/main/1810_BERT%20-%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding.md)).<br>
  Yet these explanations don’t fully account for why some runs fail while others succeed.
2. This training brittleness undermines reproducibility and fair comparisons across models and hyperparameters.

## Experiment
### 1. Catastrophic Forgetting Analysis 
- `Catastrophic Forgetting is NOT the root cause of BERT fine‑tuning instability`, but the byproduct of failing to optimize. 
- In the figure below, both failed (a) and successful (b) runs show degraded MLM perplexity when using outputs from the fine-tuned model versus pretrained layers.<br>
  &rarr; `Both failed (a) and successful (b) runs have knowledge forgetting.`
- In failed runs (a), resetting just ~10 of the top 24 layers recovers most MLM performance.<br>
  &rarr; `Catastrophic forgetting is localized in the upper layers`, not the entire model.
- The failed runs (c) never actually learn the downstream task (0.5 accuracy): their training loss remains at chance-level, and dev accuracy hovers at majority-class results.<br>
  &rarr; `The reasons for fail runs are optimization issues, not excessive forgetting.`
<div align=center><img src="/figures/2006.04884.1.png" style="height: 250px; width: auto;"/></div>

### 2. Dataset Size Analysis

## Summary 
1. 

## Tech Insights 
1. 

- z

## Summary 
1. 

## Tech Insights 
1. 
