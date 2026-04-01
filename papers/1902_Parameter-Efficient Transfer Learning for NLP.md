# Measuring the Intrinsic Dimension of Objective Landscapes
> [1902.00751](https://arxiv.org/abs/1902.00751)<br>
> Adapter Module
<div align=center><img src="/figures/1902.00751.01.png" style="height: 250px; width: auto;"/><img src="/figures/1902.00751.02.png" style="height: 250px; width: auto;"/></div>

## Summary 
1. `Bottleneck design reduces overhead` by projecting (down) hidden states into a small dimension (e.g., 64), applying nonlinearity, and projecting (up) back, then adding residually.
2. Integration points: After `multi-head attention` and `feed-forward sublayers` in Transformer layers.

## Tech Insights 
1. Adapter method solves `parameter redundancy`, `poor scalability`, `catastrophic forgetting` by frozen the pretrained-backbone during training.
2. Removing adapters in higher layers drops performance more than in lower layers-suggesting `higher-layer adapters automatically learn to focus on task-specific higher-level abstractions`.

---

## Motivation 
In the original BERT paper about downstream fine-tuning, there are two approaches.<br>
One is to fine-tune both the pre-trained backbone and added adapter modules; the other is to only fine-tune the added modules while keeping the pre-trained backbone frozen. <br>
In the original BERT paper (Table 7), it shows that `fine-tuned pretrained-backbone always performs better` than other approaches, which highlights `the importance of tuning the pretrained-backbone` for downstream tasks. 
1. During training, it is important to tune the pretrained-backbone. But at the moment, there is no efficient way to do that.
2. During inference, each task needs a full copy of pretrained-backbone, which leads to prohibitive parameter redundancy and poor scalability for multi-task/continual learning scenarios.
3. At the moment, there is a continual training issue: naive fine-tune leads to catastrophic forgetting.
4. Inspired by several works in CV about continuous learning with adapters (residual “adapter” modules, convolutional adapters, Progressive Networks)

## Experiment
### 1. Ablation Study [ FFT & Adapters ]
- With only a few performance overhead (0.4~0.8 GLUE score), it saves a lot of parameters (96.4% ~ 97.9%)<br>

<div align=center><img src="/figures/1902.00751.03.png" style="height: 100px; width: auto;"/></div>

### 2. Ablation Study [ Remove adapters with a seq of adapters ]
- This reinforces the notion that `upper layers specialize more for downstream tasks`, whereas `lower layers capture general features` that transfer well.
  - **Lower layers** (0–4): Ablating adapters in early layers barely hurts MNLI performance.
  - **Higher layers** (8–12): Removing adapters here causes a much larger drop.
- Adapter size and weight init (Gaussian σ) show performance robustness within normal ranges, indicating stable training behavior.
<div align=center><img src="/figures/1902.00751.04.png" style="height: 300px; width: auto;"/></div>
