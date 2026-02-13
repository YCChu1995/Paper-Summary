# Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet
> [2405](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)<br>
<div align=center><img src="/figures/2405.anthropic.01.png" style="height: 150px; width: auto;"/></div>

## SAE
- Architecture
  > $W_e \in R^{m \times n}, W_d \in R^{n \times m}, b_e \in R^{m}, b_d \in R^{n}, X \in R^{s,n}$<br>
  > ( `n` is the `input and output dimension`, `m` is the autoencoder `hidden layer dimension`, and `s` is the `dataset size`.)
  - Preprocessing step<br>
    Applying a scalar `normalization` to the `model activations` so their average squared L2 norm is the residual stream dimension (embedding dimension), n.<br>
    &rarr; Normalized model activations have `~ unit variance on each embedding dimension`.

  - Main SAE
    > Feature vectors: $\frac{W_{d,i}}{\left|\left| W_{d,i} \right|\right|_2$<br>
    > Feature activations: $f_i(x)\left|\left| W_{d,i} \right|\right|_2$
    
$$f(x) = ReLU(W_e x + b_e)$$
$$\hat{x} = W_d f(x) + b_d$$
$$L = \frac{1}{\left| X \right|} \sum_{x \in X} \left|\left| x - \hat{x} \right|\right|^2_2 + \lambda \sum_{i=1}^m f_i(x) \left|\left| W_{d,i} \right|\right|_2$$

   
   
- Improves from [previous work](https://github.com/YCChu1995/Paper-Summary/blob/main/2310_Towards%20Monosemanticity%20-%20Decomposing%20Language%20Models%20With%20Dictionary%20Learning.md)
  1. 


## Summary 
1. 

## Tech Insights 
1. Pre-encoder bias is useful on synthetic data from small `toy model`, but `removing pre-encoder bias` is beneficial for `real transformer` activations. ([Source](https://transformer-circuits.pub/2024/feb-update/index.html?utm_source=chatgpt.com#dict-learning-loss))

---

## Motivation 
Early work applied sparse autoencoders to `tiny models`, but it was unclear whether those techniques would `scale to real production models`.

## Chain of Thoughts

## Experiment
### 1.
### 2. 
- x
- y<br>
&rarr; y1 + y2 = y3
- z
