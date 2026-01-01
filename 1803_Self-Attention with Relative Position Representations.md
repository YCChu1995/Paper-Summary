# Self-Attention with Relative Position Representations
> [1803.02155](https://arxiv.org/abs/1803.02155)<br>

<div align=center><img src="/figures/1803.02155.01.png" style="height: 150px; width: auto;"/></div>

## Summary 
1. Architecture<br>
   Both $\alpha_{ij}^{K}$ and $\alpha_{ij}^{V}$ are learned matricies for `relative position embedding`.
   
| Block                  | Standard Self-Attention       | Relation-Aware Self-Attention |
| :--------------------: | :---------------------------: | :---------------------------: |
| Scaled Attention Score | $$e_{ij} = \frac{ \left(x_{i}W^{Q}\right)\left(x_{j}W^{K}\right)^{T} }{ \sqrt{d_{z}} }$$   | $$e_{ij} = \frac{ \left(x_{i}W^{Q}\right)\left(x_{j}W^{K}+\alpha_{ij}^{K}\right)^{T} }{ \sqrt{d_{z}} }$$ |
| Softmax                | $$a_{ij} = \frac{ exp(e_{ij}) }{ \sum_{k=1}^{n} exp(e_{ik}) }$$ | Same |
| Context matrix         | $$z_{i} = \sum_{j}a_{ij}(x_{j}W^{V})$$ | $$z_{i} = \sum_{j}a_{ij}(x_{j}W^{V}+\alpha_{ij}^{V})$$ |

<div align=center><img src="/figures/1803.02155.T1.png" style="height: 150px; width: auto;"/></div>

2. Clipping Distance<br>
   They hypothesized that precise relative position information is `not useful beyond a certain distance`.
   Their experimental result (Table 2) verifies that the distance threshold is short (2).
   
 $$\alpha_{ij} = w_{clip(j-i, k)}$$
 $$clip(x, k) = max(-k, min(k, x))$$
  
   <div align=center><img src="/figures/1803.02155.T2.png" style="height: 150px; width: auto;"/></div>

3. Ablating Relative Position Embedding<br>
   Despite the ablation study, they think further work is needed to determine whether this is true for other tasks.

   <div align=center><img src="/figures/1803.02155.T3.png" style="height: 150px; width: auto;"/></div>
   
4. Efficient Implementation<br>
   Sharing parameters `across heads`

## Motivations
- Standard Transformer uses sinusoidal position embedding (absolute position embedding).
- Problems with absolute position embedding:
  1. They treat position `independently of content`.
  2. They don’t directly encode `pairwise` relationships like “token j is 3 steps ahead of token i.”
  3. This matters especially in tasks where relative ordering is crucial (e.g., translation).
