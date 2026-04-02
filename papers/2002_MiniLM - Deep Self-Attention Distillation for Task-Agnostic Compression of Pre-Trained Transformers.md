# MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers
> [2002.10957](https://arxiv.org/abs/2002.10957)<br>
> MiniLM
<div align=center><img src="/figures/2002.10957.01.png" style="height: 250px; width: auto;"/></div>

## Summary 
1. 

## Tech Insights 
1. `Last layer` distillation outperforms `layer-to-layer` distillation.
2. `Value-Relation` transfer outperforms `raw values` transfer.
3. `Value-Relation` transfer boosts distillation performance in scale. &rarr; `Attention - routing; value-relation - content`.
   
---

## Motivation 
- At the moment (~2020), core problems of LMs are:
  1. Too large (100M+ params)
  2. Too slow for
- Existing methods compress `outputs` (logits & hidden states), not `internal reasoning` (attention structure). 
  1. Distillation - Loses performance
  2. Pruning - Hard to optimize
  3. Quantization - Limited gains
- Distillation models (DistilBERT, TinyBERT) mimic `logits` & `hidden states`, but ignore `attention structure`.
- This paper hypothesizes that `attention = knowledge`.
      
## Transformer Distillation
### 1. Overview
- Knowledge distillation is to train the `small student model` $S$ on a transfer feature set provided by the `large teacher model` $T$.
- `Transfer feature set` includes
  1. `Soft labels` (probability distribution for masked language modeling predictions).
  2. `Embedding layer outputs`.
  3. `Self-attention distribution` (scaled attention matrix with softmax, $Softmax\left( \frac{QK^T}{\sqrt{d_k}} \right)$ ).
  4. `Hidden states` (outputs of each Transformer layer).
- Examples:
  1. DistillBERT: `Soft labels` + `Embedding layer outputs`
  2. TinyBERT, MOBILEBERT: `Soft labels` + `Embedding layer outputs` + `Self-attention` + `Hidden states`
### 2. Deep Self-Attention Distillation
- `Transfer feature set` includes
  1. `Self-attention distribution` of the last layer
  2. `Value-Relation` of the last layer
  3. `Teacher assistant` (helps when the size gap between $S$ and $T$ is large)
- Training loss is the sum of `self-attention` and `value-relation`. &rarr; $L = L_{AT} + L_{VR}$
  <div align=center><img src="/figures/2002.10957.01.png" style="height: 250px; width: auto;"/></div>
#### 2-1. Self-Attention Distribution Transfer
- Minimizing the `KL-divergence` between the `self-attention distributions` of $T$ and $S$.
  <div align=center>$$L_{AT} = \frac{1}{A_h \left| x \right|} \sum_{a=1}^{A_h} \sum_{t=1}^{\left| x \right|}  D_{KL} \left( A_{L,a,t}^T \parallel A_{M,a,t}^S \right)$$</div>
  <div align=center>$$A_{l,a} = Softmax\left( \frac{Q_{l,a}K_{l,a}^T}{\sqrt{d_k}} \right)$$</div>    
  
  > $A_h$ is the number of attention heads.<br>
  > $\left| x \right|$ are the sequence length.<br>
  > $d_k = \frac{d_h}{A_h}$<br>
  > $L$ and $M$ are the indices to the last layer of $T$ and $S$.<br>
  > $Q_{l,a}, K_{l,a} \in R^{\left| x \right| \times d_k}$<br>
  > $A_{l,a} \in R^{\left| x \right| \times \left| x \right|}$ is the self-attension distribution.
  
- MiniLM only distills knowledge from the `last layer` only, unlike previous works, which distilled layer-to-layer.
#### 2-2. Value-Relation Transfer
- Minimizing the `KL-divergence` between the `value-relation` of $T$ and $S$.
  > The `value-relation` is computed via the multi-head scaled dot-product with softmax between values.
  <div align=center>$$L_{VR} = \frac{1}{A_h \left| x \right|} \sum_{a=1}^{A_h} \sum_{t=1}^{\left| x \right|}  D_{KL} \left( VR_{L,a,t}^T \parallel VR_{M,a,t}^S \right)$$</div>
  <div align=center>$$VR_{l,a} = Softmax\left( \frac{V_{l,a}V_{l,a}^T}{\sqrt{d_k}} \right)$$</div>    

  > $V_{L,a}^T \in R^{\left| x \right| \times d_k}, V_{M,a}^S \in R^{\left| x \right| \times d_k'}$ are the values of an attention head.<br>
  > $VR_L^T, VR_M^S \in R^{A_h \times \left| x \right| \times \left| x \right|}$ are the value-relation of the last Transformer layer.
### 3. Ablation Studies
- `Last layer` distillation outperforms `layer-to-layer` distillation.
  <div align=center><img src="/figures/2002.10957.T7.png" style="height: 150px; width: auto;"/></div>
- `Value-Relation` transfer outperforms `raw values` transfer.
  <div align=center><img src="/figures/2002.10957.T6.png" style="height: 180px; width: auto;"/></div>
- `Value-Relation` transfer helps distillation performance.
  <div align=center><img src="/figures/2002.10957.T5.png" style="height: 150px; width: auto;"/></div>
- `Value-Relation` transfer works in scale.
  <div align=center><img src="/figures/2002.10957.T8.png" style="height: 100px; width: auto;"/></div>
