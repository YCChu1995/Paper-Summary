# Scaling Laws for Neural Language Models
> [2001.08361](https://arxiv.org/abs/2001.08361)<br>
<div align=center><img src="/figures/2001.08361.01.png" style="height: 150px; width: auto;"/></div>



## Summary 

### 7. Convergence is inefficient
- With a `fixed compute budget C`, `unlimited model size N or available data D`, the best strategy is to train `a very large model` and `reach the target performance before convergence` to maximize the compute-efficiency.
  - Because training in convergence is very compute-inefficient. (Figure 2 - Right) 
- 
When working within a  but without any other restrictions on the model size N or available data D, we attain optimal performance by training very large models
and stopping significantly short of convergence (see Figure 3). Maximally compute-efficient training would
therefore be far more sample efficient than one might expect based on training small models to convergence,
with data requirements growing very slowly as D ∼ C
0.27 with training compute. (Section 6)

<div align=center><img src="/figures/2001.08361.02.png" style="height: 200px; width: auto;"/> <img src="/figures/2001.08361.03.png" style="height: 200px; width: auto;"/></div>

- Optimal batch size
The ideal batch size for training these models is roughly a power of the loss only,
and continues to be determinable by measuring the gradient noise scale [MKAT18]; it is roughly 1-2 million
tokens at convergence for the largest models we can train. (Section 5.1) 

## Tech Insights 
1. The compute $C$ is defined as $6NBS_{FLOPs}$, $C\approx 6NBS$.
   > (N = non-embedding parameter count; B = batch size; S = number of steps)<br>
   
   $6_{FLOPs}$ per parameter per token is coming from $3_{multiply-add} \times 2_{\frac{FLOPs}{multiply-add}}$
   - `1 multiply-add` from the forward pass
   - `1 multiply-add` from gradient wrt weights in the backward pass
   - `1 multiply-add` from gradient wrt activations in the backward pass
   - In FLOP counting, a multiply-add is counted as `2 FLOPs`
   
2. Architecture shape (depth vs width) matters much less than sheer scale (parameter count).
3. `Larger models are more sample-efficient` (reach a given loss with fewer steps/tokens).
4. For a fixed compute budget, `the optimal strategy is to make the model very large and stop well before convergence` (i.e., early stopping yields best compute efficiency).
5. Improving `in-distribution` test performance also improves `out-distribution` performance, with a `gradually widening offset`.
   > Be careful about the gradually widening [offset](#5-transfer-improves-with-test-performance).
---

## Motivation 
- There is no simple, predictive, quantitative recipe for how a language model `performance scales` with (1) `model size`, (2) `dataset size`, and (3) `compute`.
  > Before this work, extrapolating gains by `training small models and predicting large model behavior was unreliable`. 
- Before this work,
  1. `Predicting large model behavior by extrapolating gains from training small models` was `unreliable`.
  2. Uncertainty in how to spend extra budget.<br>
     > In late-2019 / early-2020 the community observed that very large models help, but lacked quantitative rules for:<br>
     > "If I double compute, should I double dataset, double model size, or train longer?"
  3. `Data` vs `model` vs `compute tradeoffs` & `overfitting`<br>
     > It was unclear how to `balance model capacity and dataset size` to `avoid overfitting efficiently`.
     
## Experiment Results
### 1. Performance depends strongly on scale, weakly on model shape
- Model performance depends
  - `weakly` on other `architectural hyperparameters` such as depth or width, within reasonable limits. (Figure 5)
  - `strongly` on scale, which consists of three factors: `the number of model parameters N` (excluding embeddings, Figure 6), `the size of the dataset D`, and `the amount of compute C` used for training.
  
<div align=center><img src="/figures/2001.08361.05.png" style="height: 150px; width: auto;"/> <img src="/figures/2001.08361.06.png" style="height: 150px; width: auto;"/></div>

### 2. Smooth power laws
- Performance has a `power-law` relationship with each of the three scale factors N, D, C when not bottlenecked by the other two, with trends spanning more than six orders of magnitude. (Figure 1)
  > We observe no signs of deviation from these trends on the upper end, though performance must flatten out eventually before reaching zero loss. (Figure 7)

<div align=center><img src="/figures/2001.08361.01.png" style="height: 150px; width: auto;"/> <img src="/figures/2001.08361.07.png" style="height: 150px; width: auto;"/></div>

### 3. Universality of overfitting
- `Scale up N and D in tandem` &rarr; `Performance improves predictably` (Figure 4)<br>
  `Scale only N or D while the other is held fixed` &rarr; `Diminishing returns`<br>
- To avoid overfitting,
  1. Dataset size D should be around greater than $(5\times 10^{3})N^{0.74}$
  2. Scale the model size 8x and the data 5x to follow the ratio $\frac{N^{0.74}}{D}$

$$L(N,D)= \left[ \left( \frac{N_{c}}{N} \right)^{\frac {\alpha_{N}}{\alpha_{D}}} + \left( \frac{D_{c}}{D} \right)  \right] ^{\alpha_{D}} \to  ratio = \frac{N^{\frac{\alpha_{N}}{\alpha_{D}}}}{D} = \frac{N^{0.74}}{D} \to D > (5\times 10^{3})N^{0.74}$$

<div align=center><img src="/figures/2001.08361.04.png" style="height: 200px; width: auto;"/></div>

### 4. Universality of training
- Training curves follow `predictable power-laws` whose parameters are `roughly independent of the model size`.<br>
  To `predict the training loss` after training, we can `extrapolate the early part` of the training curve.

### 5. Transfer improves with test performance
- `Out-Distribution` Test Loss ~ `In-Distribution` Test Loss + `Offest`<br>
  The `Offset` is `increased as model size increased` after some threshold, but `the overall test loss is still decreasing!` (Figure 8 - Left)
  > I think the gradually widening offset is due to some kind of "overfitting" to the training dataset distribution ...<br>
  > `In-Distribution` : Validation on the same dataset distribution as the training one, i.e., training on **WebText2**, and validating on **WebText2** as well.<br>
  > `Out-Distribution` : Validation on a different dataset distribution from the training one, i.e., training on **WebText2**, but validating on **Wikipedia**.
  
- The `out-distribution` test performance depends
  - `strongly` on the `in-distribution` test performance (Figure 8 - Right)
  - `weakly` on the `training duration` or `convergence`<br>
    &rarr; training much longer for `convergence` won't help `out-distribution` test performance

- Improving `in-distribution` test performance also improves `out-distribution` performance. 
  > Be careful about the gradually widening offset.

<div align=center><img src="/figures/2001.08361.08.png" style="height: 200px; width: auto;"/></div>

### 6. Sample efficiency
- Large models are more sample-efficient, reaching the same level of performance with
    - fewer optimization steps (Figure 2 - Left)
    - fewer data points (Figure 4).

<div align=center><img src="/figures/2001.08361.02.png" style="height: 200px; width: auto;"/> <img src="/figures/2001.08361.04.png" style="height: 200px; width: auto;"/></div>
