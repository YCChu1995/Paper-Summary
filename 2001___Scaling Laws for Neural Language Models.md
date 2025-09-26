# Scaling Laws for Neural Language Models
> [2001.08361](https://arxiv.org/abs/2001.08361)<br>
<div align=center><img src="/figures/2001.08361.01.png" style="height: 180px; width: auto;"/> <img src="/figures/2001.08361.03.png" style="height: 180px; width: auto;"/></div>

### 9. Optimal Model Size (under fixed compute budget)
- For a fixed compute budget, there is an `optimal model size` with minimum compute requirement. (Figure 11 - Left)
- Though models differ in size, their `local geometry in late training behaves “similarly”`.
  > Because the loss vs 𝑁 curves under fixed S or fixed C seem to follow the same functional exponents across sizes, the authors argue that this suggests that `the shape of loss curvature (i.e. Hessian eigenvalue density) is—not dramatically varying with N`. (Figure 11)
- Training with none-optimal model size will require
  - excess compute (Figure 12 - Left)
  - excess steps for smaller model size but few steps for larger model size(Figure 12 - Right)

<div align=center><img src="/figures/2001.08361.11.png" style="height: 180px; width: auto;"/> <img src="/figures/2001.08361.12.png" style="height: 180px; width: auto;"/></div>

## Summary 

## Tech Insights 
1. The compute $C$ is defined as $6NBS_{FLOPs}$, $C\approx 6NBS$.
   > (N = non-embedding parameter count; B = batch size; S = number of steps)<br>
   
   $6_{FLOPs}$ per parameter per token is coming from $3_{multiply-add} \times 2_{\frac{FLOPs}{multiply-add}}$
   - `1 multiply-add` from the forward pass
   - `1 multiply-add` from gradient wrt weights in the backward pass
   - `1 multiply-add` from gradient wrt activations in the backward pass
   - In FLOP counting, a multiply-add is counted as `2 FLOPs`
   
2. Architecture shape (depth vs width) matters much less than sheer scale (parameter count).
3. Training `larger models`  way `before convergence` are `more efficient` in compute and sample-usage. (To reach a given loss with fewer steps/tokens)
4. For a fixed compute budget, the `optimal strategy` is to make a very `large model` and reach the target loss `before convergence`.
5. Improving `in-distribution` test performance also improves `out-distribution` performance with a `gradually widening offset`. (and `generalization across domains`)<br>
   Better `out-distribution` performance &rarr; Better `generalization across domains`
   > Be careful about the gradually widening [offset](#5-transfer-improves-with-test-performance).
6. The ideal batch size for training these models can
   - be `initialized` as roughly `a power of the loss` only,
   - be `continuesly modified` by measuring the `gradient noise scale`.
7. `Indicator to overfitting` is defined as $"\frac {L(N,D)}{L(N,\infty )} - 1"$ which is a function of $\frac{N^{\frac {\alpha_{N}}{\alpha_{D}}}}{D}$. (Figure 9 - [Right](#3-universality-of-overfitting))
   
---

## Motivation 
- There is no simple, predictive, quantitative recipe for how a language model `performance scales` with (1) `model size`, (2) `dataset size`, and (3) `compute`.
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
  
<div align=center><img src="/figures/2001.08361.05.png" style="height: 180px; width: auto;"/> <img src="/figures/2001.08361.06.png" style="height: 180px; width: auto;"/></div>

### 2. Smooth power laws
- Performance has a `power-law` relationship with each of the three scale factors N, D, C when not bottlenecked by the other two, with trends spanning more than six orders of magnitude. (Figure 1)
  > We observe no signs of deviation from these trends on the upper end, though performance must flatten out eventually before reaching zero loss. (Figure 7)

<div align=center><img src="/figures/2001.08361.01.png" style="height: 180px; width: auto;"/> <img src="/figures/2001.08361.07.png" style="height: 180px; width: auto;"/></div>

### 3. Universality of overfitting
- `Scale up N and D in tandem` &rarr; `Performance improves predictably` (Figure 4, Figure 9)<br>
  `Scale only N or D while the other is held fixed` &rarr; `Diminishing returns` (Figure 4, Figure 9)
- `Indicator to overfitting` is defined as $"\frac {L(N,D)}{L(N,\infty )} - 1"$ which is a function of $\frac{N^{\frac {\alpha_{N}}{\alpha_{D}}}}{D}$. (Figure 9 - Right)
- To avoid overfitting, 
  1. dataset size D should be `initialized` roughly greater than $(5\times 10^{3})N^{0.74}$
  2. `continuously modified` the model size N 8x and the data size D 5x to follow the ratio $\frac{N^{\frac {\alpha_{N}}{\alpha_{D}}}}{D} = \frac{N^{0.74}}{D}$


$${\color{Cyan} \frac {L(N,D)}{L(N,\infty )} - 1} = \frac {L(N,D)-L(N,\infty )}{L(N,\infty )} = {\color{Cyan} f\left( \frac { N^{\frac {\alpha_{N}}{\alpha_{D}}}}{ D }\right)}；f(x) = \left[1 + \left( \frac {D_{C}}{ { N_{C} }^{\frac {\alpha_{N}}{\alpha_{D}}}}\right) \left( \frac { N^{\frac {\alpha_{N}}{\alpha_{D}}}}{ D }\right) \right] ^{\alpha_{D}}-1$$

<div align=center><img src="/figures/2001.08361.04.png" style="height: 180px; width: auto;"/> <img src="/figures/2001.08361.09.png" style="height: 180px; width: auto;"/></div>

### 4. Universality of training
- Training curves follow `predictable power-laws` whose parameters are `roughly independent of the model size`.<br>
  To `predict the training loss` after training, we can `extrapolate` the long-run from the early part of the training curve.(Figtur)



### 5. Transfer improves with test performance
- `Out-Distribution` Test Loss ~ `In-Distribution` Test Loss + `Offest`<br>
  The `Offset` is `increased as model size increased` after some threshold, but `the overall test loss is still decreasing!` (Figure 8 - Left)
  > I think the gradually widening offset is due to some kind of "overfitting" to the training dataset distribution ...<br>
  > `In-Distribution` : Validation on the same dataset distribution as the training one, i.e., training on **WebText2**, and validating on **WebText2** as well.<br>
  > `Out-Distribution` : Validation on a different dataset distribution from the training one, i.e., training on **WebText2**, but validating on **Wikipedia**.
  
- The `out-distribution` test performance depends
  - `strongly` on the `in-distribution` test performance (Figure 8 - Right)
  - `weakly` on the `training duration` or `convergence` (Figure 8 - Right)<br>
    &rarr; training much longer for `convergence` won't help `out-distribution` test performance

- Improving `in-distribution` test performance also improves `out-distribution` performance. 
  > Be careful about the gradually widening offset.

<div align=center><img src="/figures/2001.08361.08.png" style="height: 200px; width: auto;"/></div>

### 6. Sample efficiency
- `Large models` are `more efficient` in compute (Figure 2 - Right) and sample-usage, reaching the same level of performance with
    - fewer optimization steps (Figure 2 - Left, Figure 4 - Right)
    - fewer data points (Figure 4 - Left).

<div align=center><img src="/figures/2001.08361.02.png" style="height: 200px; width: auto;"/> <img src="/figures/2001.08361.04.png" style="height: 200px; width: auto;"/></div>

### 7. Convergence is inefficient
- The best strategy is to train `a large model` and `reach the target performance before convergence` to maximize the compute-efficiency,for a `fixed compute budget C`, `unlimited model size N` or `available data D`. (Figure 3) 
  - Because training to `convergence` is very `inefficient` in both compute and sample-usage. (Figure 2)
  - `Data requirements growing very slowly` wrt training compute, $D \sim C^{0.27}$. (Section 6)
  - There is an `optimal model size N` for a fix compute budge. (Figure 11 - Left)

<div align=center><img src="/figures/2001.08361.02.png" style="height: 200px; width: auto;"/> <img src="/figures/2001.08361.03.png" style="height: 200px; width: auto;"/></div>

### 8.  Optimal batch size
- The ideal batch size for training these models can be estimated be the following approaches.
   1. Following the `power-law` relation with `training loss only`, without `model size`. (As proposed in the pervious paper on gradient noise scale.)
   2. Closing to the measured `gradient noise scale`, which is a great sanity check for how to increase batch size during training.
   
$$B_{crit}(L) = \frac {B_\ast }{L^{\frac {1}{\alpha_B}}}，B_\ast \approx 2\times10^8，\alpha_B\approx 0.21$$

<div align=center><img src="/figures/2001.08361.10.png" style="height: 200px; width: auto;"/></div>


