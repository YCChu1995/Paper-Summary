# Scaling Laws for Neural Language Models
> [2001.08361](https://arxiv.org/abs/2001.08361)<br>
<div align=center><img src="/figures/2001.08361.01.png" style="height: 150px; width: auto;"/></div>



## Summary 
- Performance depends strongly on scale, weakly on model shape
  Model performance depends
    - strongly on scale, which consists of three factors: `the number of model parameters N` (excluding embeddings), `the size of the dataset D`, and `the amount of compute C` used for training.
    - weakly on other `architectural hyperparameters` such as depth or width, within reasonable limits. (Section 3)

- Smooth power laws:
  Performance has a power-law relationship with each of the three scale factors N, D, C when not bottlenecked by the other two, with trends spanning more than six orders of magnitude (see Figure 1).
  We observe no signs of deviation from these trends on the upper end, though performance must flatten out eventually before reaching zero loss. (Section 3)

Universality of overfitting: Performance improves predictably as long as we scale up N and D in tandem,
but enters a regime of diminishing returns if either N or D is held fixed while the other increases. The
performance penalty depends predictably on the ratio N0.74/D, meaning that every time we increase the
model size 8x, we only need to increase the data by roughly 5x to avoid a penalty. (Section 4)

Universality of training: Training curves follow predictable power-laws whose parameters are roughly
independent of the model size. By extrapolating the early part of a training curve, we can roughly predict the
loss that would be achieved if we trained for much longer. (Section 5)

Transfer improves with test performance: When we evaluate models on text with a different distribution
than they were trained on, the results are strongly correlated to those on the training validation set with
a roughly constant offset in the loss – in other words, transfer to a different distribution incurs a constant
penalty but otherwise improves roughly in line with performance on the training set. (Section 3.2.2)

Sample efficiency: Large models are more sample-efficient than small models, reaching the same level of
performance with fewer optimization steps (Figure 2) and using fewer data points (Figure 4).

Convergence is inefficient: When working within a fixed compute budget C but without any other restrictions on the model size N or available data D, we attain optimal performance by training very large models
and stopping significantly short of convergence (see Figure 3). Maximally compute-efficient training would
therefore be far more sample efficient than one might expect based on training small models to convergence,
with data requirements growing very slowly as D ∼ C
0.27 with training compute. (Section 6)

Optimal batch size: The ideal batch size for training these models is roughly a power of the loss only,
and continues to be determinable by measuring the gradient noise scale [MKAT18]; it is roughly 1-2 million
tokens at convergence for the largest models we can train. (Section 5.1) 

## Tech Insights 
1. Architecture shape (depth vs width) matters much less than sheer scale (parameter count).
2. `Larger models are more sample-efficient` (reach a given loss with fewer steps/tokens).
3. For a fixed compute budget, `the optimal strategy is to make the model very large and stop well before convergence` (i.e., early stopping yields best compute efficiency).

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
     
## Chain of Thoughts

## Experiment
### 1.
### 2. 
- x
- y<br>
&rarr; y1 + y2 = y3
- z
