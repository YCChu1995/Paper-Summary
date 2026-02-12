# Towards Monosemanticity: Decomposing Language Models With Dictionary Learning
> [2310.anthropic](https://transformer-circuits.pub/2023/monosemantic-features/index.html)<br>
<div align=center><img src="/figures/2310.anthropic.03.png" style="height: 300px; width: auto;"/> <img src="/figures/2310.anthropic.02.png" style="height: 150px; width: auto;"/></div>

### Summary 
Using `sparse`, `overcomplete` autoencoder to learn (decompose) `monosementic` features from `polysementic` activations.

## Tech Insights 
1. [Tips to train SAE](#tips-to-train-sae)
2. [Metrics to measure SAE](#metrics-to-measure-sae-far-from-perfect)
---

## Motivation 
- Reverse engineering neural networks to `understand` the language model.
- There is significant empirical evidence suggesting that neural networks have `interpretable` `linear` directions in `activation space`.
  &rarr; If linear directions are interpretable, it's natural to think there's some `basic set` of meaningful directions which more complex directions can be created from.
  &rarr; We call these `directions features`, and they are the target to decompose models into.
2. Superposition
3. MLP layer likely uses superposition to represent more features than it has neurons

## Chain of Thoughts
### 1. Problem
- Neurons in MLP layers are `polysemantic`.
  > MLP layer likely uses superposition to represent more features than it has neurons.
### 2. Target
- Get `directions features` (`basic set` of meaningful directions) by decomposing `polysemantic` neurons.
### 3. Solution
- Decompose the `activation vector` as a combination of `more general features`.
  > $x^j$ is the `activation vector` for the `datapoint j`.<br>
  > $f_i(x^j)$  is the `activation of feature i`.<br>
  > $d_i$ is the `direction of feature i`, which is a `unit vector` in activation space.<br>
  > ( `n` is the `input and output dimension` and `m` is the autoencoder `hidden layer dimension`)<br>

$$x^j \approx b + \sum_{i=1}^m f_i(x^j)d_i$$
  
- `Sparse`, `overcomplete` autoencoder, `SAE`.
  > Decompose into more features than there are neurons.<br>
  > $\bar{x} = x - b_d$<br>
  > $f = ReLU(W_e \bar{x} + b_e)$<br>
  > $\hat{x} = W_d f + b_d$<br>
  > $L = \frac{1}{\left | X \right |} \sum_{x \in X} \left\| x - \hat{x} \right\|^2_2 + \lambda \left\| f \right\|_1$<br>
  > $W_e \in R^{m \times n}, W_d \in R^{n \times m}$ with $m$ columns of `unit norm`, $b_e \in R^{m}, b_d \in R^{n}$<br>
  > ( `n` is the `input and output dimension` and `m` is the autoencoder `hidden layer dimension`)
<div align=center><img src="/figures/2310.anthropic.01.png" style="height: 250px; width: auto;"/> <img src="/figures/2310.anthropic.T1.png" style="height: 150px; width: auto;"/></div>

## Tips to train SAE
### 1. Pre-Encoder Bias
- Prevent overfitting with `overcomplete` autoencoder
  > Without a bias shift, an overcomplete encoder could trivially learn to `pass the input straight through` (identity) without learning meaningful structure, especially when there are more features than dimensions.
- Centres the inputs, let the encoder learning focus on the deviation
  > In some implementations, this bias is `initialized` to the geometric median of the dataset and effectively centers the input activations.<br>
  > This helps the model `focus on` the `deviation` from typical activity `instead of encoding` the `mean` level of activations.
### 2. Decoder Weights Not Tied
- Increase `representational capacity`
  > Qoute: "However, we find that in our trained models the learned encoder weights are `not the transpose` of the decoder weights ..."
- Prevent `crosstalk` among closely `related dictionary vectors`
  > Similar features which have closely `related dictionary vectors` have `encoder weights that are offset` so that they prevent crosstalk between the noisy feature inputs.
### 3. Neuron Resampling
- Steps
  1. At training steps 25,000, 50,000, 75,000 and 100,000, `identify dead neurons` in SAE which have not fired in any of the previous 12,500 training steps.
     > Resampling too frequently causes training to diverge.
  2. `Compute the loss` for the current model on a random subset of 819,200 inputs.
  3. Assign each input vector a `probability` of being picked that is proportional to the `square of the autoencoder’s loss` on that input.
     > Higher loss data has higher chance to be picked in the following steps.
  4. For each dead neuron `sample an input` according to these probabilities.<br>
     Give the dead neuron a new `dictionay vector` by setting it as the `renormalized input vector` (unit L2 norm).
     > Force dead neurons to represent high loss data.
  5. For the corresponding `encoder vector`, `renormalize the input vector` to equal the average norm of the `encoder weights` for `alive neurons × 0.2`.<br>
     Set the corresponding encoder bias element to zero.
     > Multiple the new weight by a small factor (x 0.2) to mitigate the sudden loss spikes.
  6. `Reset the Adam` optimizer parameters for every `modified weight and bias` term.
- **Pros**:<br>
  This approach `outperforms` baselines including `no resampling` and `reinitializing` the relevant encoder and decoder weights using a default Kaiming Uniform initialization.
- **Cons**:<br>
  This approach still causes `sudden loss spikes`, and resampling too frequently causes training to `diverge`.
- Cause to more dead neurons,
  1. learning rate (too high)
  2. batch size (too low)
  3. dataset redundancy (too many tokens per context or repeated epochs over the same dataset)
  4. number of training steps (too many)
### 4. Learning Rate Sweep Results
- `Lower learning rates` w/ `sufficient training steps` result in `lower total loss` and `more “real” features` discovered.
- `Annealing learning rate` over the course of training did `not further increase performance`.
### 5. Interaction Between Adam and Decoder Normalization
- Naive approach
  > gradient update w/ Adam &rarr; normalize decoder weight &rarr; erase length change
  
  Optimizer (Adam) did NOT see the true update to the weight matrix.
  
- Adam visible approach
  > discard the parallel component of the gradient &rarr; gradient (orthogonal) update w/ Adam
  
  The `parallel` component of the gradient with respect to the weight matrix `bring the weight matrix out of the unit sphere`, and the `orthogonal` component `rotate` the weight matrix.
  
$$g = \underset{parallel}{\underbrace{(g\cdot w)w}} + \underset{orthogonal}{\underbrace{g-(g\cdot w)w}}$$
$$w_{n+1} = w_n - \eta g_{\perp}$$
  
## Metrics to measure SAE ("far from perfect")
1. Training loss
2. Feature Density Histograms
3. $L^0$ norm (`interpretability` axis)
4. Reconstructed Transformer NLL (`behavioral fidelity` axis)
<div align=center><img src="/figures/2310.anthropic.04.png" style="height: 300px; width: auto;"/></div>
