# The Eiffel Tower Llama
> [2511.HuggingFace](https://huggingface.co/spaces/dlouapre/eiffel-tower-llama)
<div align=center><img src="/figures/2511.HuggingFace.01.png" style="height: 300px; width: auto;"/></div>

### 3. Steering and generation improvements
#### 3.1 Improving directions
- `Clamping` to ensure `consistent activations`
- `Repetition penalty` during generation to prevent the `gibberish` mode.
#### 3.2 Clamping


## Summary 
1. The steering coefficient 'sweet spot' is narrow. ([source](#23-results-of-a-1d-grid-search-sweep))

## Tech Insights 
1. The norm of residual activations increases as the layer index increases as well. ([source](https://github.com/YCChu1995/Paper-Summary/new/main#21-range-of-residual-activations))
   > The `optimal steering strength` (8.5) is of the order of `half the magnitude` of a `layer’s typical activation` ($15^th$ layer).<br>
   > This is consistent with the idea that steering vectors should not overwhelm the model’s natural activations.<br>
   > In the case of our feature, this is about `twice` the `maximum activation` observed in the training dataset (4.77).<br>
   > However, there is only a very narrow region leading to the best harmonic mean of LLM-judge metrics
2. `Concept inclusion` comes at the `cost` of `instruction following` and `fluency`.
   - `Concept inclusion` scores increase when impairing both `instruction following` score and `fluency` score. ([graph](#23-results-of-a-1d-grid-search-sweep))
   - LLM `instruction following` and `fluency` are highly `correlated` , but `anticorrelated` with `concept inclusion`. ([graph](#25-correlations-between-metrics))
---

## Motivation 
There are different statements about model steering with SAE, this paper try to reconcile these,
- The AxBench [paper](https://arxiv.org/abs/2501.17148) shows that even at `SAE` scale, representation steering is still `far behind` `simple prompting` and `fine-tuning` baselines.
- Golden Gate Claude demo is impressive

## Chain of Thoughts
### 1. Metrics
#### 1.1 LLM-as-a-judge
- concept inclusion
- instruction following
- fluency
#### 1.2 Auxiliary quantitative metrics
- Surprise
  > The negative log probability (per token) under the reference model, which represents the surprise in the reference model.<br>
  > (This is also essentially the `cross-entropy` between the output distribution of the `steered model` and the `reference model`.)
- n-gram repetition
  > The fraction of unique n-grams in the answers.<br>
  > Using `n=3` already leads to interesting insights, as it captures repetitions of words and short phrases.<br>
  > A value of 0.0 means that there is no repetition at all. For short answers, values above `0.2` tend to correspond to `problematic repetitions` that impair fluency.
  <div align=center>$$rep3 = \frac{N_{rep-3-grams}}{N_{3-grams}}$$</div>
  
  - $N_{rep-3-grams}$ : The count of 3-grams that appear more than once in the response.
  - $N_{3-grams}$ : The total count of 3-grams (including unique and repeated) in the response (=L-2).​
- Explicit concept inclusion
  > The `occurrence` of the exact word 'eiffel' in the answer (case-insensitive).<br>
  > This is a very crude metric, so it will not use it beyond simple monitoring.

### 2. Optimizing steering coefficient for a single feature
#### 2.1 Range of residual activations
<div align=center><img src="/figures/2511.HuggingFace.02.png" style="height: 250px; width: auto;"/></div>

#### 2.2 Range of steering coefficients
- To `avoid` completely `disrupting` the activations during steering, we expect the magnitude of the added vector to be at most of the order of the norm of the typical activation,
  <div align=center>$$\left|\left| \alpha \cdot v \right|\right|_2 ≲ \left|\left| x^l \right|\right|_2$$</div>  

#### 2.3 Results of a 1D grid search sweep
- From the harmonic mean score, we can see that the steering coefficient 'sweet spot' is `narrow`.
- `Concept inclusion` comes at the `cost` of `instruction following` and `fluency`.
   > `Concept inclusion` scores increase when impairing both `instruction following` score and `fluency` score.
<div align=center><img src="/figures/2511.HuggingFace.03.png" style="height: auto; width: 400px;"/> <img src="/figures/2511.HuggingFace.04.png" style="height: auto; width: 400px;"/></div>

#### 2.4 Detailed evaluation for the best steering coefficient (8.5)
- The `baseline prompted model` significantly `outperforms` the `steered model`.
<div align=center><img src="/figures/2511.HuggingFace.05.png" style="height: 450px; width: auto;"/></div>

#### 2.5 Correlations between metrics
- This analysis shows that although the `LLM-as-a-judge metrics` are the `most reliable`, the `auxiliary metrics` can provide `useful information` about the quality of the answers.
  - LLM `instruction following` and `fluency` are highly `correlated` , but `anticorrelated` with `concept inclusion`, showing the `tradeoff` between `steering strength` and `answer quality`.
  - `Repetition` metric is `strongly anticorrelated` with `fluency` and `instruction following`.
<div align=center><img src="/figures/2511.HuggingFace.06.png" style="height: 450px; width: auto;"/></div>
