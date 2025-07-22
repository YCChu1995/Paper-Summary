# Averaging Weights Leads to Wider Optima and Better
> [1803.05407](https://arxiv.org/abs/1803.05407)<br>
> SWA (Stochastic Weight Averaging)
<div align=center><img src="/figures/1803.05407.1.png" style="height: 250px; width: auto;"/></div>

## Motivation 
1. Ensemble methods improve accuracy with a high inference cost.
2. Traditional training procedure only picks the best model from a training trajectory, wasting information hidden among the rest of the checkpoints.

## Summary 
Instead of picking the final weights of a model at the end of training, SWA computes a `running average of weights` sampled from the `tail of the training trajectory` (~>75%), typically from a cyclical or constant learning rate schedule.

## Tech Insights 
1. Use cyclical learning rate with SGD or Adam to `explore different modes of well-fine-tuned models in a low-loss basin`.
2. `Average different modes` of well-fine-tuned models leads to a flatter solution, which `correlates with better generalization`.
