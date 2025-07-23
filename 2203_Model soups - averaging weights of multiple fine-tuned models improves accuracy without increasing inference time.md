# Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time
> [2203.05482](https://arxiv.org/abs/2203.05482)<br>
> Model soups
<div align=center><img src="/figures/2203.05482.1.png" style="height: 200px; width: auto;"/></div>

## Motivation 
1. Ensemble methods improve accuracy but at high inference cost.
2. Traditional training procedure only picks the best model from a training trajectory, wasting information hidden among the rest of the checkpoints.

## Experiment

## Summary 
Averaging over FT models yields better generalization: 
1. Sort FT models (from different hyperparameters) by validation accuracy.
2. Iteratively add a model if it boosts the soup’s validation performance.

## Tech Insights 
1. When fine-tuned models reside in the same loss basin, `weight averaging is nearly as effective as ensembling`.
2. Averaging yields flatter minima and better uncertainty calibration in many cases.
3. High learning-rate outlier models can degrade uniform soups; greedy selection mitigates this.
