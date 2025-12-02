# GLU Variants Improve Transformer
> [2002.05202](https://arxiv.org/abs/2002.05202)<br>
> GEGLU, SwiGLU
<div align=center><img src="/figures/2002.05202.T1.png" style="height: 150px; width: auto;"/></div>

## Summary 
GEGLU ~ SwiGLU > ReLU (Baseline)

---

## Motivation 
The Transformer FFN had a very standard shape (linear → nonlinearity → linear).<br>
The community was exploring alternative activations (GELU, Swish) and structural tweaks but it was not clear `which small architectural changes would yield reliable improvements` across pretraining + finetuning regimes.<br>
`This paper tests the hypothesis` that `gated/bilinear intermediate layers` are a simple, effective improvement.
