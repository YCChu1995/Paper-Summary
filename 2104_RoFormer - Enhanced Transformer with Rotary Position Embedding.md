# RoFormer: Enhanced Transformer with Rotary Position Embedding
> [2104.09864](https://arxiv.org/abs/2104.09864)<br>
> RoPE
<div align=center><img src="/figures/2104.09864.01.png" style="height: 250px; width: auto;"/></div>

## Summary 
- RoPE is a `multiplicative` positional encoding with `NO trainable` parameters, which maintains norms of representations (stable and efficient).
- The rotary matrix $R_{\theta, m}^d$ is multiplied to query and key on the `embedding dimension`.
  > Unlike traditional absolute positional encoding, which is added to the token embedding on the `sequence dimension`.<br>
  > I don't know why the rotary matrix cannot use a uniform angular frequency since it is across the `embedding dimension` rather than `sequence dimension`, the positional information will still be captured via the (m-n) term.<br>
  > Possible answers: [2410.06205](https://arxiv.org/abs/2410.06205)

---

## Motivation 
- Positional encoding had become a bottleneck for longer context modeling because:
  1. Fixed positional encoding limit longest context.
  2. Additive absolute encoding often fails to generalize beyond trained lengths.
  3. [Relative position biases](https://github.com/YCChu1995/Paper-Summary/blob/main/1803_Self-Attention%20with%20Relative%20Position%20Representations.md) improved some tasks but not others.
  4. Linear attention models (Performer, etc.) faced challenges encoding positions because `additive biases` don’t propagate through kernel approximations.

## Chain of Thoughts
### 1. Hypothesis - Position Factor Concept
The authors argue that `relative position` is the fundamental concept for language modeling, not absolute position.<br>
In self-attention, the true position factor appears only as a `relative offset between query and key`, i.e., (m–n).<br>

### 2. Target Formulation for Attention Matrix
The previous hypothesis leads to the following target formulation for the attention matrix, which `only relies` on `contents` ($x_{m}, x_{n}$) and the `relative position` ($m-n$). 

$$ \left< f_{q}(x_{m}, m), f_{k}(x_{n}, n) \right> = g(x_{m}, x_{n}, m-n)$$

### 3. RoPE (Rotary Position Embedding)
The solution to fit the target formulation:

$$f_q (x_m, m)=(W_q x_m)e^{im\theta}$$
$$f_k (x_n, n)=(W_k x_n)e^{in\theta}$$
$$g(x_m, x_n, m-n)=Re[(W_q x_m)(W_k x_n)^{\dagger}e^{i(m-n)\theta}]$$

Simplified:

$$q_m^T k_n = (R_{\theta, m}^d W_q x_m)^T(R_{\theta, n}^d W_k x_n) = x^T W_q^T R_{\theta, n-m}^d W_k x_n$$ 
