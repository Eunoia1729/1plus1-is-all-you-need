# 1 + 1 Is All You Need

A from-scratch JAX addition transformer, its Chinchilla scaling laws (dense vs MoE), and a fused MoE kernel in Pallas.

Exercise from https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html

~10M-parameter transformer in JAX + Flax + Optax that learns addition (up to 3-digit operands).
Vocab: digits 0-9, space, `+`, `=`; fixed-length padding.

## Tasks
- [ ] Train the addition transformer (fast on a T4 / free Colab TPU)
- [ ] Derive Chinchilla scaling laws; compare dense vs MoE
- [ ] Fused MoE kernel in Pallas that beats `jax.lax.ragged_dot` when F > D

## Layout
- `notebooks/` the from-scratch transformer notebook
