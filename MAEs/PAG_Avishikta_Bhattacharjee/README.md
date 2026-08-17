# Physics-Aware Gating in Transformer for Jet Classification

<p align="center">
  <img width="788" alt="GSoC Banner" src="https://github.com/user-attachments/assets/c5c91a0b-a0f6-4a92-b992-45d3f5fefaba" />
</p>

This repository contains the implementation of a **Physics-Aware Gating on Lorentz Particle Transformer (LorentzParT)** designed for self-supervised pre-training and jet reconstruction on high-energy physics datasets like **JetClass**.

The core contribution of this work is a custom attention block that injects a **normalized global invariant mass bias ($m^2$)** alongside a **gated pairwise interaction matrix ($U$)** directly into the attention mechanism, enforcing fundamental Lorentz invariance and relativistic conservation laws.

---

## Description

The existing attention mechanism uses a bias matrix $U$ to incline the transformer toward physics constraints on jet particles, achieving a strong baseline (ROC AUC > 0.90). To evaluate and improve attention across various benchmarks, this repository introduces a **gating mechanism** controlled by physics attention heads to further reduce noise from jet constituent data.

### 1. Reused Physics Bias Attention Head for Gating
Rather than calculating static physical shifts independently at every Transformer layer, the model utilizes a dedicated **Physics Bias Attention Head**. This module extracts the pairwise $U$-matrix—computed from Minkowski inner products—and fuses it with the global normalized invariant mass bias ($m^2$). 

This physical bias tensor is projected across key heads and **reused dynamically across encoder blocks** to act as a gating mask. By projecting and reusing these learned physical weights directly within the multi-head attention mechanism, the network modulates the query-key matrix ($QK^T$) before value aggregation, filtering out unphysical particle couplings and stabilizing training across dynamic batch shapes.

### 2. Global Invariant Mass Scalar ($m^2$)
For a jet of $N$ particles, the total invariant mass squared ($m^2$) is computed across aggregate energy and momentum sums:

$$m^2 = \left(\sum_{i=1}^{N} E_i\right)^2 - \left\| \sum_{i=1}^{N} \vec{p}_i \right\|_2^2$$

To ensure numerical stability across varying energy scales, $m^2$ is feature-scaled to form a **normalized invariant mass bias**.

---

## Architecture Overview

The model uses the encoder of LorentzParT embedded within a Variational Autoencoder (VAE) setup:
1. **Masking:** Random constituent particles are masked during training.
2. **Encoder:** Processes unmasked tokens through LorentzParT blocks augmented with the **Physics-Aware Gating (PAG)** layer.
3. **Decoder:** Reconstructs the 4 constituent properties ($p_T, \eta, \phi, E$) of the masked particles.

---

## Evaluation & Results

Will add in the final submission (yet to upload).
