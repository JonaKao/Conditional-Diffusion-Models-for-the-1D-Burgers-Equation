# Conditional Diffusion Models for the 1D Burgers' Equation

**Jona Kao · Niclas Leinen**
Final project for *Generative Neural Networks for the Sciences*, Heidelberg University (Prof. U. Köthe), Winter Semester 2025/26.

This project asks whether a conditional denoising diffusion probabilistic model (DDPM) can learn the distribution of solutions to the 1D viscous Burgers' equation

$$\partial_t u + u\,\partial_x u = \nu\,\partial_{xx} u$$

for random initial conditions, and whether the generated solutions respect the known physics: shock formation, energy decay, momentum conservation and the PDE residual. Given an initial condition $u(x,0)$, the model generates the full space–time field $u(x,t)$.

Ground truth comes from the **Cole–Hopf transformation**, which gives exact analytic solutions. Every generated sample can therefore be checked against noise-free references, with no numerical-solver error in the comparison. The Cole–Hopf solution serves only as an evaluation oracle. The model never sees it at inference time.

## Repository contents

| File | Description |
|---|---|
| `GNNProjectCodeFinalVersionKaoLeinen.ipynb` | Complete pipeline in one notebook: data generation, model, training, sampling, evaluation, the five experiments, extensions and follow-up checks |
| `FinalReportGNN_JonaKao_NiclasLeinen.pdf` | Final report covering methods, results, ablations, the problems we hit during development, and conclusions |

## Key results

| Metric | Value |
|---|---|
| Relative L² error (mean / median) | 0.040 / 0.030 |
| Relative L² with classifier-free guidance (w = 2.0) | **0.029** (31% better) |
| Kinetic energy error | 0.6% |
| Shock position error | 0.024 rad (about ⅓ of a grid cell) |
| PDE residual (generated / analytic) | 0.543 / 0.458 (19% above the finite-difference floor) |
| Wasserstein-1 distance | 0.014 |
| Calibration (Spearman ρ, ensemble spread vs. error) | 0.51 |
| Momentum drift (generated / analytic) | 76× (the main weakness) |

For reference, a mean-field baseline that predicts the training-set average gives L² = 1.00.

**Main findings**
1. **Stochastic sampling is essential.** Deterministic DDIM (η = 0) fails badly on shock-dominated fields. Stochastic DDIM (η = 1) with 100 steps reaches almost full DDPM quality (L² = 0.048) at about 10× lower cost.
2. **Deterministic regression is more accurate but gives no uncertainty.** The same U-Net trained by direct regression reaches L² = 0.016, but it produces one answer per input and cannot quantify uncertainty.
3. **Generalisation across viscosities is asymmetric.** The model is trained at ν = 0.05. It works within roughly a factor of two of that value and degrades fastest at low viscosity, where shocks are sharper.
4. **Classifier-free guidance has a safe range.** Guidance up to w = 2 improves accuracy. From w ≥ 3 the PDE residual rises, and calibration drops steadily as w increases.
5. **The method is data-efficient.** At constant compute (about 18k optimiser steps), 500 training samples already match the accuracy of 4,000. More data mainly reduces the spread across random seeds.

Extensions in the report and notebook:
- a momentum-conservation penalty (λ = 0.01 cuts drift by 46% and also lowers L²)
- a calibration breakdown across guidance scale, viscosity and ensemble size
- an analysis of how deterministic DDIM trajectories drift
- a per-wavenumber spectral analysis
- split conformal prediction, whose intervals reach at least nominal coverage and are widest at shock fronts

## Method overview

**Data.** Initial conditions are random Fourier series with 10 modes and coefficients $a_k, b_k \sim \mathcal{N}(0, (0.7/k)^2)$, on a periodic domain $[0, 2\pi)$. Solutions are computed analytically with Cole–Hopf (in float64) on a grid of 128 spatial points × 32 time steps over $t \in [0, 1]$. Samples with $\max|u| > 10$ are rejected. This leaves 4,000 training and 500 test samples, which are globally standardised using training-set statistics.

**Model.** A 1D U-Net with about 4.93M parameters. The 32 time slices are treated as channels and the 128 spatial points as the sequence dimension, and the initial condition is added as a 33rd input channel. The encoder uses channels 64 → 128 → 256 → 256, with two residual blocks per level (GroupNorm and SiLU) and a sinusoidal diffusion-timestep embedding. Three additions go beyond a standard DDPM:
- **Self-attention** in the bottleneck (4 heads, 16 × 256), which captures long-range coupling between shock fronts
- an **exponential moving average (EMA)** of the weights, with decay 0.9995 and a 500-step warmup
- **classifier-free guidance**, trained with 5% condition dropout

**Training.** The noise-prediction objective uses a linear β schedule (10⁻⁴ to 0.02, T = 1000 steps). Optimisation uses AdamW (learning rate 10⁻⁴, weight decay 10⁻²) with cosine annealing and gradient clipping at 1.0, for 300 epochs at batch size 64. The seed is 42.

**Sampling.** Full 1000-step ancestral DDPM sampling with EMA weights. The intermediate estimate $\hat{x}_0$ is clipped to [−15, 15] in normalised space. DDIM sampling with adjustable η and step count is included for comparison.

**Evaluation.** Eight metrics in four groups:
- *accuracy:* relative L², shock position and shock width
- *physical consistency:* energy decay, PDE residual and momentum drift
- *distributional fidelity:* Fourier power spectrum and Wasserstein-1 distance
- *calibration:* Spearman ρ between ensemble spread and error

## Getting started

### Requirements

The original runs used Python 3 with PyTorch 2.1 and CUDA 12.1 on a single NVIDIA GPU.

```bash
pip install torch numpy scipy matplotlib tqdm jupyter
```

The notebook also runs on CPU (it selects the device automatically), but 1000-step DDPM sampling will be very slow.

### Running

```bash
git clone https://github.com/JonaKao/GNN-Final-Project-Report.git
cd GNN-Final-Project-Report
jupyter notebook GNNProjectCodeFinalVersionKaoLeinen.ipynb
```

Run the cells in order. Everything is computed in memory: data is generated on the fly, and no datasets or checkpoints are read from or written to disk. Later sections reuse objects from earlier cells (`model`, `ema`, `true_fields`, …), so they cannot be run on their own.

**Runtime.** Training the main model takes about 45 minutes on a single GPU. A 1000-step DDPM sample takes about 20 seconds. The experiment sections train further models (the deterministic baseline, data-efficiency subsets over 3 seeds each, and fine-tuning with the momentum penalty), so a full top-to-bottom run takes considerably longer.

All hyperparameters are defined in the first code cell (`NU`, `NX`, `NT`, `N_TRAIN`, `DIFFUSION_STEPS`, `N_EPOCHS`, …).

### Notebook structure

| Section | Content |
|---|---|
| 1 | Data generation: Cole–Hopf analytic solver and rejection sampling |
| 2 | Normalisation and visualisation |
| 3 | DDPM: noise schedule, U-Net with self-attention, EMA |
| 4 | Training |
| 5 | Sampling (ancestral DDPM with classifier-free guidance) |
| 6 | Evaluation: L², shocks, energy, profiles, PDE residual, ensemble spread, spectrum, Wasserstein-1, momentum, calibration |
| 7 | Experiments: (1) DDPM vs. DDIM sampling, (2) deterministic baseline, (3) viscosity sensitivity, (4) guidance-scale sweep, (5) training-set size |
| 8 | Summary |
| 9 | Extensions: momentum penalty, calibration breakdown, DDIM trajectories, per-wavenumber spectrum, split conformal prediction, synthesis figure. Follow-up checks A–D: η sweep, multi-seed evaluation, mean-field baseline, ν = 0.03 |

## Limitations

- The study covers only the 1D case with periodic boundary conditions.
- Training uses a single viscosity (ν = 0.05). Other viscosities are tested out of distribution only.
- All data is noise-free and synthetic.
- Momentum conservation is not enforced by the architecture, because pixel-space diffusion has no mechanism that keeps spatial integrals constant.
- Generated fields carry about twice the true high-frequency power (spectral ratio 1.89).

Possible next steps include conditioning on viscosity, generating in spectral space or projecting onto conserved quantities, and extending to 2D. Section 6.3 of the report discusses these.

## Citation

```bibtex
@misc{kao_leinen_2026_burgers_ddpm,
  title  = {Conditional Diffusion Models for Learning the Distribution of 1D Burgers' Equation Solutions},
  author = {Kao, Jona and Leinen, Niclas},
  year   = {2026},
  note   = {Final project report, Generative Neural Networks for the Sciences, Heidelberg University},
  url    = {https://github.com/JonaKao/GNN-Final-Project-Report}
}
```
