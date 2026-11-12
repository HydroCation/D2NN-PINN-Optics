# Physics-Informed Diffractive Deep Neural Network (D2NN) for Optical 2D-DFT

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA Enabled](https://img.shields.io/badge/CUDA-12.1%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 22 Passed](https://img.shields.io/badge/Tests-22%20Passed-brightgreen.svg)]()

A production-grade, physics-informed Scientific Machine Learning (SciML) framework implementing cascaded **Diffractive Deep Neural Networks (D2NN)** for physical **2D Discrete Fourier Transform (2D-DFT)** optical synthesis in the terahertz / millimeter-wave regime ($\lambda = 0.75\,\text{mm}$, $f = 0.4\,\text{THz}$).

---

## Table of Contents

- [Physics-Informed Architecture](#physics-informed-architecture)
- [Modulation Regimes & Models](#modulation-regimes--models)
- [Repository Structure](#repository-structure)
- [Local Setup & Testing](#local-setup--testing)
- [Docker & Containerized Execution](#docker--containerized-execution)
- [Makefile Automation Harness](#makefile-automation-harness)
- [CLI Reference & Parameter Tuning](#cli-reference--parameter-tuning)
- [Multi-Regime Benchmarking Suite](#multi-regime-benchmarking-suite)
- [Empirical Benchmark Results](#empirical-benchmark-results)

---

## Physics-Informed Architecture

The optical forward pass models free-space coherent scalar wave propagation according to Rayleigh-Sommerfeld diffraction formulated via the exact band-limited **Angular Spectrum Method (ASM)**:

$$H(f_x, f_y; d) = \begin{cases} \exp\left(j \frac{2\pi d}{\lambda}\sqrt{1 - (\lambda f_x)^2 - (\lambda f_y)^2}\right), & f_x^2 + f_y^2 \le \frac{1}{\lambda^2} \\ 0, & \text{evanescent cutoff} \end{cases}$$

Wave propagation across $K$ cascaded diffractive plates separated by axial drift distance $d$:
$$u_0(x, y) = \text{Embed}_{\text{FOV}}\{u_{\text{in}}(x, y)\}$$
$$u_k(x, y) = \mathcal{P}_d\{u_{k-1}(x, y)\} \odot T_k(x, y), \quad k = 1, \dots, K$$
$$u_{\text{out}}(x, y) = \text{Extract}_{\text{FOV}}\{\mathcal{P}_d\{u_K(x, y)\}\}$$

### Nominal Physical Parameters
- **Operating Wavelength**: $\lambda = 0.75\,\text{mm}$ ($0.4\,\text{THz}$)
- **Drift Distance**: $d = 14\lambda = 10.5\,\text{mm}$
- **Number of Diffractive Layers**: $K = 8$
- **Sampling Pitch**: $\Delta x = \lambda / 2 = 0.375\,\text{mm}$ (Nyquist critical sampling)
- **Computational Grid**: $N = 128 \times 128$ ($48\,\text{mm} \times 48\,\text{mm}$)
- **Active Field of View (FOV)**: Centered $8 \times 8$ sub-grid ($3.0\,\text{mm} \times 3.0\,\text{mm}$)
- **Time-of-Flight Latency**: $\tau = \frac{(K+1)d}{c_0} = \frac{9 \times 0.0105\,\text{m}}{3 \times 10^8\,\text{m/s}} = 0.315\,\text{ns}$

---

## Modulation Regimes & Models

The framework provides three distinct diffractive modulation architectures:

1. **Model A: Phase-Only Stack (`--model-type phase_only`)**
   - Transmittance: $T_k(x, y) = \exp(j \phi_k(x, y))$ with trainable phase $\phi_k \in [-\pi, \pi)$ and unit amplitude $|T_k| = 1$.
   - Configuration: `configs/default.yaml`

2. **Model B: Complex Modulation (`--model-type complex`)**
   - Transmittance: $T_k(x, y) = \sigma(a_k(x, y)) \cdot \exp(j \phi_k(x, y))$, where $a_k$ is raw trainable amplitude initialized to $2.0$ ($\sigma(2.0) \approx 0.88$, strictly passive in $(0, 1)$).
   - Configuration: `configs/complex_modulation.yaml`

3. **Model C: Apertured Phase Stack (`--model-type apertured`)**
   - Transmittance: Phase modulation combined with selective super-Gaussian pupil beam confinement at the entrance (layer 0), midpoint (layer 3), and exit (layer 7):
     $$M_{\text{pupil}}(r) = \exp\left(-\left(\frac{r}{R_{\text{aperture}}}\right)^{2 \cdot \text{order}}\right)$$
   - Prevents numerical circular boundary wrap-around without mode truncation attenuation.
   - Configuration: `configs/apertured_phase.yaml`

---

## Repository Structure

```
├── configs/
│   ├── default.yaml                 # Model A (Phase-only co-optimization)
│   ├── complex_modulation.yaml      # Model B (Trainable amplitude + phase)
│   └── apertured_phase.yaml         # Model C (Apertured cascade)
├── src/
│   ├── __init__.py
│   ├── models.py                    # DiffractiveLayer, OpticalD2NN, spatial apertures
│   ├── propagation.py               # AngularSpectrumPropagator wave kernel
│   ├── metrics.py                   # Complex alignment, Frobenius error, PSNR, SSIM
│   ├── logger.py                    # Dataclass RunConfig, atomic checkpointing & tracking
│   ├── data.py                      # MNIST streaming loader & procedural geometric fallback
│   └── evaluate.py                  # Multi-regime evaluation engine (Harmonic, Image, Stress)
├── scripts/
│   └── eval_all.py                  # Turnkey batch evaluator across all models
├── tests/
│   ├── test_models.py               # Model forward, backward gradient, aperture tests
│   ├── test_propagation.py          # ASM transfer function & energy conservation
│   ├── test_metrics.py              # Complex cosine similarity, alignment, latency
│   ├── test_logger.py               # Atomic serialization & logging unit tests
│   └── test_evaluate.py             # Multi-regime benchmarking test harness
├── outputs/
│   ├── figures/                     # Mirrored publication-quality evaluation figures
│   └── runs/                        # Checkpoints, metric CSVs, summaries per experiment
├── Dockerfile                       # Production CUDA 12.1 runtime container
├── docker-compose.yml               # Service runner mounting host outputs without root locks
├── Makefile                         # Unified execution dispatcher
├── requirements.txt                 # Pinned Python package dependencies
├── pytest.ini                       # Pytest testpath & pythonpath configuration
└── run_experiment.py                # Main training CLI entrypoint
```

---

## Local Setup & Testing

### 1. Prerequisites
- Linux / WSL 2 with Python 3.10+
- NVIDIA GPU with CUDA drivers (optional, automatic CPU fallback supported)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/organization/optics-paper.git
cd optics-paper

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Unit Test Suite
```bash
# Execute the full 22-test regression suite
pytest -v
# or via Makefile
make test
```

---

## Docker & Containerized Execution

The repository provides a production `Dockerfile` based on `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` configured for headless, non-interactive rendering with unprivileged user output mounting.

### Build Container Image
```bash
docker build -t d2nn-sciml:latest .
# or
make docker-build
```

### Run Tests in Docker
```bash
docker run --rm --entrypoint pytest d2nn-sciml:latest -v
# or
make docker-test
```

### One-Command Training in Docker
Run training with the host's `./outputs` mounted so all checkpoints, CSV logs, and figures persist directly to the host without root permission locks:
```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/outputs:/workspace/outputs" \
  d2nn-sciml:latest \
  --config configs/apertured_phase.yaml \
  --model-type apertured \
  --epochs 2500 \
  --tag docker_model_c
```

### One-Command Evaluation in Docker
```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/outputs:/workspace/outputs" \
  --entrypoint python \
  d2nn-sciml:latest \
  -m src.evaluate \
  --checkpoint outputs/runs/run_20261112_125353_model_c_apertured_refined/phases_final.pt \
  --config configs/apertured_phase.yaml \
  --export-figures
```

### Turnkey Docker Compose
Launch training via Docker Compose:
```bash
docker compose up --build
```

---

## Makefile Automation Harness

The root `Makefile` provides one-line targets for standard workflows:

| Target | Description |
| :--- | :--- |
| `make test` | Run all 22 unit tests with verbose logging (`pytest -v`) |
| `make docker-build` | Build container image `d2nn-sciml:latest` |
| `make docker-test` | Run unit tests inside the built Docker image |
| `make train-model-a` | Train Model A (Phase-Only Baseline) with deterministic seed 42 |
| `make train-model-b` | Train Model B (Complex Modulation) with bounded transmission |
| `make train-model-c` | Train Model C (Apertured Cascade with 3-plane pupil filtering) |
| `make eval-all` | Batch evaluate latest checkpoints for Models A, B, and C with figure mirroring |
| `make docker-compose-up` | Launch containerized training with host output mounts |
| `make clean` | Remove `__pycache__`, `.pytest_cache`, and bytecode artifacts |

---

## CLI Reference & Parameter Tuning

### Training Entrypoint (`run_experiment.py`)

```bash
python run_experiment.py [OPTIONS]
```

#### Supported Arguments:
- `--config <path>`: Path to YAML or JSON config (default: `configs/default.yaml`).
- `--model-type {phase_only, complex, apertured}`: Modulation architecture regime (default: `phase_only`).
- `--loss-mode {balanced, exponential, kulce}`: Optimization loss formulation:
  - `balanced`: Logarithmic attraction penalty toward `--target-eff` ($-\beta \log(\eta / \eta_0)$).
  - `exponential` / `kulce`: Strict exponential barrier ($c_M L_{\text{fid}} + c_D \exp(-\alpha \eta)$).
- `--epochs <int>`: Total training iterations (default: 2500).
- `--batch-size <int>`: Stochastic wavefront batch size (default: 64).
- `--lr <float>`: Initial learning rate for Adam optimizer (default: 0.015).
- `--target-eff <float>`: Target active FOV efficiency (e.g. `0.035` for 3.5%, `0.012` for 1.2%).
- `--barrier-weight <float>`: Coefficient $c_D$ for the efficiency loss penalty.
- `--device <str>`: Computing device (`cuda`, `cpu`, or device index).
- `--tag <str>`: Unique run name tag for experiment directory naming.
- `--seed <int>`: Deterministic seed across NumPy, PyTorch, and CUDA (default: 42).
- `--quick-test`: Rapid 10-epoch sanity test flag.

#### Example Training Commands:
```bash
# Model A: Phase-only co-optimization
python run_experiment.py --config configs/default.yaml --model-type phase_only --epochs 2500 --tag model_a_phase_only --seed 42

# Model B: Complex modulation (absorption penalty target_eff=1.5%)
python run_experiment.py --config configs/complex_modulation.yaml --model-type complex --epochs 2500 --tag model_b_complex --seed 42

# Model C: Apertured 3-plane cascade (target_eff=1.2%)
python run_experiment.py --config configs/apertured_phase.yaml --model-type apertured --epochs 2500 --tag model_c_apertured_refined --seed 42
```

---

### Evaluation Entrypoint (`src.evaluate`)

```bash
python -m src.evaluate --checkpoint <path> [OPTIONS]
```

#### Supported Arguments:
- `--checkpoint <path>`: **(Required)** Path to trained PyTorch weights file (`phases_final.pt`).
- `--config <path>`: Optional YAML/JSON config. If omitted, automatically inherits the configuration embedded in the checkpoint or adjacent `config.json`.
- `--output-dir <path>`: Target directory for plots and JSON results (defaults to the checkpoint's parent run directory).
- `--export-figures`: Mirror generated publication figures to `outputs/figures/`.
- `--n-samples <int>`: Test images evaluated in the Image Dataset Benchmark (default: 1000).
- `--force-synthetic`: Force procedural geometric digit synthesis even if torchvision MNIST is available.
- `--device <str>`: Explicit device override (`cuda` or `cpu`).

---

## Multi-Regime Benchmarking Suite

The evaluation suite validates the learned physical operator across three rigorous regimes:

1. **Regime 1: Harmonic Plane Wave & Impulse Benchmark**
   - Discrete 2D plane waves $u(x, y) = \exp(j 2\pi(u_0 x + v_0 y)/8)$ across all spatial frequencies $(u_0, v_0) \in [0, 7]^2$.
   - Kronecker impulses $\delta(x - x_0, y - y_0)$ and 2D spatial Gaussians.
   - Computes: Spectral Peak Localization Accuracy (%), Mean Peak-to-Sidelobe Ratio (PSLR in dB), and Field Fidelity (%).
   - Generates: `harmonic_spectra_comparison.png`.

2. **Regime 2: Image Dataset Benchmark (MNIST & Procedural Digits)**
   - 1,000 test images with optimal complex scaling alignment constant:
     $$c = \frac{\langle \hat{A}' i, A i \rangle}{\|\hat{A}' i\|^2}$$
   - Evaluates: Mean Complex Field Fidelity (%), PSNR (dB), SSIM, and Active Optical Efficiency (%).
   - Generates: `mnist_optical_fft_gallery.png`.

3. **Regime 3: Robustness & Fabrication Stress Benchmark**
   - **Phase Quantization**: Bit depths $b \in \{2, 3, 4, 5, 8\}$ bits simulating lithographic discrete height levels:
     $$\phi_q = \Delta \cdot \text{round}\left(\frac{\phi}{\Delta}\right), \quad \Delta = \frac{2\pi}{2^b}$$
   - **Phase Noise Injection**: Zero-mean Gaussian phase noise $\sigma_{\phi} \in \{0.01, 0.05, 0.10, 0.20\}\,\text{rad}$ across multiple stochastic trials.
   - Generates: `robustness_ablation.png`.

---

## Empirical Benchmark Results

Evaluated across the latest checkpoints for all three architectures:

| Evaluation Metric | Model A (Phase-Only) | Model B (Complex Modulation) | Model C (Apertured Cascade) |
| :--- | :---: | :---: | :---: |
| **Transfer Matrix Similarity** | **92.57%** | **95.20%** | **87.77%** |
| **Transfer Matrix Frobenius Error** | 37.83% | 30.63% | 47.93% |
| **Active Energy Efficiency** | **2.21%** | **1.03%** | **0.65%** |
| **Spectral Peak Localization** | **100.00%** | **100.00%** | **100.00%** |
| **Peak-to-Sidelobe Ratio (PSLR)** | 17.26 dB | **20.51 dB** | 12.61 dB |
| **Harmonic Field Fidelity** | 92.55% | **95.20%** | 87.06% |
| **MNIST 1,000 Image Fidelity** | 79.02% | **83.54%** | 75.50% |
| **Image PSNR** | 29.41 dB | **30.35 dB** | 29.40 dB |
| **Image SSIM** | 0.5590 | **0.6544** | 0.5360 |
| **8-bit Quantized Fidelity** | 93.00% | **95.58%** | 87.42% |
| **4-bit Quantized Fidelity** | 91.80% | **94.86%** | 86.95% |
| **2-bit Quantized Fidelity** | 73.56% | 67.78% | 62.12% |
| **Phase Noise ($\sigma = 0.20\,\text{rad}$)** | 91.02% | **93.81%** | 86.09% |
| **Time-of-Flight Latency** | **0.315 ns** | **0.315 ns** | **0.315 ns** |
