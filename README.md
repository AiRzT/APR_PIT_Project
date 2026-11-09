# APR-PiT

English | [简体中文](README.zh-CN.md)

**APR-PiT** combines a physics-conditioned Transformer with residual-adaptive point refinement and pruning to reconstruct velocity, dynamic pressure, temperature, and smoke mass fraction on the longitudinal center plane of tunnel fires for ventilation assessment. This repository provides PyTorch code and workflows for training, prediction, and reference-field comparison.

All large data files used in this study are stored in the following cloud drives: [Google Drive](https://drive.google.com/drive/folders/1JGFw06BPNis-8mYIm7WMLB3A50Q7rRUH?usp=sharing) | [Baidu Netdisk](https://pan.baidu.com/s/1mh7BpiarOyjJi_IX-wVYUw) (extraction code: `d5ed`).

## Method described in the paper

### Reconstruction task and variables

The paper uses a quasi-two-dimensional model of the longitudinal center plane, with transverse velocity `v = 0` and transverse derivatives `∂/∂y = 0`. Equation (24) defines the mapping

$$
\mathcal{G}_{\theta_c}:(t,x,z,V^*)\longmapsto(u,w,\pi,T,C_s).
$$

Here, `u` and `w` are longitudinal and vertical velocity, `pi` is dynamic pressure, `T` is temperature, and `C_s` is smoke mass fraction. The case-level conditioning variable `V* = V / U_ref` represents forced longitudinal ventilation and is zero in the open-boundary benchmark. Automatic differentiation acts only on the space–time coordinates.

The benchmark covers a `200 m × 6 m` center plane over `0–120 s`. Space–time coordinates use fixed reference scales; the velocity, temperature-rise, and smoke-mass-fraction scales are `5 m/s`, `300 K`, and `1.0e-3`, respectively.

### Physics constraints and network architecture

- **Governing equations:** low-Mach variable-density mass, momentum, temperature, and smoke transport, with density given by `rho = p0 / (R T)`, Smagorinsky eddy viscosity, simplified volumetric radiation loss, and wall heat transfer.
- **Prescribed fire source:** a Gaussian volumetric heat source is normalized over the three-dimensional physical domain and evaluated on the center plane for the two-dimensional energy and smoke residuals.
- **Context encoding:** Fourier features and input embeddings enter the Transformer. Sparse attention combines local space–time windows with global high-residual anchors; its physics bias uses normalized residuals and alignment with the gravity axis.
- **Physics-conditioned decoding:** local mass-conservation, momentum-residual, and buoyancy indicators generate affine modulation coefficients. The modulated context is decoded into physical fields by an MLP, with an auxiliary decoder supplying latent-physics consistency loss.
- **Conditioning cache:** the paper uses residuals and local physics indicators from the preceding adaptive cycle as conditioning inputs for the current cycle. Training losses use governing-equation residuals from the current predictions.

The main architecture settings in Table 5 are:

| Module | Settings |
| --- | --- |
| Transformer | 4 layers, width 256, 8 heads (6 feature heads and 2 residual-correction heads), FFN width 1024 |
| Fourier encoding | 64 frequencies, 128 sine/cosine features, standard deviation 1.0 |
| Sparse attention | Local window 64, global anchors 32, switching threshold 4096 |
| Physics modulation | 2 hidden layers of width 128, Tanh; residual/buoyancy bias coefficients 0.5/0.1 |
| Field decoder | 4 hidden layers of width 256, Tanh |

### Training and case evaluation

The objective combines governing-equation, initial-condition, boundary-condition, and latent-physics consistency losses:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{PDE}}+100\mathcal{L}_{\mathrm{IC}}+100\mathcal{L}_{\mathrm{BC}}+0.1\mathcal{L}_{\mathrm{lat}},\qquad\mathcal{L}_{\mathrm{data}}=0.
$$

Collocation points are initialized by Latin-hypercube sampling, then refined and pruned using residuals. Sampling probabilities are proportional to squared normalized residuals, with a cap of 300,000 points. Adam optimization is followed by L-BFGS refinement; a fixed evaluation set makes losses and residuals comparable across adaptive cycles. FDS and experimental fields are used for validation after the checkpoint is frozen, and do not enter the training loss, point selection, or checkpoint selection.

The paper studies three configurations: a single fire with open portals, dual fires with longitudinal ventilation, and a single fire with longitudinal ventilation and localized ceiling extraction. Each source layout and boundary topology has a separately trained model. Each forced-ventilation checkpoint is trained at one fixed inlet velocity. For forced ventilation, the velocity lifting in Eq. (51) imposes the inlet velocity. Off-design evaluation freezes the weights within the same configuration, changes `V*` and the inlet lifting, and compares predictions with the corresponding reference data.

## Current code interface

The commands below use `configs/tunnel_2d.yaml`. The model accepts normalized space–time coordinates `(t, x, z)` and a physical inlet speed in m/s through `model(coordinates, inlet_velocity=V)`. It internally appends the constant condition `V* = V / scales.velocity` and returns physical `(u, w, pi, T, C_s)` fields. Training, PDE residuals, adaptive sampling, and prediction share this interface. `V*` is not an additional spatial axis: automatic differentiation, attention geometry, and collocation points retain their original space–time dimensions.

The supplied configurations enable `model.velocity_conditioning: true`; `model.input_dim` is 4 for 2D and 5 for 3D, while the coordinate tensor remains `N × 3` or `N × 4`. Both configurations default to `boundary.inlet_mode: open` and `boundary.inlet_velocity: 0.0`. Set `inlet_mode: velocity` with a nonnegative physical speed, or use the training option `--inlet-velocity`, to train with a prescribed upstream velocity.

For a prescribed inlet, the decoded physical velocity is lifted using the normalized streamwise coordinate `xi = x / length_x`:

$$
\hat u=V+\xi\,\tilde u_\theta,\qquad
\hat w=\xi\,\tilde w_\theta
\quad (\hat v=\xi\,\tilde v_\theta\text{ in 3D}).
$$

This imposes the inlet velocity exactly. PDE and boundary derivatives are evaluated on the lifted fields, including derivatives of `xi`. The inlet loss uses the same requested speed; it no longer penalizes that inlet toward zero velocity. Open mode uses the open-boundary pressure/normal-gradient operator at both portals and does not apply velocity lifting. Solid-wall no-slip conditions remain separate.

Boundary samples exclude intersecting edges/corners and `t = 0`; initial-condition samples lie at `t = 0` in the spatial interior. A nonzero, uniform inlet and no-slip walls have incompatible values at their intersection, and a suddenly applied inlet differs from the initially stationary interior at the initial inlet edge. These intersection points are excluded from the loss sets. A direct prediction at an inlet intersection follows the inlet lifting. No time ramp or inlet-profile smoothing is introduced.

Legacy checkpoints without `model.velocity_conditioning` retain their original coordinate inputs and load without a wind-speed override. To use `V*` and lifting, train a new conditioned model; changing an old checkpoint's configuration does not supply the missing learned input weights. An open-portal checkpoint cannot be switched to nonzero forced ventilation during evaluation.

The repository also contains `configs/tunnel_3d.yaml` and `scripts/evaluate_3d.py` for three-dimensional computational extensions. The paper's main neural reconstruction model is the two-dimensional center-plane model described above.

The code provides physics residuals, adaptive collocation, Adam–L-BFGS optimization, checkpoints, JSONL training histories, field export, and reference-error calculation. It supports CPU smoke tests and GPU training. Download the required data from the cloud drives above before running reference-field comparisons.

## Project layout

```text
APR_PIT_Project/
├── README.md                   # English overview (default)
├── README.zh-CN.md              # Chinese overview
├── requirements.txt            # Python dependencies
├── configs/tunnel_2d.yaml       # default two-dimensional training configuration
├── configs/tunnel_3d.yaml       # three-dimensional configuration
├── scripts/train.py             # Adam -> APR -> L-BFGS training
├── scripts/evaluate.py          # grid inference and field figures
├── scripts/evaluate_3d.py        # three-dimensional field evaluation
├── scripts/compare_reference.py # same-grid FDS/reference error metrics
├── scripts/plot_history.py      # loss-history figure
├── src/apr_pit/
│   ├── model.py                 # Fourier + sparse Transformer + decoder
│   ├── physics.py               # PDE, IC, and BC residuals
│   ├── sampling.py              # LHS pool and APR controller
│   ├── trainer.py               # optimization and checkpoints
│   ├── config.py
│   └── utils.py
└── tests/                      # core physics and velocity-conditioning tests
```

## Environment

A CUDA-enabled PyTorch installation is recommended for manuscript-scale training. CPU execution is sufficient for the smoke test and unit tests.

Clone the repository and enter its root directory:

```bash
git clone https://github.com/AiRzT/APR_PIT_Project.git
cd APR_PIT_Project
```

Create a dedicated Conda environment, for example:

```bash
conda create -n apr-pit python=3.10 -y
conda activate apr-pit
```

Install the required packages from the repository root:

```bash
python -m pip install -r requirements.txt
```

The requirements include `h5py` for HDF5 export in `scripts/evaluate_3d.py`.

For GPU training, install a PyTorch build that matches your CUDA version before installing the remaining requirements. See the official PyTorch installation instructions for the appropriate command for your workstation.

## Quick verification

From the repository root, run the unit tests:

```bash
python -m unittest discover -s tests -v
```

Run a smoke test with small point counts, two Adam updates, and one L-BFGS iteration:

```bash
MPLCONFIGDIR=/tmp/apr_pit_matplotlib \
python scripts/train.py \
  --smoke-test --device cpu --output outputs/tunnel_2d
```

The smoke-test checkpoint is written to:

```text
outputs/tunnel_2d_smoke/
```

## Training configuration

On a CUDA workstation, run:

```bash
python scripts/train.py \
  --config configs/tunnel_2d.yaml \
  --device cuda \
  --output outputs/tunnel_2d_full
```

The default two-dimensional configuration uses the following sampling and optimization settings from Table 5:

| Parameter | Value |
| --- | --- |
| Initial interior points | 150,000 |
| Boundary-condition points | 25,000 |
| Initial-condition points | 25,000 |
| Collocation cap | 300,000 |
| High-residual candidate region | Top 10% |
| Sampling exponent / pruning threshold | 2.0 / 0.05 |
| APR interval | Every 10,000 epochs |
| Adam epochs | 30,000 |
| Initial Adam learning rate | 1.0e-3, multiplied by 0.95 every 1,000 epochs |
| PDE / BC / IC batch sizes | 8,192 / 2,048 / 2,048 |
| Maximum L-BFGS iterations | 20,000 |
| Fixed residual-evaluation set | 50,000 points |
| Convergence thresholds | Relative loss change < 1.0e-4 and mean residual < 1.0e-3 for 3 consecutive cycles |

See `configs/tunnel_2d.yaml` for the full configuration. The training schedule and convergence criteria determine the stopping point.

The repository defaults to random seed `2026`; the paper uses seed `42` for PINN field maps and profile curves. Set and record the seed for the case being run.

## Field evaluation

Evaluate a trained checkpoint on a regular grid:

```bash
MPLCONFIGDIR=/tmp/apr_pit_matplotlib \
python scripts/evaluate.py \
  outputs/tunnel_2d_full/checkpoint_final.pt \
  --times 10 60 120 --device cuda --output outputs/tunnel_2d_full/evaluation
```

For each requested time, the script exports a compressed `.npz` field file and a white-background PNG containing velocity magnitude, temperature, and smoke mass fraction.

### Prescribed-inlet training and fixed-weight prediction

Train one conditioned model at `0.65 m/s`, then evaluate the same checkpoint at `0.30 m/s`:

```bash
python scripts/train.py --config configs/tunnel_2d.yaml --inlet-velocity 0.65 --device cuda --output outputs/inlet_v065
python scripts/evaluate.py outputs/inlet_v065/checkpoint_final.pt --inlet-velocity 0.30 --times 120 --device cuda --output outputs/inlet_v065/evaluation_v030
```

For a CPU workflow check, add `--smoke-test --device cpu` to training (replace `--device cuda`) and evaluate `outputs/inlet_v065_smoke/checkpoint_final.pt` with `--device cpu --nx 9 --nz 5`. Smoke tests check execution, not predictive accuracy.

The evaluation override changes both the internal condition and the inlet lifting, without invoking an optimizer or changing checkpoint weights. Omitting the override uses the checkpoint's configured speed. The same option is available in `scripts/evaluate_3d.py` for a model trained with `configs/tunnel_3d.yaml`. Field exports record `training_inlet_velocity_m_s`, `evaluation_inlet_velocity_m_s`, `training_v_star`, `evaluation_v_star`, and `inlet_mode` (NPZ entries in 2D; HDF5 attributes in 3D).

### Training-history plot

Plot the training history:

```bash
MPLCONFIGDIR=/tmp/apr_pit_matplotlib \
python scripts/plot_history.py \
  outputs/tunnel_2d_full/history.jsonl
```

## Reference-field comparison

After downloading the reference data and aligning its grid and time with the prediction, run:

```bash
python scripts/compare_reference.py \
  outputs/tunnel_2d_full/evaluation/fields_t120s.npz \
  data/fds_fields_t120s.npz \
  --fields temperature smoke
```

`data/fds_fields_t120s.npz` is an example path; replace it with the actual path to the downloaded and prepared reference file. The reference `.npz` file must use the field names expected by `scripts/compare_reference.py` and match the prediction's array shapes, grid, time, and physical units. Check coordinates and perform any required grid interpolation before comparison.

## Notes on reproducibility

The reproduction workflow covers environment setup, physics-informed training, adaptive collocation updates, field export, and reference-field error calculation.

The implemented velocity condition and inlet lifting close the corresponding code-interface gap. They do not by themselves establish reproduction of manuscript Tables 6–15 or accuracy at unseen velocities. Those results require the matching case definitions, reference data, trained checkpoints, and table-specific evaluations.

Record the hardware, PyTorch and CUDA versions, random seed, and optimizer configuration for each run, and retain the model checkpoints, training history, and evaluation results.

## Citation

If you use this code, please cite the associated manuscript:

```text
Citation information will be added after publication.
```

## License

This project is released under the MIT License. See `LICENSE` for details.

