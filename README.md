# BLEA Motion Library — BML-v1

**Multi-Actor Sport and Cultural Motion Library for GR00T-Compatible Humanoid Imitation**

> Preprint: [arXiv: TODO:xxxx.xxxxx]  
> Repository: https://github.com/crisguycabs-84/blea-udi  
> Unitree G1 · 29 DOF · GR00T-SONIC · IsaacLab v2.3.2

---

## Overview

BML-v1 is a curated, GR00T-SONIC-compatible motion library of sport and cultural movements for the Unitree G1 humanoid (29 DOF). Motions are recorded with multiple actors at four speed levels and delivered in the `motionlib` PKL format natively consumed by NVIDIA's GR00T-SONIC framework.

**Contributions:**
- **C1 — Motion dataset.** Multi-actor, multi-speed PKL library covering sport and cultural skills in GR00T-SONIC `motionlib` format.
- **C2 — IL↔MJ permutation pipeline.** Derivation, implementation, and validation of the critical 29-DOF joint permutation between Isaac Lab and MuJoCo orderings. Absent or misapplied, this produces arm-spasticity artifacts on physical hardware.
- **C3 — Learnability study and ablation.** Policy training results in IsaacLab-SONIC per skill, plus an actor/speed ablation quantifying training-data diversity effects.
- **C4 — Jetson JP5.1 adaptation.** Reproducible engineering stack for deploying GR00T-SONIC on Jetson Orin NX JP5.1 / TRT 8.5.2 (commercial G1 units shipped prior to 2026).
- **C5 — Sim-to-real validation protocol.** Gated runbook with physical deployment evidence on a Unitree G1.

---

## Motion Library

### Skills

| Skill | Category | Actors | Speeds | PKL status | W&B run |
|---|---|---|---|---|---|
| Saludo Militar | Functional gesture | 2 (cris, edson) | 4 | ✅ 8 PKLs confirmed | `2awai2u3` |
| Macarena | Cultural / dance | 1 (cris) | 4 | ✅ 4 PKLs confirmed | `9q7w1178` |
| Taekwondo | Sport | 2 | 4 | 🔲 PKL count pending | — |
| Fútbol | Sport | 2 | 4 | 🔲 PKL count pending | — |
| Lenguaje de Señas | Communicative | — | — | 🔲 Pending (hand-open / hand-closed, adapted to G1 finger DOF) | — |

Speed levels: `lento` · `normal` · `rápido` · `sostenido`

### PKL Schema

Both `motionlib` (training) and `vis` (deploy) PKLs share the same top-level structure:

```python
{
    "skill_name": {
        "dof":               np.ndarray,  # shape (T, 29)
        "root_trans_offset": np.ndarray,  # shape (T, 3)
        "root_rot":          np.ndarray,  # shape (T, 4), wxyz quaternion
        "fps":               50
    }
}
```

Load with `joblib.load(path)`. Do **not** use `pickle.load` — PKLs are joblib-serialised.

---

## Pipeline Architecture

> ⚠️ **The pipeline forks at §6. The two branches are incompatible.** Feeding a `motionlib` PKL (MJ order) to the deploy binary produces scrambled joint trajectories. See [IL↔MJ Permutation](#ilmj-permutation) below.

```
Raw capture (MuJoCo qpos logs)
        │
        ▼
   §6 — PKL generation
        │
        ├──► (A) DEPLOY BRANCH ──────────────────────────────────────►
        │         §6.1  deploy PKL  (dof in MJ order, ~960 frames)
        │         §7.1  vis PKL     (dof pre-applies MJ_TO_IL → IL order, ~9353 frames)
        │         §7.2  convert_motions.py → reference/{SKILL}/{SKILL}/*.csv
        │                                                              │
        │                                                              ▼
        │                                               g1_deploy_onnx_ref binary
        │                                               (applies IL→MJ at runtime)
        │
        └──► (B) TRAINING BRANCH ───────────────────────────────────►
                  §6.3  motionlib PKL  (dof in MJ order)
                  robot_filtered_motionlib_*/
                                                                       │
                                                                       ▼
                                                        IsaacLab / GR00T-SONIC training
```

**Dead end:** `motionlib_derived/` atomic PKLs are intermediate artefacts — never use them as a source for deploy CSVs.

---

## IL↔MJ Permutation

This is the **most common source of correctness failures** in the pipeline.

```python
IL_TO_MJ = np.array([
     0,  3,  6,  9, 13, 17,
     1,  4,  7, 10, 14, 18,
     2,  5,  8, 11, 15, 19,
    21, 23, 25, 27, 12, 16,
    20, 22, 24, 26, 28
])
MJ_TO_IL = np.argsort(IL_TO_MJ)
```

| PKL type | `dof` ordering | Used by |
|---|---|---|
| `vis` / deploy PKL | IL order — pre-applies `dof_mj[:, MJ_TO_IL]` | `g1_deploy_onnx_ref` (binary applies IL→MJ internally) |
| `motionlib` PKL | MJ order | IsaacLab motion-library loader |

**Gate at §7.1 close:** replay the vis PKL in MuJoCo with `scene_29dof_freebase.xml`, gravity disabled, free-floating base. Visually confirm limb positions match the recorded motion before any physical deployment.

---

## Learnability Results (W&B: `TRL_G1_Track`, entity `crisguycabs-udi`)

| Skill | Run ID | FR₀ | FR₁₀₀₀ | Reduction |
|---|---|---|---|---|
| Saludo Militar | `2awai2u3` | 0.1880 | 0.0365 | 80.6 % |
| Macarena (macarena2) | `9q7w1178` | — | 0.0218 | 97.8 % |

Metric: `adp_samp/failure_rate_mean`. Training: PPO, 1,000 iterations, default SONIC hyperparameters, RTX 5060 Ti (16 GB VRAM), IsaacLab v2.3.2, Ubuntu 26.04, CUDA 12.x.

### Actor/Speed Ablation — Saludo Militar

| Condition | W&B ID | Timeout rate | Notes |
|---|---|---|---|
| Reference (SONIC default) | `5wuntqtr` | — | Baseline |
| `only_cris` | `sll7o0m5` | — | Best arm precision; foot instability |
| `only_edson` | `h2lqaqmw` | — | Non-convergent |
| `both_clean` ✅ | `xd28vibk` | 0.73–0.75 | **Selected for sim-to-real** |

`both_clean` (multi-actor, quality-filtered) achieves the best balance between arm precision and whole-body stability.

---

## Jetson JP5.1 Adaptation

Commercial G1 units shipped prior to 2026 carry **JetPack 5.1.1 (R35.3.1) + TRT 8.5.2**, for which no official SONIC migration path exists. The following patches enable deployment on this stack:

| Patch | Detail |
|---|---|
| ONNX opset-14 conversion | 44 `LayerNormalization` nodes decomposed into primitives via `opset14_converter.py` → `planner_sonic_opset14.onnx` (773.7 MB) |
| Constant-folding graph surgery | Polygraphy CF: `TopK` K promoted to initialiser; 8,284 → 3,750 nodes → `planner_sonic_opset14_cf2.onnx` |
| Einsum ellipsis patch | Replaced with explicit `Transpose([0,1,2,4,3]) + MatMul` |
| `InferenceEngine.cpp` | Comment out `kINT64`/`kBF16`/`kFP8`; `setShapeValuesV2` → `setShapeValues`; cast shape tensors to `int32_t`; `0U` → `1U << kEXPLICIT_BATCH` |
| Compiler | Install `gcc-10/g++-10`; pass to CMake explicitly |
| cppzmq | Replace system 4.3.1 headers with 4.9.0 (requires `zmq::sockopt`) |
| CMake submodule fix | `add_subdirectory(thirdparty/unitree_sdk2)` → `find_package(unitree_sdk2 REQUIRED)` |

**Build command (confirmed):**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_PREFIX_PATH="/usr/local" \
         -DCMAKE_CXX_COMPILER=g++-10 \
         -DCMAKE_C_COMPILER=gcc-10
make -j$(nproc)
```

**TRT engine sizes (Jetson):** encoder 47 MB · decoder 39 MB · planner 710 MB

---

## Deploy — Keyboard Sequence

```
ReleaseMode()       # via unitree_sdk2_python — robot must exit sport mode
                    # Robot LED: cyan = ready
./g1_deploy_onnx_ref eth0 policy/release/model_decoder.onnx reference/{SKILL}/ \
    --obs-config policy/release/observation_config.yaml \
    --encoder-file policy/release/model_encoder.onnx
                    # Wait for "Init Done" in log (~60-90s for planner TRT cache)
]                   # Start control system (silent if robot is green/damping — must be cyan)
T                   # Play reference motion to completion
O                   # Emergency stop
```

> ⚠️ `]` is **silently ignored** in green/damping state. Robot must be in cyan **before** launching the binary.

---

## Repository Structure

```
blea-udi/
├── robot_filtered_motionlib_clean/      # Saludo Militar: 8 motionlib PKLs (MJ order)
├── robot_filtered_motionlib_macarena/   # Macarena: 4 motionlib PKLs (MJ order)
├── reference/
│   ├── saludo_militar/saludo_militar/   # Deploy CSVs (from vis PKL, IL order)
│   └── macarena/macarena/              # Deploy CSVs (from vis PKL, IL order)
├── gear_sonic_deploy/
│   ├── target/release/g1_deploy_onnx_ref
│   ├── policy/
│   │   ├── saludo_militar/              # encoder + decoder ONNXs, step 001000
│   │   └── macarena2/                   # encoder + decoder ONNXs, step 001000
│   ├── planner/target_vel/V2/
│   │   └── planner_sonic_opset14_cf2.onnx
│   └── policy/release/observation_config.yaml
├── convert_motions.py                   # vis PKL → deploy CSVs
├── opset14_converter.py                 # ONNX opset 17 → 14 (LayerNorm decomposition)
├── sim2real_v2.ipynb                    # Active runbook (§1–§9)
└── scene_29dof_freebase.xml            # MuJoCo free-base validation scene
```

---

## 29-DOF Joint Order (MuJoCo / CSV `q_0`..`q_28`)

```
 0–5   Left leg:   L_hip_pitch, L_hip_roll, L_hip_yaw, L_knee, L_ankle_pitch, L_ankle_roll
 6–11  Right leg:  R_hip_pitch, R_hip_roll, R_hip_yaw, R_knee, R_ankle_pitch, R_ankle_roll
12–14  Waist:      waist_yaw, waist_roll, waist_pitch
15–21  Left arm:   L_shoulder_pitch, L_shoulder_roll, L_shoulder_yaw,
                   L_elbow, L_wrist_roll, L_wrist_pitch, L_wrist_yaw
22–28  Right arm:  R_shoulder_pitch, R_shoulder_roll, R_shoulder_yaw,
                   R_elbow, R_wrist_roll, R_wrist_pitch, R_wrist_yaw
```

---

## Citation

```bibtex
@article{barajas2026bml,
  title   = {Multi-Actor Sport and Cultural Motion Library for {GR00T}-Compatible
             Humanoid Imitation: Dataset, Pipeline, and Sim-to-Real Validation},
  author  = {Barajas Solano, Cris\'{o}stomo Alberto and {[TODO: Co-authors]}},
  journal = {arXiv preprint arXiv:{TODO:xxxx.xxxxx}},
  year    = {2026}
}
```

---

## Acknowledgements

Universidad de Investigación y Desarrollo (UDI) · GIDSAW Research Group · Semillero Master Digital · Actors: Cris, Edson · [TODO: funding]

---

*BLEA — Behavior Learning from Embodied Action. UDI Humanoid Robotics Lab, Aula Enrico Fermi 401, Bucaramanga, Colombia.*
