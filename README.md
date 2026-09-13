<h2 align="center">DFDETR: PCB Defect Detection Based on Direction-Aware and Frequency-Enhanced Hybrid Encoding</h2>

<p align="center">
    <a href="#architecture">
        <img alt="method" src="https://img.shields.io/badge/backbone-ResNet18-blue">
    </a>
</p>

---

**DFDETR** is an end-to-end detection Transformer for **printed circuit board (PCB) defect detection**, built on top of [RT-DETR](https://arxiv.org/abs/2304.08069). It addresses three practical challenges of PCB imagery — strongly regular trace textures, orientation-sensitive defects, and insufficient use of frequency information — by cascading two lightweight, resolution-preserving modules **between multi-scale feature projection and FPN-PAN fusion**:

- **DADC — Direction-Aware Deformable Convolution.** The local dominant trace direction is predicted, and a structured 3×3 deformable sampling grid is reconstructed along the direction and its orthogonal axis. This aligns the receptive field with the elongated, locally oriented geometry of PCB traces.
- **SWFD — Stationary Wavelet-inspired Feature Decomposition.** A downsampling-free Haar filter bank decomposes the feature into four subbands (LL/LH/HL/HH); the HH detail subband guides a learnable enhancement attention while all subbands are retained, so low-frequency structure is never discarded.

Both modules are wrapped in **zero-initialized residual gates**, so each branch starts as an identity mapping and only begins to act when it benefits the detection objective. The detector interface, decoder, and prediction heads of RT-DETR remain unchanged.

---

## 🚀 Highlights

- Complementary **direction + frequency** cues to separate defects from the strongly regular trace background.
- Backward compatible with the standard RT-DETR head — no change to the decoder or prediction heads.

---

A **ResNet18** backbone extracts three feature levels `{S3, S4, S5}` at 1/8, 1/16, and 1/32 input resolution. A `1×1` projection maps every level to `C = 256` channels. Each projected feature then passes through **DADC → SWFD** without changing its `B×C×H×W` shape. The enhanced features enter the standard top-down FPN and bottom-up PAN fusion; the `S5` path additionally uses the RT-DETR Transformer encoder with 2D sin–cos positional encoding.

The insertion point is deliberate: applying the modules after projection gives all scales the same channel dimension (no per-stage reimplementation), and applying them before FPN-PAN lets the enhanced high-resolution detail and low-resolution semantics interact through both the top-down and bottom-up paths.

### DADC (Direction-Aware Deformable Convolution)

For an input `x ∈ R^{B×C×H×W}`, a direction predictor compresses channels with `1×1`, aggregates local context with a `3×3` convolutional + BatchNorm + GELU stack, and a final `1×1` layer emits an angle logit `g_θ(x)`. The bounded angle and its orthogonal unit vectors are

```
θ(u,v) = (π/2) · tanh(g_θ(x))
d      = (cos θ, sin θ)
d⊥     = (−sin θ, cos θ)
```

Rather than predicting 18 independent offsets, DADC reconstructs the 3×3 base grid in the local coordinate system `(d, d⊥)`. Globally learned parameters `α_n` and `β_n` control the sampling scale parallel and perpendicular to the predicted direction:

```
s_∥,n = 1 + tanh(α_n)      s_⊥,n = 1 + tanh(β_n)
q_n   = s_∥,n·r^x_n·d + s_⊥,n·r^y_n·d⊥          Δp_n = q_n − r_n
```

where `r_n ∈ {−1,0,1}²` is the n-th position of the base grid. The nine 2-D offsets are concatenated to form an 18-channel field. When `θ = 0` and `α_n = β_n = 0`, the offsets vanish and the grid reduces to standard convolution. With learnable weights `W`, deformable convolution computes

```
z(p0) = Σ_n W(p_n)·x(p0 + p_n + Δp_n)        y = x + γ·z
```

The residual gate `γ` is initialized to zero, so the initial mapping is exactly `y = x`. Training then introduces direction-aligned sampling only when it benefits detection. This structured parameterization is more constrained than conventional deformable convolution, but it directly represents the elongated, locally oriented geometry of PCB traces.

### SWFD (Stationary Wavelet-inspired Feature Decomposition)

SWFD follows the stationary-wavelet principle to preserve spatial resolution, applying four fixed Haar filters channel-wise with **dilation 2, stride 1, padding 1**:

```
h_LL = [[.5, .5], [.5, .5]]      h_LH = [[ .5,  .5], [−.5, −.5]]
h_HL = [[.5, −.5], [.5, −.5]]    h_HH = [[ .5, −.5], [−.5,  .5]]
```

The LL response captures smooth structure, LH/HL emphasize horizontal/vertical boundaries, and HH captures joint high-frequency details. The **HH subband is treated as a detail cue** — it may respond to defects, normal corners, or noise, so it is not amplified unconditionally. All four subbands are concatenated into a `4C`-channel tensor, fused back to `C` channels by a `1×1` Conv + BatchNorm + GELU, and a lightweight two-layer branch maps HH to `A_HH ∈ [0,1]^{B×C×H×W}`:

```
F_enh = F_fused ⊙ (1 + δ·A_HH)
F_out = F_in + γ·F_enh
```

Both `δ` and the residual gate `γ` are initialized to zero. The learnable sign of `δ` permits either enhancement or suppression of HH responses, while the residual path initially preserves the pretrained feature. The module therefore begins as an identity mapping and progressively learns a task-dependent balance among smooth structure, directional boundaries, and localized details.

---

## ⚙️ Setup

```bash
# 1. Install dependencies
cd rtdetr_pytorch
pip install -r requirements.txt

# 2. Train (select dataset via DATASET in tools/train.py: 'pcb' | 'deeppcb' | ...)
python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml

# multi-gpu
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml

# 3. Evaluate
python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -r path/to/checkpoint --test-only

# 4. Inference on PCB-DATASET / DeepPCB
python tools/infer-pcbdataset.py
python tools/inferdeeppcb.py
```

> Datasets are configured through the `DATASET` variable in [`tools/train.py`](rtdetr_pytorch/tools/train.py). For training details, ablation switches (`use_amsf` / `use_edge_enhancer` / `use_texture_aware`), and custom datasets, see [`rtdetr_pytorch/`](rtdetr_pytorch/) and [`rtdetr_pytorch/tools/README.md`](rtdetr_pytorch/tools/README.md).
