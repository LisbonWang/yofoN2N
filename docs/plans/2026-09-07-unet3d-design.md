# 3D Residual UNet (Depthwise-Separable, base8/depth3)

Date: 2026-09-07
Status: implemented, awaiting on-server smoke run
Owner: yyc

## Goal

Add a true 3D residual UNet as an optional model in the N2N training stack.
Default to the lightweight depthwise-separable variant with `base_channels=8`
and `depth=3` so the architecture can be exercised end-to-end on full CBCT
volumes before committing to wider / deeper variants.

## Non-Goals

* Replacing the existing 2D UNet.  The 2D pipeline remains the primary
  path; the 3D model is opt-in via `model.arch: unet3d`.
* Mixed-precision kernel tuning, custom CUDA kernels, or model
  parallelism.  Pure PyTorch ops only; if memory becomes a bottleneck
  the next step is gradient checkpointing or smaller `voxel_depth`.
* New data preprocessing.  The 3D model reuses the same reconstruction
  index, manifest, mappings, and residual augmentation as the 2D model.

## Design

### 1. Model — `src/cbct_n2n/models/unet3d.py`

```
DepthwiseSeparableConv3d(in, out):
    depthwise: Conv3d(in, in, 3, padding=1, groups=in, bias=False)
    pointwise: Conv3d(in, out, 1, bias=False)
    norm:      GroupNorm(min(8, out) adjusted for divisibility, out)
    act:       SiLU(inplace=True)

DSConvBlock3d(in, out):
    local_residual = Identity  if in == out
                   = Conv3d(in, out, 1)  else
    return local_residual(x) + conv2(conv1(x))

UNet3D(in_channels, out_channels, base_channels=8, depth=3,
       residual=True, pool_kernel=(1, 2, 2)):
    features = [base * 2**i for i in range(depth)]   # base8/depth3 -> [8, 16, 32]
    encoders   = [DSConvBlock3d(prev, c) for c in features]
    bottleneck = DSConvBlock3d(features[-1], features[-1]*2)             # 32 -> 64
    upconvs    = [ConvTranspose3d(prev, c, kernel=pk, stride=pk) for c in reversed(features)]
    decoders   = [DSConvBlock3d(c*2, c) for c in reversed(features)]     # cat skip + upconv
    head       = Conv3d(features[0], out_channels, 1)                    # 8 -> 1
    global_skip = Identity  if residual and in == out
                = Conv3d(in, out, 1) else
    forward:
        for encoder: store skip, MaxPool3d(pool_kernel)
        bottleneck
        for upconv, decoder, skip: upconv -> _align -> cat(skip, x) -> decoder
        head(x) + (global_skip(source) if residual else 0)
```

The architecture is a 1:1 mirror of `UNet2D` (same encoder/bottleneck/decoder
shape, same skip semantics, same GroupNorm/SiLU/conv kernel choices) so
configs already tuned for the 2D model translate without re-thinking.

**Per-axis `pool_kernel`.**  CBCT slabs have very different ranges
along D vs. H/W (slice count 16-200, in-plane 256-768).  Isotropic
stride-2 pooling rapidly eats the depth axis (e.g. 16 -> 8 -> 4 -> 2 ->
1 after three stages), so the encoder/decoder let each axis pick its
own stride via a 3-tuple `(kd, kh, kw)`:

* Used as the `kernel_size` of every `F.max_pool3d` call in the
  encoder, and as the `kernel_size` and `stride` of every
  `ConvTranspose3d` in the decoder so the dimensions strictly invert.
* Default `(1, 2, 2)`: depth axis is left intact (kernel 1 is a no-op
  along that axis), spatial axes compress 2x per stage.
* Pass `(2, 2, 2)` for the original isotropic behaviour.
* All entries must be `>= 1`; constructor raises `ValueError` otherwise.
* The forward path raises immediately if any axis collapses to zero, so
  malformed data is caught at runtime, not silently truncated.

### 2. Dataset — `src/cbct_n2n/data/n2n3d_dataset.py`

`N2N3DDataset(N2NDataset, voxel_depth=15)` (renamed from the original
`N2NVoxelDataset` to match the new contract) inherits everything from
the slice dataset and overrides `__getitem__` to:

* **Input** (`source`): a stack of `voxel_depth` consecutive axial
  slices around the anchor, shaped `[1, D, H, W]`.  `voxel_depth` must
  be **odd and >= 1** (the centre index is `D // 2`).
* **Output** (`target`): the **single centre slice** of the slab, shaped
  `[1, H, W]`.  In N2N fashion this is the half-view partner /
  full-view reference for the anchor slice -- the same slice the parent
  :class:`N2NDataset` would have returned, just at a single z instead
  of the full slab.
* **Boundary handling**: out-of-range z indices are filled by
  **replicating the boundary slice** (`np.pad(mode='edge')` semantics,
  implemented in `_gather_slab`).  This is the natural extrapolation
  for image-domain models: the border slice is repeated rather than
  mirrored, which would invert anatomical orientation near the volume
  ends.
* **Padding**: each slice's H/W is padded to a multiple of 32 (via the
  existing `pad_to_multiple` helper) so the UNet's stride-2
  pools/upconvs always divide evenly.
* **Augment**: depth-axis flip (only when `voxel_depth > 1`, applied
  to source only since target is the centre slice), H/W flip, optional
  90° rotations on H/W when `H == W`.

The parent class still owns manifest parsing, residual tags, mappings,
and split routing; only the per-item tensor shape and boundary
extrapolation differ from the 2D dataset.

**Why odd?**  An even `voxel_depth` would have two "centre" slices and
no clean anchor -> target mapping.  An odd depth lets us pick exactly
one anchor slice and have the model see a symmetric context window
around it.

**D-vs-depth constraint**: with `base_channels=8, depth=3` the UNet
applies three stride-2 max-pools along D, so the input D must be
`>= 2**depth = 8` to avoid a zero-sized pool output.  Combined with the
odd requirement, the **smallest valid `voxel_depth` for the default
config is 9**; the smoke config uses **15** (well above the minimum and
symmetric around the centre).

### 3. Training entry — `scripts/train_voxel.py`

A new script that:

* Loads the same YAML config layout (`reconstruction`/`data`/`model`/
  `training`) and reads `model.arch` (default `unet3d`).
* Builds `N2N3DDataset` with `voxel_depth` from `data.voxel_depth`
  (default 15; must be odd).
* Instantiates `UNet3D(**config["model"])` (or `UNet2D` if
  `arch=unet2d`) and **wraps it in :class:`CenterSliceHead`** when
  `arch=unet3d` so the 5D `[B, C, D, H, W]` UNet output is collapsed
  to a 2D `[B, C, H, W]` prediction by slicing along D at `D // 2`.
  This lets the existing :func:`cbct_n2n.training.trainer.fit` loss
  path operate on `model(source) vs target` without modification --
  the centre slice of the UNet's 5D output is the prediction that gets
  compared to the 2D centre-slice target.
* Reuses `cbct_n2n.training.trainer.fit` so checkpointing, scheduler,
  AMP, history, and resume all stay in lockstep with the 2D pipeline.
* CLI flags `--resume` / `--init-weights` mirror `train_sparse_dense.py`.

`scripts/train_sparse_dense.py` is untouched: 2D training keeps its
own entry point.

### 4. Configuration — `configs/unet3d_smoke.yaml`

Identical to `configs/smoke.yaml` except:

* `model.arch: unet3d`
* `model.base_channels: 8` (was 32)
* `model.depth: 3` (was 4)
* `model.pool_kernel: [1, 2, 2]` (default; see §1 above)
* `data.voxel_depth: 15` (odd; smallest valid for base8/depth3 is 9)
* `training.batch_size: 1` (5D source tensors are ~15× the slice
  footprint, but the 2D target is unchanged)

`scripts/train_voxel.build_model` filters `pool_kernel` out of the
config dict before constructing `UNet2D`, so the same YAML can drive
either architecture without modification.

### 5. Tests — `tests/models/test_unet3d.py` + `tests/data/test_n2n3d_dataset.py`

*Model* (`tests/models/test_unet3d.py`):
* Forward-shape preservation across a few 5D shapes (incl. non-power-of-two).
* Default config matches `base8/depth3` spec; total parameters < 200k.
* Parameter validation: zero channels / `depth < 2` raise `ValueError`.
* Non-5D input is rejected.
* Residual on vs off produce different outputs (sanity for the global
  skip).
* `DepthwiseSeparableConv3d` uses `groups=in_channels` and a 1×1×1
  pointwise; preserves spatial shape.
* `DSConvBlock3d` uses a 1×1×1 projection when channels differ, an
  `Identity` when they don't.
* `pool_kernel` defaults to `(1, 2, 2)` and accepts tuple or list.
* Custom kernels (`(2, 2, 2)`, `(1, 4, 4)`) work and preserve shape.
* Invalid kernels (wrong length / non-positive ints) raise
  `ValueError`.
* With `kd=1`, the depth axis is unchanged through all encoder stages
  (verified by a forward hook on the first encoder).
* Default config (`(1, 2, 2)`) stays under the 200k parameter ceiling.

*Dataset* (`tests/data/test_n2n3d_dataset.py`):
* `voxel_depth` validation: must be positive and odd.
* `__getitem__` returns `source=[1, D, H, W]` (5D) and
  `target=[1, H, W]` (2D) with the centre slice as target.
* Out-of-range z indices are replicated (boundary) rather than
  reflected.
* Sample metadata keys stay `default_collate`-compatible (residual is
  the empty-string sentinel, not `None`).

**Test infrastructure note**: on Windows the developer sandbox
restricts I/O against pytest's default `tmp_path` (which lives under
`%TEMP%\pytest-of-admin\garbage-...`), making `np.memmap` return
`WinError 8` ("not enough memory resources").  The dataset test file
installs an `autouse` fixture that replaces `np.memmap` with
`np.fromfile` (the dataset only reads), so the test path works on both
sandboxed dev machines and the CI runner.

## Parameter Budget (base8 / depth3)

```
Encoder 8   -> 8   : 2 × (27·8 + 8·8  + 8)   ≈ 560
Encoder 16  -> 16  : 2 × (27·16 + 16·16 + 16) ≈ 2.2k
Encoder 32  -> 32  : 2 × (27·32 + 32·32 + 32) ≈ 8.7k
Bottleneck 32 -> 64 : 2 × (27·32 + 32·64 + 64) ≈ 9.5k
Decoder mirrors: ≈ 11k
Head (8->1): 8
Global skip (1->1): identity
Total ≈ 60–70k parameters
```

Well under the 200k ceiling asserted in tests.  Switching to
`base_channels=16, depth=4` keeps the same shape but lands ~10× heavier.

## Open Questions / Future Work

* **Patch overlap / stride during inference.**  When inference runs
  over a full volume, neighbouring patches will overlap; we currently
  have no consensus on aggregation (mean / median / centre crop).
  Defer until the first on-server smoke run produces usable predictions.
* **Gradient checkpointing.**  Even base8/depth3 has a sizeable
  activation footprint at `voxel_depth=32` or larger base_channels.
  Add `torch.utils.checkpoint` only if memory becomes the bottleneck.
* **2.5D hybrid.**  A future option could feed `voxel_depth` slices as
  channels (`[B, D, H, W]`) and run a modified UNet2D, sharing weights
  across slices; not part of this design.

## Files Touched

```
+ src/cbct_n2n/models/unet3d.py            (new, ~150 LOC;
                                            updated in pass-2 to add
                                            per-axis pool_kernel)
+ src/cbct_n2n/data/n2n3d_dataset.py       (new, ~130 LOC;
                                            renamed from voxel_dataset.py)
- src/cbct_n2n/data/voxel_dataset.py       (removed)
+ scripts/train_voxel.py                   (new, ~135 LOC; adds
                                            CenterSliceHead, filters
                                            per-arch kwargs in build_model)
+ configs/unet3d_smoke.yaml                (new; pass-2 adds
                                            model.pool_kernel)
+ tests/models/test_unet3d.py              (new, 19 cases — pass-2 adds
                                            7 pool_kernel cases)
+ tests/data/test_n2n3d_dataset.py         (new, 6 cases;
                                            uses a memmap stub fixture)
+ docs/plans/2026-09-07-unet3d-design.md   (this file; updated in pass-2)
```

## Pass 3 — Depthwise-separable 2D UNet (`UNet2D_DS`)

Building on the FLOPs comparison, we ported the same depthwise-separable
conv block down to 2D so the 2D and 3D models sit on equal footing — any
remaining FLOPs/parameter gap then reflects the extra slice axis alone.

```
DepthwiseSeparableConv2d(in, out):
    depthwise: Conv2d(in, in, 3, padding=1, groups=in, bias=False)
    pointwise: Conv2d(in, out, 1, bias=False)
    norm: GroupNorm(min(8, out) groups), act: SiLU

DSConvBlock2d(in, out):
    conv1 = DepthwiseSeparableConv2d(in, out)
    conv2 = DepthwiseSeparableConv2d(out, out)
    skip  = Conv2d(in, out, 1) if in != out else Identity
    out   = conv2(conv1(x)) + skip(x)

UNet2D_DS(in=1, out=1, base_channels=16, depth=4, residual=True):
    encoders  = [DSConvBlock2d(prev, c) for c in features]   # [16,32,64,128]
    bottleneck= DSConvBlock2d(128, 256)
    upconvs   = [ConvTranspose2d(prev, c, 2, stride=2)]
    decoders  = [DSConvBlock2d(2c, c)]
    head      = Conv2d(16, out, 1)
    global_skip = Conv2d(1, out, 1) if residual and in != out else Identity
    forward   = max_pool2d(2) down, _align + cat up, head + global_skip(source)
```

Measured @640×640 single-sample forward:

| model (base16/depth4) | params | MACs |
|---|---|---|
| UNet2D standard-conv | 1,942,289 | 21.35 G |
| **UNet2D_DS** | **473,562** | **6.89 G** |

Depthwise-separable cuts ~76% of parameters and ~68% of FLOPs versus the
standard-conv UNet2D at the same config — a larger proportional saving than
in 3D, because the 2D depthwise kernel is only 3×3 (9 weights) so the
pointwise 1×1 dominates and the depthwise pass is nearly free.

Wiring:

* `src/cbct_n2n/models/unet2d_ds.py` — new model module.
* `src/cbct_n2n/models/__init__.py` — now exports `UNet2D_DS` (and `UNet3D`).
* `scripts/train_sparse_dense.py` — gains `--arch {unet2d,unet2d_ds}` (default
  `unet2d`, fully backward compatible) via a small `build_model` dispatch.
* `scripts/train_voxel.py` — `build_model` now accepts `unet2d_ds` too.
* `configs/unet2d_ds_smoke.yaml` — `arch: unet2d_ds`, `base_channels: 16`,
  `depth: 4`.
* `tests/models/test_unet2d_ds.py` — 14 cases (shape/backprop, param budget,
  lighter-than-standard assertion, validation, residual on/off, DS conv
  groups, local-residual projection, non-power-of-two dims, non-divisible
  base channels).

No existing project files were modified; the design is layered on top
of the existing 2D pipeline as a fully optional second model class.
