# DeepRodent — Architecture Notes

**Author:** Teerapong Panboonyuen

This note summarizes how the paper's equations map onto the code, for
reviewers and contributors who want to trace a symbol back to its
implementation.

## 1. Overall prediction function (Eq. 1)

```
F_theta(I_t) = { B_t, M_t, O_t, E_t }
```

Implemented by `DeepRodent.forward()` in `deeprodent/models/deeprodent.py`,
returning a dict with keys `det` (B_t), `seg_proto` (M_t), `obb` (O_t), and
`embed` (E_t).

## 2. Backbone (Eq. 2-3)

```
F_l = phi_l(F_{l-1})
F   = F^cls ⊕ F^box ⊕ F^seg ⊕ F^obb
```

`deeprodent/models/backbone.py::MultiScaleBackbone` implements the
layer-wise stack (`phi_l` = `CSPBlock` inside `DownStage`), producing a
[P3, P4, P5] pyramid. Task-specific heads in `heads.py` consume the
scale-aggregated feature to realize the F^cls / F^box / F^seg / F^obb
decomposition.

## 3. Oriented bounding boxes (Eq. 4-5)

```
o_i = (x_i, y_i, w_i, h_i, theta_i)
R(theta) = [[cos, -sin], [sin, cos]]
```

`deeprodent/models/heads.py::OBBHead` regresses `theta` through a
tanh-scaled activation bounded to `[-pi/2, pi/2)`.
`deeprodent/utils/obb_utils.py::rotation_matrix` implements `R(theta)`
directly, used by `obb_to_polygon` for corner-point conversion.

## 4. Segmentation head (Eq. 6-7)

```
M_i = sigmoid(W_m * F_seg(i))
p_ij = softmax(f_i^T f_j)
```

`deeprodent/models/heads.py::SegmentationHead` predicts prototype masks;
`assemble_instance_masks` implements the linear-combination + sigmoid
step. The pairwise affinity `p_ij` is naturally realized by any
attention-style module operating on `feat_agg`; see `MotionAwareUpdate`
for an analogous gated combination used in the temporal branch.

## 5. Multi-scale aggregation (Eq. 8)

```
F_agg = sum_s alpha_s * F_s,  alpha_s = softmax(gamma_s)
```

`deeprodent/models/backbone.py::ScaleAwareFusion`.

## 6. Temporal consistency (Eq. 9-10)

```
L_temp = sum_t || E_t - E_{t+1} ||_2^2
E_t = psi(E_{t-1}, F_t, delta_x_t)
```

`deeprodent/losses/losses.py::TemporalConsistencyLoss` and
`deeprodent/models/heads.py::MotionAwareUpdate`.

## 7. Multi-task loss (Eq. 11-22)

All terms (`L_cls`, `L_seg`, `L_box`, `L_obb`, `L_KL`, uncertainty
reweighting, `L_domain`) are implemented in
`deeprodent/losses/losses.py` and combined by `DeepRodentLoss`, matching
Eq. (22):

```
L_DeepRodent = L_final + lambda_6 * L_domain + lambda_7 * L_temp
```

## 8. Training protocol (Eq. 23, 27-30)

`deeprodent/engine/train.py::Trainer` configures `torch.optim.SGD` with
`(eta_0=1e-3, mu=0.937, weight_decay=5e-4)` and a `CosineLRWithWarmup`
scheduler implementing Eq. (30) exactly.

## 9. Dataset format (Eq. 24-25)

`deeprodent/data/dataset.py::RodentSegDataset` parses the
`c x1 y1 ... xn yn` polygon format described in Section 3, rasterizing
masks and deriving axis-aligned boxes / OBBs on the fly.

## 10. Evaluation (Eq. 26)

`deeprodent/engine/evaluate.py::Evaluator` computes Precision / Recall /
mAP50 / mAP50-95 / FPS; `Evaluator.multi_seed_summary` implements the
3-seed averaging protocol of Eq. (26).
