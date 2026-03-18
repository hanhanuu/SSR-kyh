_base_ = ['./SSR_e2e_risk_fuse_mlp_aspp_futureocc_12ep.py']

# Micro-tune: remove dilation on future occupancy target to reduce over-conservatism.
model = dict(
    pts_bbox_head=dict(
        risk_target_dilate=0,
    )
)

