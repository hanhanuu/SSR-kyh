_base_ = ['./SSR_e2e_boxcol_warmup_12ep.py']

# Experiment-2 / Group-2:
# Group-1 + gradient conflict handling (PCGrad-style) between L2 and BoxCol losses.
model = dict(
    pts_bbox_head=dict(
        use_plan_pcgrad=True,
        pcgrad_eps=1e-8,
        pcgrad_min_scale=0.25,
        pcgrad_max_scale=2.0,
    ),
)
