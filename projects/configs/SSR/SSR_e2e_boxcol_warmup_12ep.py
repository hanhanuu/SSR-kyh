_base_ = ['./SSR_e2e.py']

# Experiment-2 / Group-1:
# Baseline + differentiable box-collision loss (future occupancy proxy) + warmup.
model = dict(
    pts_bbox_head=dict(
        use_plan_box_col_loss=True,
        plan_box_col_loss_weight=0.6,
        plan_box_col_warmup_epochs=2.0,
        plan_box_col_warmup_start=0.1,
        plan_box_col_use_future_occ=True,
        # Approximate ego footprint for soft box collision in meters (w, l).
        metric_box_kernel_m=(1.8, 4.2),
    ),
)
