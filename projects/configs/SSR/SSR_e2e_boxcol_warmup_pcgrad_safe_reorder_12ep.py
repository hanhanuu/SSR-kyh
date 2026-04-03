_base_ = ['./SSR_e2e_boxcol_warmup_pcgrad_12ep.py']

# Experiment-2 / Group-2:
# Group-1 + PCGrad + inference-time safety reordering.
model = dict(
    infer_safe_reorder=True,
    infer_safe_reorder_use_box=True,
    infer_safe_reorder_risk_weight=1.0,
    infer_safe_reorder_cmd_penalty=0.08,
    infer_safe_reorder_dev_weight=0.05,
    infer_safe_reorder_box_kernel_m=(1.8, 4.2),
)
