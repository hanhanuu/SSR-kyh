
## Corruption Robustness Eval (3D_Corruptions_AD / nuScenes)

目标：**不重新训练**，直接用 clean nuScenes 上训练好的 `checkpoint`，在 corruption 数据上 **测试/评测**，看指标下降幅度（drop）来证明鲁棒性。

本仓库的 `vad_nuscenes_infos_temporal_*.pkl` 里保存了 `lidar_path` / `cams[*].data_path` 等**具体文件路径**，所以需要为每个 `corruption × severity` 准备一份“路径已改写”的 info PKL。

### 0) 从 0 生成（如果你还没有 corruption 数据目录）

如果你当前机器上还没有 `.../<corruption>/<severity>/samples/...` 这种目录，可以先用脚本在仓库内直接生成 **val 集合的相机 corruption 图像**：

脚本：`tools/generate_nuscenes_weather_corruptions.py`

示例（fog severity=3，生成到 `data/nuscenes_corruptions/fog/3/samples/...`）：
```
python tools/generate_nuscenes_weather_corruptions.py \
  --info data/nuscenes/vad_nuscenes_infos_temporal_val.pkl \
  --out-root data/nuscenes_corruptions/fog/3 \
  --corruption fog \
  --severity 3 \
  --workers 16
```

生成完成后，继续执行下面的 “生成改写路径的 info PKL” + “dist_test” 即可。

### 1) 放置/链接 corruption 数据

保证 corruption 数据的目录里仍然包含 nuScenes 的相对结构（至少有 `samples/`，如有多帧点云则还有 `sweeps/`），且文件名与原 nuScenes 对齐。

推荐在仓库内做软链接，保持路径可复现，例如：
```
mkdir -p data/nuscenes_corruptions/fog
ln -s /ABS/PATH/TO/3D_Corruptions_AD/nuscenes/fog/3 data/nuscenes_corruptions/fog/3
```

### 2) 生成“改写路径”的 info PKL（推荐 anchor 模式）

脚本：`tools/create_nuscenes_corruption_infos.py`

仅改写相机图片路径（如果 corruption 只改了图像）：
```
python tools/create_nuscenes_corruption_infos.py \
  --src data/nuscenes/vad_nuscenes_infos_temporal_val.pkl \
  --dst data/corruptions/vad_nuscenes_infos_temporal_val_fog3_cam.pkl \
  --mode anchor \
  --new-root ./data/nuscenes_corruptions/fog/3 \
  --rewrite camera \
  --check 50
```

同时改写相机 + LiDAR（如果 corruption 同时提供了图像/点云）：
```
python tools/create_nuscenes_corruption_infos.py \
  --src data/nuscenes/vad_nuscenes_infos_temporal_val.pkl \
  --dst data/corruptions/vad_nuscenes_infos_temporal_val_fog3_cam_lidar.pkl \
  --mode anchor \
  --new-root ./data/nuscenes_corruptions/fog/3 \
  --rewrite camera lidar \
  --check 50
```

说明：
- `--check N` 会抽样检查改写后的文件是否存在；如果缺文件会直接报错（可用 `--allow-missing` 跳过失败）。
- 如果 token/标注不变（常见情况），**GT 与 split 仍然沿用 clean nuScenes**，无需重新生成标注。

### 3) 测试（dist_test）

用 wrapper config：`projects/configs/SSR/SSR_e2e_corruptions_eval.py`

通过环境变量指定：
- `SSR_BASE_CFG`：你要评测的模型配置（Baseline/Ours 均可）
- `NUSC_CORRUPT_INFO`：上一步生成的 corruption info PKL

示例：
```
export SSR_BASE_CFG=projects/configs/SSR/SSR_e2e_risk_fuse_mlp_12ep_g123.py
export NUSC_CORRUPT_INFO=data/corruptions/vad_nuscenes_infos_temporal_val_fog3_cam.pkl

PYTHONPATH=$(pwd):$(pwd)/mmdetection3d:$PYTHONPATH \
./tools/dist_test.sh projects/configs/SSR/SSR_e2e_corruptions_eval.py <ckpt>.pth 4 \
  --out work_dirs/<exp>/results_fog3.pkl
```

### 4) 规划指标计算（compute_plan_metrics.py）

如果 corruption 数据保持 token 不变（最常见），`--info` 可以继续用 clean 的：
```
PYTHONPATH=$(pwd):$(pwd)/mmdetection3d:$PYTHONPATH \
python tools/compute_plan_metrics.py \
  --results work_dirs/<exp>/results_fog3.pkl \
  --info data/nuscenes/vad_nuscenes_infos_temporal_val.pkl \
  | tee work_dirs/<exp>/metrics_fog3.txt
```

如果 token/split 被你改动过，则把 `--info` 换成对应 corruption 的 info PKL。
