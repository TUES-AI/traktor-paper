# Offline DINO clustering

The runtime project is visionless: no camera, DINO, MobileNet, VMM, or PCVM input
is allowed into the GRU/SAC policy.

The one retained vision utility is `tools/replay_dinov3_onnx_clusters.py`. It is
an offline paper-analysis tool for saved images. It can show that frozen DINO
features cluster rooms/views well, and it may support a future reward-only
ablation where camera feedback affects the scalar reward after an action. Even in
that ablation, visual embeddings must not be policy observations.

Example:

```bash
/Volumes/SSD/v/py/bin/python tools/replay_dinov3_onnx_clusters.py \
  --frame-dir /path/to/saved_frames \
  --out-dir results/offline_dino_clusters \
  --size 336 \
  --known-dist 0.11
```
