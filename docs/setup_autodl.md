## AutoDL 5090 Setup

This project is designed so Mac-side work can focus on coding and tests, while GPU
training runs on AutoDL.

```bash
cd /root/alibaba-ai
pip install -r requirements.txt
pip install -e .
python scripts/setup/check_gpu.py
```

Expected training command for `3.1.1`:

```bash
python scripts/train/train_instance_segmentation.py \
  --model-config configs/model/instance_segmentation.yaml \
  --paths-config configs/paths.autodl.yaml
```

AutoDL dataset and checkpoint paths are configured in `configs/paths.autodl.yaml`.
