# MVP-1 Protocol (Hard Synthetic)

Use this protocol to run a matched drift vs no-drift comparison on the harder synthetic generator.

## Configs
- Drift: `config_mvp1_hard.yaml`
- No-drift control: `config_mvp1_hard_nodrift.yaml`

## Single-Seed Smoke
```bash
python train.py --config config_mvp1_hard.yaml
python eval.py --config config_mvp1_hard.yaml --ckpt outputs/mvp1_hard_drift/model_best.pt --split val

python train.py --config config_mvp1_hard_nodrift.yaml
python eval.py --config config_mvp1_hard_nodrift.yaml --ckpt outputs/mvp1_hard_nodrift/model_best.pt --split val
```

## Multi-Seed Matched Sweep (Colab Cell)
```python
import os, json, copy, yaml, subprocess, sys, statistics

DRIFT_CFG = "config_mvp1_hard.yaml"
CTRL_CFG = "config_mvp1_hard_nodrift.yaml"
SEEDS = [42, 43, 44]

def run_cfg(base_cfg_path, tag):
    with open(base_cfg_path, "r") as f:
        base_cfg = yaml.safe_load(f)
    rows = []
    for seed in SEEDS:
        cfg = copy.deepcopy(base_cfg)
        cfg["seed"] = seed
        cfg["output_dir"] = f"outputs/mvp1_{tag}/seed{seed}"
        os.makedirs(cfg["output_dir"], exist_ok=True)
        tmp_cfg = f"outputs/mvp1_{tag}/tmp_seed{seed}.yaml"
        with open(tmp_cfg, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        subprocess.run([sys.executable, "train.py", "--config", tmp_cfg], check=True)
        subprocess.run([sys.executable, "eval.py", "--config", tmp_cfg, "--ckpt", f"{cfg['output_dir']}/model_best.pt", "--split", "val"], check=True)

        with open(f"{cfg['output_dir']}/eval_val.json", "r") as f:
            m = json.load(f)
        row = {"seed": seed, "dice": m["dice"], "bce_loss": m["bce_loss"], "sec_per_sample": m["sec_per_sample"]}
        rows.append(row)
        print(tag, row)

    mean_dice = statistics.mean(r["dice"] for r in rows)
    std_dice = statistics.pstdev(r["dice"] for r in rows)
    return rows, {"mean_dice": mean_dice, "std_dice": std_dice, "n": len(rows)}

drift_rows, drift_agg = run_cfg(DRIFT_CFG, "hard_drift")
ctrl_rows, ctrl_agg = run_cfg(CTRL_CFG, "hard_nodrift")
print("drift:", drift_agg)
print("control:", ctrl_agg)
print("delta_drift_minus_control:", drift_agg["mean_dice"] - ctrl_agg["mean_dice"])
```
