# Running the FABRIC experiments on molab (marimo, GPU)

Check molab's current GPU options, session time limits and storage rules in its UI before starting.
Assume the session disk can be wiped: download or upload every result folder as soon as a run finishes.

## 0. Before you open molab
- Your repo (main branch) with the T1 split fix, `explain_ig.py` and `scripts/run_seed.py` must be pushed
.
- If the repo is private, create a GitHub fine-grained token with read access to that repo only.
- Optional: a Hugging Face write token if you want to push checkpoints (use private repos; never link them in the paper).

## 1. Notebook cells (create a new marimo notebook, one block = one cell)

Cell 1: GPU check
```python
import subprocess, torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
```

Cell 2: get the code
```python
import subprocess, os
REPO = "https://<TOKEN>@github.com/<your-username>/<your-repo>.git"   # your own private repo
os.chdir("/tmp") if not os.path.exists("/tmp/dermai-explainability") else None
subprocess.run(["git", "clone", REPO, "/tmp/dermai-explainability"], check=False)
os.chdir("/tmp/dermai-explainability")
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
```

Cell 3: dependencies (keep the preinstalled torch if it already sees the GPU)
```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=5.12", "scikit-learn", "pandas", "pyyaml", "matplotlib", "captum"], check=True)
```

Cell 4: data (about 2.8 GB from Harvard Dataverse, DOI 10.7910/DVN/DBW86T)
```python
import subprocess, sys
subprocess.run([sys.executable, "scripts/download_data.py"], check=True)
```

Cell 5: split sanity check (must print True before any training)
```python
import subprocess, sys
r = subprocess.run([sys.executable, "scripts/run_seed.py", "--check-split-only"], capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:])
```

Cell 6: one run (edit MODEL and SEED, rerun the cell for each of the 6 runs)
```python
import subprocess, sys
MODEL, SEED = "efficientnet", 42          # efficientnet | vit ; seeds 42, 43, 44
r = subprocess.run([sys.executable, "scripts/run_seed.py", "--model", MODEL, "--seed", str(SEED)],
                   capture_output=True, text=True)
print(r.stdout[-4000:]); print(r.stderr[-4000:])
```
Order: efficientnet 42 first (fast, confirms the pipeline reproduces the paper's seed-42 numbers within
noise), then vit 42, then seeds 43 and 44. Time the first epoch and extrapolate before launching the rest.

Cell 7: package results for download
```python
import shutil
run = f"results/final/{MODEL}_s{SEED}"
shutil.make_archive(f"/tmp/{MODEL}_s{SEED}", "zip", run)
print("download:", f"/tmp/{MODEL}_s{SEED}.zip")
```
Download each ZIP (or push to a private HF dataset repo) and upload it to the cloud Claude session, which
unzips it into `results/final/` and runs `scripts/aggregate.py`.

## 2. What each run must produce (`results/final/<model>_s<seed>/`)
- `config_used.yaml`, `checkpoint_sha256.txt`, `test_image_ids.txt`
- `classification.json` (val selection metric, test macro-F1, balanced accuracy, per-class F1)
- `heatmaps_native/` (Grad-CAM or rollout .npy), `heatmaps_ig/` (.npy)
- `faithfulness_{native,ig,random,center}.csv` and curve `.npz`
- `localization_{native,ig,center}_tol{0,15}.csv`
- `summary.json`
Heatmap folders can be large; if needed keep only the CSV/JSON plus 20 qualitative examples.

## 3. If something fails
- CUDA OOM on ViT: lower batch size for IG only (IG steps are batched internally); do not change training config.
- Seed-42 numbers far from the paper (test macro-F1 0.655 EN / 0.748 ViT): stop and report; checkpoint or
  config differs from what produced the paper (trap T2/T3).
- Session timeout mid-run: rerun that single MODEL/SEED; runs are independent.
