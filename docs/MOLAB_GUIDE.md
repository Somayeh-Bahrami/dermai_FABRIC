# Running the FABRIC experiments on molab (marimo, GPU)

Check molab's current GPU options, session time limits and storage rules in its UI before starting.
Assume the session disk can be wiped: download or upload every result folder as soon as a run finishes.

## 0. Before you open molab
- Your repo with the T1 split fix, `explain_ig.py` and `scripts/run_seed.py` pushed (branch
  `claude/tender-cannon-rxc8ic` until it is merged to main; check it out in Cell 2).
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
subprocess.run(["git", "fetch", "origin", "claude/tender-cannon-rxc8ic"], check=False)
subprocess.run(["git", "checkout", "claude/tender-cannon-rxc8ic"], check=False)
subprocess.run(["git", "pull", "--ff-only"], check=False)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
```

Cell 3: dependencies (keep the preinstalled torch if it already sees the GPU)
```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=5.12", "scikit-learn", "pandas", "pyyaml", "matplotlib", "captum", "pytest"], check=True)
```

Cell 4: data (about 2.8 GB from Harvard Dataverse, DOI 10.7910/DVN/DBW86T)
```python
import subprocess, sys
subprocess.run([sys.executable, "scripts/download_data.py"], check=True)
```

Cell 5: split sanity check (must print `identical: True` and `passed` before any training)
```python
import subprocess, sys
r = subprocess.run([sys.executable, "scripts/run_seed.py", "--check-split-only"], capture_output=True, text=True)
print(r.stdout, r.stderr[-2000:])
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_split.py"], capture_output=True, text=True)
print(r.stdout[-1500:])
```
If it prints `identical: False`, stop and tell Claude (the sklearn version may split differently).

Cell 6: one run (edit MODEL and SEED, rerun the cell for each run). Use a background-safe call so the
log survives a notebook UI disconnect:
```python
import subprocess, sys
MODEL, SEED = "efficientnet", 42          # efficientnet | vit ; seeds 42, 43, 44
log = f"/tmp/{MODEL}_s{SEED}.log"
with open(log, "w") as fh:
    r = subprocess.run([sys.executable, "scripts/run_seed.py", "--model", MODEL, "--seed", str(SEED)],
                       stdout=fh, stderr=subprocess.STDOUT)
print("exit code", r.returncode); print(open(log).read()[-3000:])
```
If the session dies after training finished, rerun the same cell with `"--skip-train"` added to the
argument list: it reuses `outputs/final/<run>/` and redoes only the evaluation.
If IG runs out of GPU memory on ViT, add `"--ig-internal-batch-size", "8"` (does not change results).

Order (decided Oct 8):
1. efficientnet 42, 43, 44 (start now).
2. vit 42 as a reproduction check of T2 (phase-1 lr 5e-2). Open
   `results/final/vit_s42/classification.json` and compare `test_macro_f1` with the paper's 0.748.
   Upload this run to Claude first; Claude judges "within seed noise" against the EfficientNet seed SD.
3. Only if it reproduces: vit 43, 44. If it does not, stop the ViT runs and report the gap.

Cell 7: package results for download
```python
import shutil
run = f"results/final/{MODEL}_s{SEED}"
shutil.copy(f"/tmp/{MODEL}_s{SEED}.log", run)
shutil.make_archive(f"/tmp/{MODEL}_s{SEED}", "zip", "results/final", f"{MODEL}_s{SEED}")
print("download:", f"/tmp/{MODEL}_s{SEED}.zip")
```
Download each ZIP (or push to a private HF dataset repo) and upload it to the cloud Claude session, which
unzips it into `results/final/` and runs `scripts/aggregate.py`. Checkpoints stay on molab in
`outputs/final/`; keep them until the paper is submitted (their sha256 is in each run folder).

Optional Cell 8 (once, any time after Cell 4): center-prior reference file
```python
import subprocess, sys
r = subprocess.run([sys.executable, "scripts/center_prior_baseline.py",
                    "--ids-from", "results/localization_efficientnet_best.csv",
                    "--mask-dir", "data/HAM10000_segmentations_lesion_tschandl",
                    "--output", "results/localization_center_prior.csv"], capture_output=True, text=True)
print(r.stdout, r.stderr[-2000:])
```
Upload `results/localization_center_prior.csv` too.

## 2. What each run produces (`results/final/<model>_s<seed>/`)
- `config_used.yaml`, `checkpoint_sha256.txt`, `test_image_ids.txt`, `training_log.csv` (per-epoch val)
- `classification.json` (selected val macro-F1, test macro-F1, balanced accuracy, per-class F1)
- `heatmaps_native/` (Grad-CAM or rollout .npy), `heatmaps_ig/` (.npy): first 40 sorted test ids only
- `faithfulness_{native,ig,random,center}.csv` and curve `.npz`
- `localization_{native,ig,center}.csv` (IoU, `pointing_hit_tol0`, `pointing_hit_tol15`, lesion area)
- `summary.json` (everything above summarized, git commit, versions, timings)

## 3. If something fails
- Seed-42 numbers far from the paper (test macro-F1 0.655 EN / 0.748 ViT): stop and report; checkpoint or
  config differs from what produced the paper (trap T2/T3).
- Session timeout mid-run: rerun that single MODEL/SEED (with `--skip-train` if the checkpoint exists).
