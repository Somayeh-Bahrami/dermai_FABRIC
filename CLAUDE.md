# CLAUDE.md: REED-AI @ ACM TRUST 2027 revision of the DermAI explainability paper

Working repo: the user's own private copy (code copied from the team repo without git history; the team
repo is never modified). Original course CLAUDE.md is kept at `docs/course_CLAUDE_original.md`.
Paper inputs live in `paper_inputs/` (Sep 21 PDF = content reference; `old_latex/` = Aug 2 source, outdated).

## Goal
Turn the CS 7643 report "Quantifying Explanation Faithfulness and Localization in CNNs vs Vision
Transformers for Skin Lesion Classification" into an anonymous, honest submission to the REED-AI track.
Venue (changed Oct 8 from FABRIC @ WACV 2027): REED-AI, "Responsible, Explainable, and Ethical-by-Design AI
in Clinical Applications", Emerging Area track of ACM TRUST 2027 (https://pitthexai.github.io/REED-AI/).
Abstract registration Oct 24, 2026; paper deadline Oct 31, 2026 (verify the time zone in CMT).
Notification Dec 31, 2026. Double-blind, Microsoft CMT. ACM sigconf two-column (acmart, review+anonymous).
Max 9 pages INCLUDING appendices; references and the required GenAI Usage Disclosure section do not count.
Scope: core study only (2 models x 3 seeds); no DINOv2. Say it is an evaluation study of conventional
pretrained models.
Fallback: if the gates below cannot all be met by Oct 29 evening, stop and recommend a later venue.

## Who is who
- Authors (all must approve submission, author list, order, and final PDF in writing BEFORE submission):
  Xiaoyan Xing, Somayeh Bahrami, Sebastian Gonzalez, Ka Wing Ariel Lee.
- Compute: GPU runs happen on molab (marimo notebooks). This cloud session has NO GPU.
  Claude writes and CPU-tests code here; the user runs GPU jobs on molab and uploads result files back.

## Ground rules (non-negotiable)
1. Never invent or estimate a number. Every number in the paper must trace to a file in `results/`
   listed in `results/PROVENANCE.md` (checkpoint id, seed, split seed, config, script, output file).
2. Validation-only selection. The test split is touched once per frozen final model. Do not report test
   scores of configurations that were not selected (remove the weighted-vs-unweighted test columns or move
   them to a clearly labeled post-hoc note).
3. All three axes (classification, faithfulness, localization) and all qualitative figures come from the
   SAME final checkpoints and the SAME 1,002 test images.
4. Claims must be no stronger than the evidence. Write "model-explainer system" unless the shared
   explainer isolates the backbone. No clinical-readiness, ABCDE, or Fitzpatrick claims.
5. Anonymity: no names, emails, affiliations, acknowledgments, course references (CS 7643, Georgia Tech),
   GitHub/Hugging Face links or user names in the paper, figures, PDF metadata, or supplement ZIP.
   Write "Code will be made publicly available." WACV strongly discourages links during review.
6. No em dashes anywhere (paper, code comments, commit messages).
7. Cite only papers verified to exist (title, authors, venue, year). Mark anything unverified as TODO and ask.
8. Keep code simple and reusable; extend existing modules (`src/dermai/*`) rather than duplicating them.

## Verified problems in the current paper (fix all)
P1. Center-prior baseline beats Grad-CAM on localization. A fixed centered heatmap (no model), same
    evaluator (top 20% pixels, masks resized to 224x224), on the same 1,002 test images:
    IoU 0.476 +/- 0.172, pointing game 0.995 (15 px tolerance) / 0.963 (exact pixel).
    Grad-CAM: IoU 0.466, PG 0.983. Rollout: IoU 0.401, PG 0.650. Mean lesion area 26% of image.
    Action: add center prior as a row in the localization table (and as a faithfulness ordering control),
    rewrite localization claims relative to it. This is the paper's most important honest finding.
P2. Attention rollout is class-agnostic (Chefer et al., CVPR 2021: "the result of rollout is fixed given an
    input sample, regardless of the target class"; confirmed in `src/dermai/attrollout.py`, target class unused).
    Deletion/insertion on the predicted-class probability structurally favors class-specific Grad-CAM.
    Action: add one shared, class-specific explainer on both models (Integrated Gradients via captum).
P3. Pointing-game definition mismatch: paper says "inside the mask", code default `pointing_tolerance=15`.
    Action: report exact (tolerance 0) as primary, 15 px as secondary, and say so.
P4. Single seed. Action: 3 training seeds per model (42, 43, 44) with a FIXED split (see Trap T1).
P5. Loss-selection text shows test scores for the unselected loss. Action: validation-only reporting.
P6. Course-report residue: sections 4.2 to 4.5 (rubric answers), lay-audience intro, Team Contributions.
P7. Missing related work: Chefer et al. CVPR 2021; Komorowski, Baniecki, Biecek CVPRW 2023 ("Towards Evaluating
    Explanations of Vision Transformers for Medical Imaging"); Saarela et al. 2022 Applied Sciences
    (fidelity of IG/LIME on HAM10000); Petsiuk et al. RISE (already cited).
P8. Reference [5] Kuntal & Bhat 2025 (ISCON) could not be verified. Verify or remove.
P9. Capacity and pretraining mismatch (5.3M vs 86M params; ImageNet-1k vs ImageNet-21k pretraining): report
    in a model table, treat as a limitation, never as a cause.

## Code traps (check before any GPU run)
T1. `src/dermai/data.py` uses `random_state=self.seed` for the lesion-grouped split, and train.py passes
    `config.seed`. Changing the training seed CHANGES THE TEST SET. Fix: add `split_seed: 42` to Config,
    use it in DataModule; keep `seed` for training only. Assert the 1,002 test image ids equal
    `results/localization_efficientnet_best.csv` ids.
T2. `configs/vit.yaml` phase-1 lr is 5.0e-2, outside the paper's reported grid (max 1e-2). Confirm with the
    team which value produced the HF checkpoint `xyxing/dermai_vit_trained`; fix config or paper.
T3. Confirm the HF checkpoints (`xyxing/dermai_efficientnet-b0_trained`, `xyxing/dermai_vit_trained`) were
    trained from the current configs with seed 42; if not, retrain seed 42 too.
T4. Rollout needs `attn_implementation="eager"` (transformers 5.x).

## Work plan (in order)
Day 1 (CPU here + authorship)
- [ ] Authorship emails sent (draft in ADMIN.md, not yet sent); written approvals tracked in `ADMIN.md`.
- [x] Work on short feature branches in the user's repo; fix T1; add `scripts/center_prior_baseline.py` (localization) and a center-prior
      ordering option for deletion/insertion.
- [x] Add `src/dermai/integrated_gradients.py` (captum IntegratedGradients, 32 steps, baseline = mean-fill
      in normalized space i.e. zeros, target = predicted class, attribution = abs sum over channels,
      output 224x224 npy with the same filename scheme as Grad-CAM) and `explain_ig.py` CLI.
- [x] One driver `scripts/run_seed.py --model {efficientnet,vit} --seed S` that: trains, evaluates test,
      writes Grad-CAM or rollout heatmaps, IG heatmaps, faithfulness (incl. random and center-prior controls),
      localization (tolerance 0 and 15), and a JSON summary to `results/final/<model>_s<S>/`.
- [x] CPU smoke test with `--limit 8` and 1 tiny epoch.
Days 2 to 3 (molab GPU, see MOLAB_GUIDE.md)
- [ ] 6 runs: {efficientnet, vit} x seeds {42, 43, 44}. Upload each `results/final/<run>/` folder back.
- [ ] `scripts/aggregate.py`: mean +/- SD over seeds, paired bootstrap CIs over images (10,000 draws),
      writes LaTeX tables + `results/PROVENANCE.md`.
Days 3 to 5 (writing)
- [x] ACM sigconf template in `paper/` (review, anonymous); compile on Overleaf.
- [ ] New title suggestion: "Do Explanation Benchmarks Measure the Model? Center Bias and Class-Agnostic
      Attribution in Dermoscopy Explainability" (or the more conservative "Faithfulness and Lesion
      Localization of Explanations for Pretrained Vision Models in Dermoscopy"). Pick with the team.
- [ ] Structure: Intro (validity of explanation evaluation for responsible clinical AI) / Related work / Protocol /
      Results (classification, faithfulness, localization with center prior, shared explainer, seeds) /
      Qualitative failure cases (image, mask, Grad-CAM, rollout, IG, center prior) / Limitations & ethics
      (HAM10000 de-identified public data from Austria and Australia; no skin-type labels; research only) /
      Recommendations for reporting / Conclusion / GenAI Usage Disclosure.
- [ ] Anonymized supplement ZIP (code + instructions, no names in LICENSE/headers/git metadata).
- [ ] Final gates below.

Decided Oct 8: no extra backbone (DINOv2 dropped); the paper is an evaluation study of conventional
pretrained models.

## Final gates (all must be true to submit)
- [ ] All four authors approved in writing (ADMIN.md).
- [ ] Anonymous; <= 9 pages including appendices (references and GenAI disclosure excluded); ACM sigconf;
      PDF metadata clean.
- [ ] GenAI Usage Disclosure present, wording checked against the CMT/ACM instructions, approved by all authors.
- [ ] Abstract registered in CMT by Oct 24; REED-AI track selected.
- [ ] No test result used for selection; split ids verified identical across seeds.
- [ ] Every number traced in results/PROVENANCE.md; same checkpoints across all axes.
- [ ] 3 seeds, mean +/- SD and paired CIs; no claim the uncertainty does not support.
- [ ] Center-prior row and shared explainer (IG) in main tables.
- [ ] Pointing-game tolerance stated; references verified; no em dashes.
- [ ] Not under review at any other archival venue.
