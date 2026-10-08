# First message to paste into the new cloud session

Select your own repo (the one created from `fabric-workspace/`) on branch `main`.
Repo layout: code at the root, `CLAUDE.md` at the root, `docs/MOLAB_GUIDE.md`,
`scripts/center_prior_baseline.py`, `paper_inputs/latest_paper_Sep21.pdf` (content reference),
`paper_inputs/old_latex/` (Aug 2 LaTeX source, outdated: port text and numbers from the Sep 21 PDF),
`paper_inputs/Needed_edition.docx`, and `paper_inputs/wacv2027_author_kit/` (WACV 2027 author kit).

---

You are helping me revise our CS 7643 group paper into an anonymous FABRIC @ WACV 2027 workshop submission
(deadline Oct 12, 2026). Read CLAUDE.md first and follow it exactly; it lists verified problems (P1 to P9),
code traps (T1 to T4), the work plan, and the final gates.

Context:
- No GPU here. You write and CPU-test code; I run GPU jobs on molab using MOLAB_GUIDE.md and upload the
  result ZIPs back to you.
- Never invent numbers. If a number is not in a results file, write TODO and tell me.
- Ask me before any decision that changes the scientific claims, the title, or the author list.
- No em dashes anywhere.

Do now, in this order, and keep a task list:
1. Verify the repo state: branch, configs, and traps T1 to T4. Report what you find before changing code.
2. Fix T1 (fixed split_seed) and add a test that the 1,002 test image ids match
   results/localization_efficientnet_best.csv.
3. Add scripts/center_prior_baseline.py to the repo (uploaded file), integrated_gradients.py + explain_ig.py,
   center-prior ordering for deletion/insertion, tolerance 0 and 15 localization, scripts/run_seed.py,
   scripts/aggregate.py. CPU smoke-test everything with --limit 8.
4. Tell me exactly which molab cells to run, and stop until I upload results.
5. In parallel, set up the WACV 2027 template from the author kit, port the latest LaTeX, remove course
   residue and identifying information, and draft the new structure with TODO placeholders for numbers.
6. Draft the authorship-approval email to my three coauthors (short, includes title options, target venue,
   deadline, and asks for written OK on author order and final PDF).
