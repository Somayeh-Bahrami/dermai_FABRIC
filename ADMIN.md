# Admin: authorship approvals for the REED-AI @ ACM TRUST 2027 submission

Not part of the supplement. Keep out of the anonymized ZIP.

## Approval tracker (all four required in writing before submission)

| Author (current order) | Author list + order OK | Title OK | GenAI disclosure OK | Final PDF OK | Not under review elsewhere | Date / link to written reply |
|---|---|---|---|---|---|---|
| 1. Xiaoyan Xing | | | | | |
| 2. Somayeh Bahrami | | | | | |
| 3. Sebastian Gonzalez | | | | | |
| 4. Ka Wing Ariel Lee | | | | | |

Order above is the order on the Sep 21 draft; change it only if the team agrees in writing.

## Draft email (send to the three coauthors)

Subject: OK needed by Oct 22: submitting our HAM10000 explainability paper to REED-AI @ ACM TRUST 2027

Hi all,

I would like to submit a revised version of our CS 7643 paper to REED-AI (Responsible, Explainable, and
Ethical-by-Design AI in Clinical Applications), an Emerging Area track of ACM TRUST 2027
(https://pitthexai.github.io/REED-AI/). It is double-blind, ACM two-column format, up to 9 pages including
appendices. Abstract registration is Oct 24 and the paper deadline is Oct 31, 2026 (CMT). Notification is
Dec 31; the conference is in March 2027.

What changes from the course version:
- Three training seeds per model on a fixed test split, with confidence intervals.
- A model-free "center prior" baseline (a fixed centered heatmap). In a preliminary check on our test set it localizes about
  as well as Grad-CAM, so the localization claims will be rewritten relative to it.
- Integrated Gradients added as one shared explainer on both models, since attention rollout ignores the
  target class.
- Course material (rubric sections, team contributions, GitHub and Hugging Face links) removed for anonymity.
- A short "recommendations for reporting" section and an expanded ethics section, to fit the track.
- The venue requires a GenAI usage disclosure. We used an AI coding assistant for evaluation code and for
  drafting text; the draft disclosure is in paper/sec/9_genai_disclosure.tex. Please read and approve it.

Title options (please vote or suggest):
  A. Do Explanation Benchmarks Measure the Model? Center Bias and Class-Agnostic Attribution in
     Dermoscopy Explainability
  B. Faithfulness and Lesion Localization of Explanations for Pretrained Vision Models in Dermoscopy
  C. Are Dermoscopy Explanation Benchmarks Fit for Responsible Clinical AI? Center Bias and
     Class-Agnostic Attribution

Xiaoyan: could you confirm that the ViT phase-1 learning rate of 5e-2 in configs/vit.yaml (your Jul 30
commit) is the value behind the 0.748 test macro-F1? We are rerunning seed 42 to check it.

Could each of you reply in writing by Oct 22, 6 pm with:
1. OK to submit, and OK with the author order: Xiaoyan Xing, Somayeh Bahrami, Sebastian Gonzalez,
   Ka Wing Ariel Lee (or the order you propose).
2. Your title choice (A, B or C), and OK on the GenAI disclosure text.
3. Confirmation that this work is not under review at any other archival venue.
I will register the abstract by Oct 24, circulate the final PDF by Oct 28, and will only submit after
each of you replies "OK to submit this PDF".

Thanks,
[Your name]
