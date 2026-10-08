# Admin: authorship approvals for the FABRIC @ WACV 2027 submission

Not part of the supplement. Keep out of the anonymized ZIP.

## Approval tracker (all four required in writing before submission)

| Author (current order) | Author list + order OK | Title OK | Final PDF OK | Not under review elsewhere | Date / link to written reply |
|---|---|---|---|---|---|
| 1. Xiaoyan Xing | | | | | |
| 2. Somayeh Bahrami | | | | | |
| 3. Sebastian Gonzalez | | | | | |
| 4. Ka Wing Ariel Lee | | | | | |

Order above is the order on the Sep 21 draft; change it only if the team agrees in writing.

## Draft email (send to the three coauthors)

Subject: OK needed by Oct 10: submitting our HAM10000 explainability paper to FABRIC @ WACV 2027

Hi all,

I would like to submit a revised version of our CS 7643 paper to FABRIC 2027, the Workshop on Foundation
AI for Biomedical Reasoning, Imaging, and Cognition at WACV 2027
(https://sites.google.com/view/fabric-wacv/call-for-papers). It is double-blind, 8 pages plus references,
and the submission deadline is Oct 12, 2026 (OpenReview). Notification is Oct 30.

What changes from the course version:
- Three training seeds per model on a fixed test split, with confidence intervals.
- A model-free "center prior" baseline (a fixed centered heatmap). In a preliminary check on our test set it localizes about
  as well as Grad-CAM, so the localization claims will be rewritten relative to it.
- Integrated Gradients added as one shared explainer on both models, since attention rollout ignores the
  target class.
- Course material (rubric sections, team contributions, GitHub and Hugging Face links) removed for anonymity.

Title options (please vote or suggest):
  A. Do Explanation Benchmarks Measure the Model? Center Bias and Class-Agnostic Attribution in
     Dermoscopy Explainability
  B. Faithfulness and Lesion Localization of Explanations for Pretrained Vision Models in Dermoscopy

Xiaoyan: could you confirm that the ViT phase-1 learning rate of 5e-2 in configs/vit.yaml (your Jul 30
commit) is the value behind the 0.748 test macro-F1? We are rerunning seed 42 to check it.

Could each of you reply in writing by Oct 10, 6 pm with:
1. OK to submit, and OK with the author order: Xiaoyan Xing, Somayeh Bahrami, Sebastian Gonzalez,
   Ka Wing Ariel Lee (or the order you propose).
2. Your title choice (A or B).
3. Confirmation that this work is not under review at any other archival venue.
I will circulate the final PDF on Oct 11 and will only submit after each of you replies "OK to submit
this PDF".

Thanks,
[Your name]
