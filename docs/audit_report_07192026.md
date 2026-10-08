# Fine-Tuning Configuration Audit — 2026-07-19

## Original Prompt

Please review all Python files in this project and do the following:

1. FINE-TUNING CONFIGURATION AUDIT
   Find all training config values (learning rate, dropout, weight decay,
   warmup steps, label smoothing, stochastic depth / drop_connect_rate)
   for both EfficientNet-B0 and ViT-B/16. For each value found, tell me:
   - What it is currently set to
   - Where in the code it is set (file + line)
   - Whether it looks appropriate for fine-tuning on a small medical
     dataset (~8,000 training images, 7 classes) vs. what the original
     papers recommend

2. BACKBONE FREEZING AUDIT
   Find all code that controls which layers are frozen or unfrozen —
   look for calls like requires_grad, set_backbone_trainable,
   named_parameters, or any manual freeze/unfreeze logic. For each:
   - Show me exactly which layers are frozen in Stage 1 (head-only)
   - Show me exactly which layers are unfrozen in Stage 2 (full model)
   - Confirm whether the classification head is correctly excluded from
     freezing in Stage 1 so it trains from the start
   - Flag any layers that should be frozen but aren't, or vice versa

3. CLASSIFICATION HEAD AUDIT
   For each model, show me the classification head definition and confirm:
   - The output dimension matches num_classes=7
   - Dropout is applied before the final linear layer
   - The head is freshly initialized (not loaded from pretrained weights)

4. DIFFERENTIAL LEARNING RATE CHECK
   Check whether the optimizer uses different learning rates for the
   backbone vs. the classification head. If it uses a single LR for
   everything in Stage 2, flag this as something to consider changing.

5. ViT-SPECIFIC CHECK
   For the ViT-B/16 model specifically:
   - Confirm a learning rate warmup scheduler is in place
   - Confirm AdamW (not Adam) is used
   - Confirm weight decay is set (ideally 0.01–0.05 for fine-tuning)
   - Flag if any of these are missing, since ViT fine-tuning on small
     datasets is sensitive to all three

Please structure your response as one section per audit item above.
For each issue found, show the relevant code snippet and give a concrete
suggested fix. Do not make any changes yet — review and report only.

---

## Findings

Reviewed all project Python files (excluding `venv/`). This is a small, clean
codebase — the entire training configuration lives in `src/dermai/config.py`,
`src/dermai/models.py`, `src/dermai/trainer.py`, and the two YAML files
(`configs/efficientnet.yaml`, `configs/vit.yaml`).

### 1. Fine-Tuning Configuration Audit

| Value | EfficientNet-B0 | ViT-B/16 | Location |
|---|---|---|---|
| Stage 1 LR (head) | `1.0e-3` | `1.0e-3` | configs/efficientnet.yaml:12, configs/vit.yaml:12 |
| Stage 2 LR (full) | `1.0e-4` | `2.0e-5` | configs/efficientnet.yaml:13, configs/vit.yaml:13 |
| Weight decay | `0.01` | `0.01` | config.py:25 (default), overridden identically in both YAMLs |
| Warmup ratio | `0.1` (of each phase's steps) | `0.1` | config.py:26, applied in trainer.py:51 |
| Label smoothing | not set anywhere | not set anywhere | `nn.CrossEntropyLoss` at trainer.py:24 takes no `label_smoothing` arg |
| Dropout / drop_connect | inherited from checkpoint config: `dropout_rate=0.2`, `drop_connect_rate=0.2` | inherited from checkpoint config (standard `google/vit-base-patch16-224` ships `hidden_dropout_prob=0.0`, `attention_probs_dropout_prob=0.0`) | not overridden in code at all — see below |

**Assessment vs. papers:**

- EfficientNet: `dropout_rate=0.2` / `drop_connect_rate=0.2` (confirmed from the
  cached checkpoint's `config.json`) matches the original EfficientNet paper's
  B0 spec exactly (paper scales dropout 0.2→0.5 as B0→B7). This is
  appropriate — nothing to change.
- ViT: the released checkpoint has zero dropout
  (`hidden_dropout_prob=0.0`, `attention_probs_dropout_prob=0.0`), and this
  transformers version's `ViTConfig` doesn't even expose a `drop_path_rate`
  (stochastic depth) field. So ViT fine-tunes with no internal regularization
  at all. On ~8,000 images / 7 classes this is a real overfitting risk — the
  ViT paper itself uses dropout when fine-tuning on small datasets. Nothing in
  the code sets this; it would need to be passed explicitly:

  ```python
  # src/dermai/models.py — ModelFactory.build
  return AutoModelForImageClassification.from_pretrained(
      model_id,
      num_labels=len(CLASSES),
      hidden_dropout_prob=0.1,
      attention_probs_dropout_prob=0.1,
      ...
  )
  ```

- Label smoothing: absent for both models. For a 7-class, class-imbalanced
  medical dataset this is a cheap, well-supported regularizer
  (`nn.CrossEntropyLoss(weight=..., label_smoothing=0.1)`). Worth
  considering, not currently used.
- LR magnitudes look reasonable relative to typical fine-tuning practice
  (head LR higher, backbone LR much lower in stage 2), but see §4 — both
  models currently use one flat LR per phase rather than a backbone/head
  split in stage 2.

### 2. Backbone Freezing Audit

All freezing logic lives in one function:

```python
# src/dermai/models.py:36-40
@staticmethod
def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        if not name.startswith("classifier"):
            parameter.requires_grad = trainable
```

Called once per phase in trainer.py:44:

```python
ModelFactory.set_backbone_trainable(self.model, phase.unfreeze_backbone)
```

- **Stage 1 (`head`, `unfreeze_backbone: false`)**: every parameter whose name
  doesn't start with `"classifier"` gets `requires_grad = False`. Since the
  function only ever touches non-`classifier` params, the classifier's
  `requires_grad` (True by default at construction) is never touched here —
  it stays trainable. Head trains from step 1, backbone is fully frozen.
- **Stage 2 (`full`, `unfreeze_backbone: true`)**: the same non-`classifier`
  params get `requires_grad = True`, unfreezing the entire backbone.
  Classifier remains `True` throughout.

Both HF classes (`ViTForImageClassification` and
`EfficientNetForImageClassification`) name their output layer
`self.classifier`, so the `name.startswith("classifier")` check correctly
matches the head for both architectures — verified against the installed
`transformers` 5.13.0 source. No layer is frozen/unfrozen incorrectly for
either model.

One thing to flag: this is a blunt all-or-nothing freeze — there's no
gradual/layer-wise unfreezing (e.g. unfreeze last N transformer blocks
first). Not wrong, just worth knowing it's binary.

### 3. Classification Head Audit

Head construction happens inside `transformers`, not this repo, driven by
models.py:19-26:

```python
return AutoModelForImageClassification.from_pretrained(
    model_id,
    num_labels=len(CLASSES),          # len(CLASSES) == 7
    id2label={...}, label2id=LABEL_TO_INDEX,
    ignore_mismatched_sizes=True,
)
```

`CLASSES` has 7 entries (data.py:12), so `num_labels=7` for both models.

**ViT head** (`modeling_vit.py:530`):

```python
self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
```

- Output dim: `config.num_labels = 7`. Correct.
- No dropout before the final linear layer — `ViTForImageClassification`
  applies dropout only inside the encoder (via `hidden_dropout_prob`, which
  is 0.0 on this checkpoint per §1), not on the pooled feature before
  `self.classifier`. So effectively zero dropout regularizes the ViT head.

**EfficientNet head** (`modeling_efficientnet.py:523-524`):

```python
self.dropout = nn.Dropout(p=config.dropout_rate)
self.classifier = nn.Linear(config.hidden_dim, self.num_labels) if self.num_labels > 0 else nn.Identity()
```

- Output dim: `config.num_labels = 7`. Correct.
- Dropout (`p=0.2`) is applied before the final linear layer. Correct.

**Fresh initialization**: `ignore_mismatched_sizes=True` (models.py:25) means
whenever `num_labels` (7) mismatches the pretrained checkpoint's original
head size (1000-way ImageNet for both base checkpoints), HF discards the
pretrained classifier weights and randomly reinitializes that layer per
`post_init()`. Confirmed both `ViTForImageClassification.__init__` and
`EfficientNetForImageClassification.__init__` call `self.post_init()` right
after defining `self.classifier`. Head is freshly initialized for both
models.

**Suggested fix** for the ViT gap: not a one-line fix since HF's
`ViTForImageClassification` doesn't expose a pre-classifier dropout knob —
would require either overriding `classifier_dropout` in `ViTConfig` or
subclassing to insert `nn.Dropout` before `self.classifier`. Flagging for
attention rather than proposing an in-place patch.

### 4. Differential Learning Rate Check

No differential LR exists. Both stages use one flat `phase.lr` for every
trainable parameter:

```python
# trainer.py:48-49
optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                  lr=phase.lr, weight_decay=self.config.weight_decay)
```

In Stage 2, backbone and head share `phase.lr` (`1.0e-4` for EfficientNet,
`2.0e-5` for ViT) — the head doesn't get a separately-tuned rate once the
backbone unfreezes, even though the head was originally suited to a much
higher LR (`1.0e-3`) in Stage 1. This is a reasonable thing to reconsider: a
discriminative/param-group LR (e.g. head at `phase.lr * 10`) is a common
practice for exactly this two-stage fine-tuning setup and is cheap to add:

```python
# concrete suggested fix (not applied)
backbone_params = [p for n, p in self.model.named_parameters() if not n.startswith("classifier") and p.requires_grad]
head_params = [p for n, p in self.model.named_parameters() if n.startswith("classifier") and p.requires_grad]
optimizer = AdamW([
    {"params": backbone_params, "lr": phase.lr},
    {"params": head_params, "lr": phase.lr * 10},
], weight_decay=self.config.weight_decay)
```

### 5. ViT-Specific Check

| Requirement | Status | Evidence |
|---|---|---|
| LR warmup scheduler | Present | trainer.py:51 — `get_linear_schedule_with_warmup(optimizer, int(config.warmup_ratio * steps), steps)`, `warmup_ratio=0.1` in configs/vit.yaml:7. Applied fresh each phase (Stage 2 gets its own 10%-of-phase warmup). |
| AdamW (not Adam) | Correct | trainer.py:5 — `from torch.optim import AdamW`, used at line 48. |
| Weight decay 0.01–0.05 | In range | `weight_decay: 0.01` in configs/vit.yaml:6 — at the low end of the recommended range but valid. Note: it's applied to all trainable params uniformly, including biases/LayerNorm weights, which the original ViT/BERT-style recipes typically exclude from weight decay. Not asked for explicitly, but worth flagging alongside the dropout gap below. |
| Dropout / drop_path present | Missing | As detailed in §1/§3: checkpoint config has `hidden_dropout_prob=0.0`, `attention_probs_dropout_prob=0.0`, and this `transformers` version's `ViTConfig` has no `drop_path_rate` field at all. Combined with no dropout before the classifier head, ViT fine-tunes with zero regularization, which is the riskiest gap found in this audit given the dataset size (~8,000 images, 7 classes). |

**Summary of the one real gap**: ViT has warmup, AdamW, and weight decay
correctly configured, but is running with no dropout/stochastic depth
anywhere in the network. That's the item most worth addressing before the
next training run.
