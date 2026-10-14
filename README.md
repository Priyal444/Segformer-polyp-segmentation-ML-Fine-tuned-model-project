# Polyp Segmentation with SegFormer-b0

Fine-tuning a 3.7M-parameter transformer to segment colorectal polyps in
colonoscopy frames, with an explicit look at how far the result generalises
beyond the dataset it was trained on.

**[Live demo](https://huggingface.co/spaces/USERNAME/SPACE)** · **[Model weights](https://huggingface.co/USERNAME/MODEL)**
*(the Space sleeps when idle — first load takes ~30s)*

> **Research demonstration only.** Not a medical device. Not validated on any
> clinical population. Not for diagnostic use or to inform patient care.

---

## Results

| Evaluation | Dice | IoU | Recall | Precision |
|---|---|---|---|---|
| Kvasir-SEG held-out test (in-domain) | — | — | — | — |
| CVC-ClinicDB (external, never trained on) | — | — | — | — |

*Fill these in after your runs. Report both rows even if the second one is bad —
especially if the second one is bad.*

The gap between those two rows is the most informative number in this
repository. In-domain scores on Kvasir-SEG are easy to push into the high 0.80s;
what a model does on images from different equipment, a different hospital, and
a different patient population is the question that actually decides whether it
is useful.

## Why these metrics

Accuracy is meaningless here. Polyps occupy a small minority of pixels, so a
model that predicts "no polyp" everywhere scores well above 95% while being
completely useless.

- **Dice / IoU** are averaged per image, not pooled over pixels. Pixel-pooled
  scores let a few large, easy polyps mask consistent failures on small ones.
- **Recall and precision are reported separately** because they are not
  interchangeable in a screening context. A missed polyp can become an interval
  cancer; a false positive costs a few seconds of a clinician's attention.
- **The threshold is a choice, not a constant.** `train.py` sweeps thresholds on
  the validation split (never the test split) so the recall/precision trade-off
  is visible rather than silently fixed at 0.5. The demo exposes the same slider.

## Method

- **Model** — SegFormer-b0 (`nvidia/mit-b0`), ImageNet-pretrained encoder with a
  freshly initialised decode head. Small enough to serve on a free CPU tier,
  which is a design constraint rather than an afterthought.
- **Loss** — BCE + soft Dice. BCE alone underfits the minority foreground; Dice
  alone gives unstable early gradients.
- **Optimisation** — AdamW with a 10× higher learning rate on the new decode head
  than on the pretrained encoder, OneCycle schedule, mixed precision.
- **Augmentation** — flips, shift/scale/rotate, and fairly aggressive colour
  jitter. Endoscopy illumination and white balance vary enormously between
  scopes, so colour augmentation is doing real generalisation work.

## Data

| Dataset | Images | Use | Source |
|---|---|---|---|
| Kvasir-SEG | 1,000 | train / val / test (80/10/10) | [SimulaMet](https://datasets.simula.no/kvasir-seg/) |
| CVC-ClinicDB | 612 | external validation only | [Kaggle](https://www.kaggle.com/datasets/balraj98/cvcclinicdb) |

Both download without credentialing. Deliberately avoided PhysioNet-hosted sets,
which require ethics training and multi-week approval.

## Reproduce

```bash
pip install -r requirements.txt

# 1. Download Kvasir-SEG and train
wget https://datasets.simula.no/downloads/kvasir-seg.zip && unzip kvasir-seg.zip
python train.py --data-root Kvasir-SEG --epochs 40

# 2. External domain-shift check on CVC-ClinicDB
python train.py --eval-only --ckpt best.pt \
    --ext-images CVC-ClinicDB/Original --ext-masks "CVC-ClinicDB/Ground Truth"

# 3. Camera corruption stress test (motion blur, low light, specular glare)
#    Runs on the same Kvasir-SEG held-out test split that train.py uses.
python stress_test.py --ckpt best.pt --data-root Kvasir-SEG
```

Or open `notebook.ipynb` in Colab and run the cells top to bottom.

Training time: *(fill in from your run — the notebook prints it)* on a free
Colab/Kaggle GPU at 384×384, batch size 8.

## Failure Mode & Robustness Analysis

1. **Size Stratification**:
   - Scores are broken down by polyp area at 384×384: Small ($< 1024$ px, under ~0.7% of the frame), Medium ($1024 - 9216$ px), and Large ($> 9216$ px, over 6.25%), each with its sample count ($N$), at both the 0.5 headline threshold and the validation-chosen threshold.
   - The Kvasir test split has only 100 images and Kvasir polyps are mostly large, so its small bin is tiny: N = *(fill in)*. CVC-ClinicDB (612 images) gives a more reliable read on small lesions: N = *(fill in)*.

2. **Camera Corruption Stress-Testing**:
   - Measures performance degradation under synthetic motion blur, low illumination, and specular glare across 5 severity levels, on the Kvasir-SEG held-out test split, so the curves isolate the corruption from the hospital shift. Corruptions are seeded per image, so reruns are identical.
   - Training already applies ±20% brightness jitter, so mild low-light levels (severity 1–2) are expected to barely affect the score.
   - `python stress_test.py --ckpt best.pt --data-root Kvasir-SEG` writes `stress_test_curve.png` and `stress_test_results.csv`. To stress-test an external set instead, pass `--images`/`--masks`; there, severity 0 is already out-of-distribution, so compare each curve against its own severity-0 point.

## Limitations

- Single-source training data from one Norwegian hospital's equipment. Nothing
  here establishes that it transfers to other scopes, other populations, or
  poorer bowel preparation.
- Still frames only. Real colonoscopy is video, where temporal consistency and
  latency matter and a per-frame model has no notion of either.
- No calibration guarantee. The sigmoid output is a score, not a probability,
  and should not be read as confidence.
- Ground-truth masks are single-annotator. Inter-observer disagreement on polyp
  boundaries is substantial, so the ceiling on Dice is well below 1.0.
