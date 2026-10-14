"""
Gradio demo for the polyp segmentation model. Runs on a free CPU Space.

Local:  python app.py
Colab:  GRADIO_SHARE=1 python app.py   (prints a temporary public link)
Space:  push this file + requirements.txt, set MODEL_REPO to your HF model repo.
"""

import os

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import SegformerForSemanticSegmentation

MODEL_REPO = os.environ.get("MODEL_REPO", "your-username/segformer-b0-kvasir-polyp")
CKPT_NAME = os.environ.get("CKPT_NAME", "best.pt")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DISCLAIMER = """
> **Research demonstration only.** This is not a medical device, is not
> validated on any clinical population, and must not be used for diagnosis or
> to inform patient care. It was fine-tuned on 1,000 images from a single
> source and its behaviour on other equipment, populations, or bowel-prep
> quality is unknown.
"""


def load_model():
    path = CKPT_NAME if os.path.exists(CKPT_NAME) else hf_hub_download(MODEL_REPO, CKPT_NAME)
    ckpt = torch.load(path, map_location="cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(
        ckpt.get("encoder", "nvidia/mit-b0"), num_labels=1, ignore_mismatched_sizes=True
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    default_thresh = float(ckpt.get("best_threshold", 0.5))
    return model, int(ckpt.get("size", 384)), default_thresh


MODEL, SIZE, DEFAULT_THRESH = load_model()


def preprocess(image, size):
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


@torch.no_grad()
def segment(image, threshold, opacity):
    if image is None:
        return None, "Upload an endoscopy frame to begin."

    h, w = image.shape[:2]
    logits = MODEL(pixel_values=preprocess(image, SIZE)).logits
    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    prob = torch.sigmoid(logits)[0, 0].numpy()
    mask = prob > threshold

    overlay = image.copy()
    if mask.any():
        tint = np.zeros_like(image)
        tint[mask] = (255, 60, 60)
        overlay = cv2.addWeighted(overlay, 1.0, tint, opacity, 0)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)

    coverage = mask.mean() * 100
    peak = float(prob.max())
    mean_conf = float(prob[mask].mean()) if mask.any() else 0.0

    if not mask.any():
        note = (f"No region above threshold {threshold:.2f}. "
                f"Highest pixel probability was {peak:.3f} — lowering the threshold "
                f"will surface weaker candidates at the cost of false positives.")
    else:
        note = (f"Flagged **{coverage:.2f}%** of the frame across "
                f"{len(contours)} region(s).\n\n"
                f"Mean confidence inside the mask: **{mean_conf:.3f}**  |  "
                f"peak: **{peak:.3f}**")

    return overlay, note


with gr.Blocks(title="Polyp Segmentation — SegFormer-b0") as demo:
    gr.Markdown("# Polyp segmentation in colonoscopy frames")
    gr.Markdown(
        "SegFormer-b0 (3.7M parameters) fine-tuned on Kvasir-SEG. "
        "Upload a frame and the model outlines candidate polyp regions."
    )
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Endoscopy frame")
            threshold = gr.Slider(
                0.05, 0.95, value=DEFAULT_THRESH, step=0.05, label="Decision threshold",
                info="Lower = higher sensitivity, more false positives. "
                     "In screening, a missed polyp usually costs more than a false alarm.",
            )
            opacity = gr.Slider(0.1, 0.8, value=0.4, step=0.05, label="Overlay opacity")
            btn = gr.Button("Segment", variant="primary")
        with gr.Column():
            out = gr.Image(label="Prediction")
            info = gr.Markdown()

    btn.click(segment, [inp, threshold, opacity], [out, info])
    inp.change(segment, [inp, threshold, opacity], [out, info])
    threshold.release(segment, [inp, threshold, opacity], [out, info])

    if os.path.isdir("examples"):
        gr.Examples(
            examples=[[f"examples/{f}"] for f in sorted(os.listdir("examples"))],
            inputs=inp,
        )

if __name__ == "__main__":
    # GRADIO_SHARE=1 gives a temporary public link (needed in Colab); Spaces ignore it.
    demo.launch(share=os.environ.get("GRADIO_SHARE") == "1")
