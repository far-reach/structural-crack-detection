# Structural Crack Detection

Automated multi-class structural damage detection and segmentation for UAV-based infrastructure inspection using deep learning.

Reference implementation for Abtahi, S.F. (2026), *Automated Multi-Class Structural Damage Detection and Segmentation for Unmanned Aerial Vehicle-Based Infrastructure Inspection Using Deep Learning*, engrXiv preprint, [doi.org/10.31224/7685](https://doi.org/10.31224/7685).

---

## Correction notice

**The results reported in v1.0 are superseded. Do not cite them as current.**

After the preprint was posted I audited this code and found two problems.

**1. A class-order error.** Class indices were applied inconsistently between training and the Stage 3 flight simulation. This mislabelled the simulation result and two figures in the preprint.

**2. Source confounding in the training data.** Each damage class was drawn from a single image source: cracks and intact surfaces from SDNET2018, corrosion and spalling from a separate Kaggle set. Because class and source were perfectly correlated, the network could separate the classes by recognising the source rather than the damage. The reported accuracy was therefore measuring the wrong thing, and it does not transfer to imagery from any other source.

Every affected result has been regenerated under a corrected protocol, and the cost of each protocol flaw has been measured rather than assumed. A second manuscript documents the audit in full and rebuilds the method as a two-tier scheme: a lightweight vision stage screens every zone, a single ultrasonic pulse-echo reading confirms only the zones it flags, and the two kinds of evidence are fused per zone. Both stages are evaluated on public datasets from independent laboratories, with test specimens held out by source.

That manuscript is in preparation. This notice will be updated with its identifier when it is posted.

**Which release to use.** `v1.0` is the code as published with the preprint, kept for reproducibility of the record. Later releases carry the corrected protocol. If you are reproducing the preprint, use `v1.0` and read this notice first. If you are building on the method, do not start from `v1.0`.

---

## Pipeline

A three-stage pipeline for bridges, offshore platforms and other civil infrastructure.

| Stage | Task | Model |
| --- | --- | --- |
| 1 | Four-class damage classification: crack, corrosion, spalling, intact | ResNet50, transfer-learned from ImageNet |
| 2 | Pixel-level semantic segmentation | DeepLabV3+ with a ResNet50 backbone |
| 3 | UAV grid inspection simulation, boustrophedon traversal of a 4x5 grid | Stage 1 and 2 applied per zone |

All stages were trained and benchmarked on CPU hardware, with no GPU required at inference.

## v1.0 reported results, superseded

Kept for the record. See the correction notice above before using any of these figures.

| Metric | v1.0 value |
| --- | --- |
| Stage 1 classification accuracy | 85.69% |
| Stage 2 mIoU, three-class | 0.789 |
| Stage 2 mIoU, four-class | 0.645 |
| Stage 3 zones detected | 16 of 20 |
| Stage 1 inference | 16.29 ms per frame, CPU |

The crack class was the weakest throughout, which the preprint attributed to low pixel density. That analysis still holds. The accuracy figures do not.

## Repository structure

```
src/                  pipeline source
results/plots/        generated figures
train_multiclass.py   Stage 1 training
train_segmentation.py Stage 2 training
drone_simulation.py   Stage 3 flight simulation
failure_analysis.py   per-class failure analysis
gradcam.py            Grad-CAM attribution
per_class_iou.py      segmentation metrics
inference_time.py     timing benchmarks
requirements.txt      dependencies
```

## Requirements

Python 3.9 or later.

```
pip install -r requirements.txt
```

PyTorch 2.1.0, torchvision, numpy, matplotlib, scikit-learn, Pillow, opencv-python.

## Datasets

Not bundled. Download separately and place under the paths given in `setup_dataset.py`.

- **SDNET2018**, crack and intact classes
- **Kaggle corrosion and spalling set**, Stages 1 and 2
- **Kaggle UAV crack set**, four-class Stage 2

Note that these sources are the origin of the confounding described above. Any protocol built on them must hold out by source, not only by image.

## Pretrained weights

Three checkpoints are attached to the `v1.0` release: the Stage 1 ResNet50 classifier and two DeepLabV3+ segmentation models. Place them in a `models/` directory at the repository root. These weights carry the v1.0 protocol and its flaws.

## Citation

```bibtex
@misc{abtahi2026uav,
  author = {Abtahi, Seyed Farhad},
  title  = {Automated Multi-Class Structural Damage Detection and Segmentation
            for Unmanned Aerial Vehicle-Based Infrastructure Inspection
            Using Deep Learning},
  year   = {2026},
  note   = {engrXiv preprint},
  doi    = {10.31224/7685}
}
```

If you are citing the corrected method rather than the original preprint, contact the author for the current manuscript.

## Author

Seyed Farhad Abtahi, Independent Researcher, Istanbul, Türkiye
ORCID [0009-0007-8865-2300](https://orcid.org/0009-0007-8865-2300)

## Licence

MIT. See `LICENSE`.
