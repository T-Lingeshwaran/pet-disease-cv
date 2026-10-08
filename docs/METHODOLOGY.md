# Methodology

## Why this isn't "AI X-ray vision"

A phone camera captures visible light reflecting off fur/skin — it cannot
image bone. This project deliberately does not claim fracture detection.
Instead it screens for the *external, behavioral* signs of limb pain that
a worried owner might miss, and recommends a vet visit when those signs
appear. This is the same category of tool used in veterinary/livestock
research (see citations below) — just aimed at companion animals and a
phone camera instead of a farm or clinic setup.

## Pose estimation: SuperAnimal-Quadruped

- Model: DeepLabCut's SuperAnimal-Quadruped (Ye et al. 2023,
  https://arxiv.org/abs/2203.07436), an HRNet-w32 backbone trained on
  ~40-80K images spanning dogs, cats, horses, and other four-legged
  animals.
- Zero-shot: works on your video without you labeling any frames, because
  it was pretrained across many species/environments already.
- `video_adapt=True` runs a short self-supervised fine-tuning pass on your
  specific clip to stabilize keypoints for your exact lighting/background
  — this is a DeepLabCut feature, not a from-scratch training step.

## Gait metrics and what each one means clinically

| Metric | What it measures | Clinical rationale |
|---|---|---|
| Stance ratio | Fraction of the gait cycle each paw spends planted vs. swinging | An animal in pain typically unloads the sore leg early — shorter stance relative to the other three legs. Same principle used in [multi-cattle lameness pose research](https://www.nature.com/articles/s41598-023-31297-1), which tracked back-arch and head-position keypoints for lameness scoring. |
| Stride length symmetry | Left/right and front/hind stride length ratio | Healthy quadrupeds are close to bilaterally symmetric; a sore leg produces a shortened stride on that side. |
| Head-bob amplitude (normalized by body length) | Vertical head displacement over the gait cycle | Established equine lameness sign: the head *rises* when weight lands on a sore **front** leg (to shift weight off it) and *drops* when weight lands on a sore **hind** leg. This is the basis of the AAEP (American Association of Equine Practitioners) lameness grading scale, and has been used to identify forelimb lameness via head-nod trajectory analysis compared against sound-gait baselines. |

## Classification: rule-based v1 → trainable v2

**v1 (current, in `classify.py`)**: fixed thresholds on the three metrics
above. These are *reasonable starting points, not clinically validated
cutoffs* — there is no public "confirmed-limping pet" dataset to fit
thresholds against.

**v2 (once you have labeled clips)**: film/collect clips of pets with a
vet-confirmed limp and pets with a normal gait, run them through this
pipeline, and use `classify.train_classifier()` to fit a logistic
regression on the same feature vector instead of hand-set thresholds.
Even 20-40 labeled clips (split healthy/limping) is enough to get a real
accuracy number worth reporting — cite this as your evaluation section.

## Known limitations (be upfront about these in a presentation)

- Works best with a clean side-on view, full body in frame, decent
  lighting, and the animal actually moving (not standing still).
- Occlusion (fur covering legs, tail blocking a paw) will degrade
  keypoint confidence — the pipeline already drops low-confidence
  detections (`pcutoff`/`LIKELIHOOD_THRESHOLD`) rather than trusting them.
- Thresholds are uncalibrated until you have labeled data (see v2 above).
  Say this plainly rather than overclaiming an accuracy number you
  haven't measured.
- This screens for *lameness/discomfort signals*, not internal injuries,
  fractures, or conditions with no visible gait change.

## Dataset landscape (for context/citations, not used directly here)

Public pose-labeled datasets (Animal Pose, StanfordExtra, AP-10K) provided
the training data for SuperAnimal-Quadruped itself, but none of them are
labeled for lameness — hence the zero-shot pose + custom gait-metric
approach instead of trying to find/train on a "limping pet" dataset that
doesn't publicly exist.

## Citations

1. Ye, S., Filippova, A., Lauer, J., Vidal, M., Schneider, S., Qiu, T.,
   Mathis, A., & Mathis, M.W. (2023). *SuperAnimal pretrained pose
   estimation models for behavioral analysis*. arXiv:2203.07436.
2. Mathis, A., Mamidanna, P., Cury, K.M., Abe, T., Murthy, V.N.,
   Mathis, M.W., & Bethge, M. (2018). *DeepLabCut: markerless pose
   estimation of user-defined body parts with deep learning*.
   Nature Neuroscience, 21(9), 1281.
3. Kang, X. et al. (2023). *Deep learning pose estimation for
   multi-cattle lameness detection*. Scientific Reports.
   https://www.nature.com/articles/s41598-023-31297-1
4. Feuser, C. et al. — forelimb lameness demonstrated via head-nod
   trajectory analysis against the AAEP lameness scale (equine),
   referenced in: *Automatic gait analysis in canines using computer
   vision*, Frontiers in Veterinary Science.
