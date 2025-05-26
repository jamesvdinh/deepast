---
title: "Challenges"
hide_table_of_contents: true
---

import PrizeCard from '@site/src/components/PrizeCard';

<head>
  <html data-theme="dark" />

  <meta
    name="description"
    content="A $1,000,000+ machine learning and computer vision competition"
  />

  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://scrollprize.org" />
  <meta property="og:title" content="Vesuvius Challenge" />
  <meta
    property="og:description"
    content="A $1,000,000+ machine learning and computer vision competition"
  />
  <meta
    property="og:image"
    content="https://scrollprize.org/img/social/opengraph.jpg"
  />

  <meta property="twitter:card" content="summary_large_image" />
  <meta property="twitter:url" content="https://scrollprize.org" />
  <meta property="twitter:title" content="Vesuvius Challenge" />
  <meta
    property="twitter:description"
    content="A $1,000,000+ machine learning and computer vision competition"
  />
  <meta
    property="twitter:image"
    content="https://scrollprize.org/img/social/opengraph.jpg"
  />
</head>

This page lists all the competition challenges:

## 1. Task: Image segmentation + annotation

### The Challenge

#### Refining segmentation strategy

To develop this answer, I'll focus on the dataset size needed for segmentation. With 200 classes of signs and variation in handwriting styles, each photo of tablets showing multiple signs means we need a solid number of annotated segments. A starting point would be around 1,000 instances per class for deep learning. I'll aim for a minimum of 60,000 segments, but top classes might require more. I'll also need to define proper development sets: train, validation, and test, and consider using active learning to improve iteratively. Should multi-sign segments be labeled as a single 'multi' class? I'll clarify this next.

#### Finalizing segmentation plan

For ground truth clarity, I'll recommend annotating multi-character clusters separately, with a "cluster: true" label and option to assign individual signs within, using instance segmentation with polygons. I'll propose a two-level dataset: individual sign boxes and cluster polygons. For dataset size, aiming for 50k-100k segments seems ideal to capture variance, with stratified sampling — 70-80% for training. Targeting at least 200k instances for broad coverage. If the budget's tight, 80k-100k segments should suffice. I’ll also look up literature to finalize segmentation sizes.

#### Addressing dataset structure

For clear evaluation, it’s best to annotate multi-character clusters separately, adding a “cluster: true” attribute. Individual sign labels within clusters can also use instance segmentation with polygons. I’ll recommend a two-level dataset: 1) individual sign bounding boxes; 2) cluster region polygons. To ensure proper sampling, a goal of 50,000–100,000 signs, well distributed, with at least 100 instances for rare signs, seems ideal. For validation, I’ll suggest stratified sampling — 70-80% for training, 10-15% for validation and testing, aiming for around 200,000 instance masks.

#### Finalizing dataset size recommendations

For a baseline challenge, I’d suggest aiming for at least 200k sign masks, though if the annotation budget is tight, 80-100k segments should be sufficient. With 10,000 images and an average of 20 signs per photo, that’ll give us 200k segments. A smaller dataset of 50k could also work. I’ll recommend optional multi-sign segments, but they should include an attribute. The dataset spec should reference bounding polygons for signs. I'll prepare to cite references such as COCO and PASCAL VOC datasets for justification.

### (1) How many sign instances should you annotate?

| Reference set (for comparison) | # classes | # images | # instance masks | ~instances / class |
| --- | --- | --- | --- | --- |
| PASCAL-VOC 2012 (seg.) | 20 | ≈7 k | ≈20 k | ≈1 k ([Dataset Ninja](https://datasetninja.com/pascal-voc-2012?utm_source=chatgpt.com)) |
| MS-COCO 2017 (seg.) | 80 | ≈164 k | ≈1.8 M | ≈22 k ([Dataset Ninja](https://datasetninja.com/coco-2017?utm_source=chatgpt.com)) |

Deep‑learning segmentation models begin to generalise once they see hundreds of masks per class and keep improving up to a few thousand. With 200 cuneiform signs, a realistic, competition‑ready target is:

≈ 300–1 000 masks / sign  →  60 000 – 200 000 total masks

*If you can afford a single full pass over all 10 000 photos (average 20–25 signs per face) you will land naturally in the upper half of that range, giving participants a dataset on par with modern benchmarks.*

### (2) Handling overlapped or "ligatured" signs

Because ancient scribes often wrote signs against or on top of each other, copy modern practice from COCO:

- **Annotate every sign you can separate** with its own polygon & category_id.

When two or more signs are truly inseparable, draw **one tight polygon around the whole cluste**r** and mark

`"iscrowd": 1        /* or  "cluster": true  */`

- *Evaluation scripts then ignore that region for per‑instance scoring while still letting models learn its shape.*

Optionally add a `"components": ["A", "KU"]` array for reference; it will help later corpus studies but is unnecessary for the challenge metric.

### (3) Proposed dataset split

| Split | Images | Masks | Purpose |
| --- | --- | --- | --- |
| Train | 7,000 | 100 k - 150 k | model fitting |
| Dev / public-leaderboard | 1,500 | 15 k - 20 k | parameter tuning; visible scores |
| Private test | 1,500 | 15 k - 20 k | final ranking; hidden labels |

Keep the original frequency imbalance of signs; do not artificially balance the train set—realistic skew is part of the task.

For very rare signs (≤30 masks), either merge them into an “other” class or label them but exclude from scoring.

### (4) Annotation conventions (JSON, COCO-style)

```JSON
{
  "images": [
    { "id": 42, "file_name": "VAT_123.tif", "width": 2048, "height": 1536 }
  ],
  "annotations": [
    {
      "id": 1017,
      "image_id": 42,
      "category_id": 73,          // sign: DU
      "segmentation": [ [...polygon...] ],
      "bbox": [x,y,w,h],
      "iscrowd": 0
    },
    {
      "id": 1018,
      "image_id": 42,
      "category_id": 999,         // cluster / ignored
      "segmentation": [ [...polygon...] ],
      "bbox": [x,y,w,h],
      "iscrowd": 1,
      "components": ["GU", "ŠE"]  // optional metadata
    }
  ],
  "categories": [
    { "id": 73,  "name": "DU" },
    …,
    { "id": 999, "name": "cluster" }
  ]
}
```

*Participants train ordinary instance‑segmentation models (Mask R‑CNN, YOLOv8‑seg, etc.); evaluation computes mAP over* `iscrowd = 0` *masks.*

### (5) Quality-of-life tips

1. **Active‑learning loop** – label 10 % of photos, train a quick model, use its heat‑maps to suggest the next uncertain regions; you halve annotation cost.

2. **Tile large photos** to 1024 × 1024 crops; it speeds annotators and model throughput.

3. **Version control** the annotation set (e.g., via DVC) so test data never leaks.

4. **Publish a validation script** that:
   1. checks mask overlap $\leqε$
   2. verifies every `category_id` is in `categories`,
   3. flags missing/extra images.

### Botton line

- **Aim for 60 k – 150 k well‑drawn sign masks** to give each of the 200 categories several hundred examples.
- **Mark inseparable clusters with an `iscrowd` (or `cluster`) flag** so they don’t handicap scoring but still enrich the visual grammar your competitors can learn.
- **Follow a COCO‑style JSON schema**; the community tooling, baselines, and evaluation code will work out of the box.

With those choices you strike the right balance between scholarly fidelity and a tractable computer‑vision benchmark.
