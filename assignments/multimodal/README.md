# Multimodal Classification - Assignment 1

This folder contains the GitHub Pages structure for the multimodal branch of Assignment 1.

## Directory Structure

- index.html
- dataset-eda/index.html
- zero-shot/index.html
- few-shot/
  - index.html
  - linear-probe/index.html
  - adapter/index.html
  - [other-method]/index.html
- results/index.html
- error-analysis/index.html
- ablation-interpretability/index.html
- figures/ (per-page figure storage)
  - overview/
  - dataset-eda/
  - zero-shot/
  - few-shot/
    - linear-probe/
    - adapter/
  - results/
  - error-analysis/
  - ablation-interpretability/

## Page Overview (what to fill)

- multimodal/index.html - hub: links (dataset/code/report/video), approaches summary, key findings, navigation cards, comparison preview.
- dataset-eda/index.html - dataset motivation, stats, splits, class/text/image distributions, sample pairs, EDA observations.
- zero-shot/index.html - motivation, pipeline, prompts, implementation notes, results preview.
- few-shot/index.html - definition, shared setup, list of approaches, evaluation protocol, comparison preview.
- few-shot/[method]/index.html - method overview, pipeline diagram, training details, results placeholders (8/16/32/64-shot), qualitative notes.
- results/index.html - main table, learning curves, per-class breakdown, confusion matrices, efficiency charts, findings, significance.
- error-analysis/index.html - failure cases, confusion patterns, class-wise weaknesses, modality-specific errors, lessons.
- ablation-interpretability/index.html - component ablations, prompt/feature/training ablations, interpretability visuals, calibration/confidence plots.

## Figures
- Store all plots/figures under `figures/` in the matching subfolder for each page.
- Example filenames: `figures/dataset-eda/class-distribution.png`, `figures/results/confusion-matrix.png`, `figures/few-shot/adapter/pipeline.png`.
