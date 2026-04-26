# Feature visualization for BGHNetV4Trainer and MedNeXtTrainerV2

This project now includes a lightweight post-training feature visualization tool:

`tools/extract_feature_visualization.py`

## Goal

Use latent feature embeddings to support interpretability claims in the paper, by comparing the feature distributions learned by:

- `BGHNetV4Trainer`
- `MedNeXtTrainerV2`

The script extracts a middle-level feature map, reduces it to 2D with t-SNE or UMAP, and saves both the embedding and a scatter plot.

## Recommended feature layers

Start with one of these layer filters:

- `bottleneck`
- `encoder`
- `decoder`

If you want a more specific block, inspect `named_modules()` and use a substring that matches the target module name.

## Example commands

```bash
python tools/extract_feature_visualization.py \
  --trainer BGHNetV4Trainer \
  --fold 0 \
  --task Task530_EsoTJ_30pct \
  --method tsne \
  --feature-layer bottleneck \
  --num-cases 12 \
  --outdir feature_vis_output
```

```bash
python tools/extract_feature_visualization.py \
  --trainer MedNeXtTrainerV2 \
  --fold 0 \
  --task Task530_EsoTJ_30pct \
  --method umap \
  --feature-layer bottleneck \
  --num-cases 12 \
  --outdir feature_vis_output
```

## Output files

For each run, the script writes:

- `*.npz` — extracted feature matrix, 2D embedding, case ids, labels
- `*.png` — scatter plot for the paper
- `*.json` — metadata about the run

## Notes

- UMAP requires `umap-learn`.
- The current implementation uses a forward hook and a layer-name substring match, so it is minimally invasive.
- If you need a more precise layer selection, inspect the model's module names and set `--feature-layer` accordingly.
