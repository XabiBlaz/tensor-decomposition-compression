# Layer relevance diagnostic

The analysis view can show a bar plot of **mean |weight × loss gradient| per parameter** for Linear and convolution layers. It uses a small set of *labeled* calibration batches and the task loss: cross-entropy for classification and segmentation, and masked next-token negative log likelihood for causal language models. The companion gradient RMS metric is useful when inspecting scale, but the weight–gradient score is the primary plot value. The plot should label both the raw value and a 0–1 value relative to the largest displayed layer.

This is a diagnostic inspired by first-order connection sensitivity in [SNIP (Lee et al., ICLR 2019)](https://arxiv.org/abs/1810.02340) and Taylor importance in [Molchanov et al. (CVPR 2019)](https://arxiv.org/abs/1906.10771). Our implementation takes the mean absolute weight–gradient product over each layer's weight elements, then averages batch scores weighted by valid labeled images, pixels, or next-token targets. This is **not** the exact loss change from deleting or compressing a layer. Coupled weights, nonlinear effects, and recovery training can change outcomes. Validate each actual compressed candidate against the requested quality metric.

[Centered kernel alignment, or CKA (Kornblith et al., ICML 2019)](https://proceedings.mlr.press/v97/kornblith19a.html) measures similarity between representations. It can answer a different question, such as whether two layers produce similar representations on the same examples. High CKA alone does not establish that a layer can be compressed safely, so it is not used as a compression score here.

```python
from tn_compression.relevance import analyze_relevance

result = analyze_relevance(
    model, calibration_batches, task="classification", max_batches=4, max_layers=64
)
# result["layers"] is in model depth order, ready for a bar plot.
```

For vision, each batch must be `(images, targets)`. For language, each batch is a tokenized mapping with `input_ids`, optional `attention_mask`, and optional `labels`; positions with label `-100` and padding are excluded. The function does not update weights or `.grad`, restores original training modes and `requires_grad` flags, and accepts iterable batches without retaining them. It limits work to four batches and 64 layers by default, spreading selected layers across model depth when necessary. It returns JSON-compatible data:

```json
{
  "task": "classification",
  "metric": "mean_abs_weight_grad",
  "batches": 4,
  "units": 128,
  "unit_label": "labeled images",
  "layers_available": 120,
  "layers_analyzed": 64,
  "layers": [
    {
      "layer_path": "encoder.layer1.0.conv1",
      "module_type": "Conv2d",
      "parameters": 36864,
      "gradient_rms": 0.0001,
      "taylor_saliency": 0.000002,
      "observed_batches": 4,
      "relative_relevance": 0.3
    }
  ]
}
```

Interpret larger bars as more sensitivity on the supplied labeled batches, **not** as a cross-model ranking or a guaranteed compression decision. The exact values depend on data and loss. A randomly initialized model or synthetic labels can verify the workflow but cannot provide a credible claim about deployment quality.
