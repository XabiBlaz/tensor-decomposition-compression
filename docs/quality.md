# Quality contracts

Validated analysis constrains an explicit task metric. Set `analysis.quality_metric`
in your YAML and pass `--max-quality-loss` to `tn-compress analyze`. The bound is
the maximum **absolute degradation from the original model**, evaluated on the
same validation batches. An improvement has a negative degradation.

```yaml
task: classification
analysis:
  quality_metric: accuracy
```

For this configuration, `--max-quality-loss 0.01` permits at most **one percentage
point** of top-1 accuracy loss, for example 0.90 to 0.89. It does not mean a one
percent relative drop. `accuracy` is an alias for the evaluator's `top1` metric;
reports record the canonical name `top1`.

| Task | Available metrics | Default | Better direction |
| --- | --- | --- | --- |
| Classification | `loss`, `top1` / `accuracy` | `loss` | Lower loss; higher accuracy |
| Segmentation | `loss`, `mean_iou`, `mean_dice` | `loss` | Lower loss; higher overlap |
| Causal language modeling | `nll`, `perplexity` | `nll` | Lower |
| Detection | `ap`, `ap50` | `ap` | Higher |

Accuracy, overlap and AP use fractions (0 to 1), so a bound of 0.01 means one
percentage point. Loss, NLL and perplexity use absolute native metric units. For
example, `nll` with a bound of 0.05 permits an increase from 2.0 to 2.05. Existing
configurations retain their previous default metrics and numerical bounds.

The report and compression plan both retain `constraints` containing
`quality_metric`, `quality_direction` (`higher` or `lower`), `quality_units`,
`max_quality_loss` and `target_size_bytes`. Candidate `quality_loss` is signed
degradation in those units. Individual metric deltas remain measured minus
baseline, so their signs differ from degradation for higher-is-better metrics.

The baseline must contain the selected finite metric. Missing, null, boolean or
nonfinite values stop analysis; a candidate with invalid intervention or cumulative
evidence is rejected. Nonfinite scalar diagnostics also invalidate an evaluation.
Bounds must be finite, target bytes positive, and maximum degradation nonnegative.
Every cumulative proposal is checked against the original baseline before it is
accepted, and analysis restores the supplied model after selection.

These are validation-data guarantees, not guarantees for unseen inputs or a
different runtime. Use representative calibration and validation samples, reserve
an independent final test set, and reevaluate after recovery, checkpoint reload
and deployment export. The current contract has one chosen quality metric;
simultaneous bounds on multiple metrics and statistical confidence bounds are not
implemented. Selection is greedy and may report an infeasible target even when a
different allocation could meet it.
