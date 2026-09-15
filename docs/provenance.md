# Sources and attribution

The public starting point is commit `732e612`. The source inventory records the
four locally available repositories, their revisions, working-diff fingerprints
and selected source-file hashes. Working changes were inspected; source checkouts
and their remotes are left untouched. A fingerprint is provenance, not a claim
that every source file has been migrated.

Source contributors include Xabier Blazquez, Xabier Zurutuza, Ander Muniategui,
Alvaro Rodrigo and G. Fernandez (as identified by source metadata and Git history).
The user confirmed authorization to publish reusable code with attribution.
Dataset artifacts, private deployment configuration and credentials are excluded.

## Mathematical implementations

The original CP/Tucker code credits
[yuncheng97/Tensor-decomposition-pytorch](https://github.com/yuncheng97/Tensor-decomposition-pytorch)
and [Jean Kossaifi's TensorLy notebooks](https://github.com/JeanKossaifi/tensorly-notebooks).
The retained VBMF implementation credits
[Cas van den Bogaard's VBMF](https://github.com/CasvandenBogaard/VBMF) and the
Nakajima et al. variational matrix factorization work. Preserve upstream notices;
the repository license does not override third-party terms.

TensorLy performs CP and partial Tucker decomposition. The reconciled execution
code fixes bias, dtype/device, coefficient absorption, and spatial geometry.
CP3 means pointwise–spatial depthwise–pointwise; CP4 splits the spatial operation
into vertical and horizontal depthwise convolutions, for either rank-selection mode.
`cp2` is retained as an alias of `svd`: it is matrix SVD, not tensor CP or TT.

TT-SVD uses successive matrix SVDs following Oseledets' tensor-train construction.
TT-matrix factorization now interleaves output/input axes explicitly and contracts
cores directly. `PLAIN_TT` remains an explicitly dense-reconstruction compatibility
path. It is unsuitable as evidence of factorized inference acceleration.

## Migration ledger

| Source | Incorporated | Still pending |
| --- | --- | --- |
| Existing public project | API, demonstrations, CI, corrected decomposition kernels | Research workflows |
| tn_compression_module | DLF configuration contract; external Fisher-group plans | Complementary analysis utilities |
| 925013_qiavs | Bias/dtype fixes reconciled with numerical checks | Task, recovery and evaluation behavior |
| segmentation_models_pytorch_Lortek | Source review | Thin SMP adapter and losses |
| arodrigo-RAM_Usage | Source review | Isolated measurements and export |

The flawed full-spatial Tucker path in the QIAVS source assigns a spatially reduced
core without applying its spatial factors. It was not imported. Use validated
`partial_tucker`. Custom Conv2d forward methods, tied parameters and parent access
to child weights require adapters and are reported as unsupported.
