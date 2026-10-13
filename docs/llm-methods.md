# LLM compression choices

There is no universal best method. Compare the saved artifact's task quality,
storage, memory and latency on the same workload and runtime. Token NLL measures
next-token prediction; instruction following and downstream capability need
separate evaluations. Choose on validation data and evaluate the frozen choice
on held-out data.

| Method | What changes | Deployment consequence | Repository status |
| --- | --- | --- | --- |
| Quantization: RTN, GPTQ, AWQ | Store weights at fewer bits; GPTQ/AWQ use calibration information | Native packed weights and compatible kernels can reduce memory and improve speed; shape/device dependent | W4A16 GPU pilot prepared; native execution awaiting GPU verification |
| Low-rank SVD / weighted SVD | Replace a matrix by two smaller factors | Standard dense operations, but two operations can be slower than one | Core implementations; custom UI recipe supports energy-based SVD |
| Structured MLP pruning | Physically remove paired gated-MLP channels | Smaller dense matrices; architecture changes must survive checkpoint reload | Supported Llama/Qwen gated MLPs; custom recipe and combination tests |
| Unstructured pruning: Wanda, SparseGPT | Remove individual weights while preserving dimensions | Zeros alone do not shrink dense storage or accelerate dense kernels | Not integrated into the native pilot |
| Semi-structured N:M pruning | Enforce a hardware-specific pattern, such as two retained weights per four | Requires supported sparse packing, shapes and kernels; do not infer speed from sparsity | Not integrated |
| Head, block or depth pruning | Remove architectural components | Cache, attention, configuration and checkpoint handling require model-specific support | Not a general supported route |
| Dimension slicing: SliceGPT | Rotate representations and remove rows/columns consistently | Dense smaller matrices; architecture-specific transformations | Research candidate, not integrated |
| Distillation / recovery training | Teach a smaller student or adapt the compressed model | Extra training/data cost; compare with dense given equal additional training budget | Core LoRA recovery exists; not included in the native W4 pilot |
| KV-cache quantization/compression | Reduce the inference cache rather than model weights | Useful for long context/concurrency; affects a different memory budget | Not included in this pilot |

For an m-by-n weight matrix, rank-r factorization stores r(m+n) numbers instead
of mn. It saves parameters only when r < mn/(m+n); scales, biases and serialized
metadata change achieved storage. SVD energy is the retained fraction of squared
singular values, not an accuracy guarantee. Activation-aware approximations can
preserve outputs better than weight-only energy selection; verify that advantage
at comparable achieved budgets. Tensor-train/Tucker representations require
tensorization and appropriate execution paths, so they are not interchangeable
with a standard Transformers checkpoint.

The custom UI applies structural pruning, then SVD, then packed int8 weight
storage. Its int8 modules dequantize for ordinary floating-point execution: they
are a portable storage path, not native INT8/INT4 kernels. The separate native
W4 pilot loads RTN/GPTQ/AWQ artifacts in vLLM and is currently CLI-only. Combining
methods can compound quality loss; validate the final reloaded model, not only
each intermediate operation.

Recommended progression: run the three-method native pilot first; add a matched
budget weighted-SVD/structured-MLP comparison after inspecting those results.
Integrate unstructured or N:M pruning only when there is an explicit supported
artifact and sparse inference route. Agentic selection can be considered after
this deterministic measurement contract works; it cannot substitute for evidence.

Primary references:

- [GPTQ paper](https://arxiv.org/abs/2210.17323) and [AWQ paper](https://arxiv.org/abs/2306.00978).
- [Wanda reference implementation](https://github.com/locuslab/wanda) uses weight magnitude and input activation norms.
- [SparseGPT reference implementation](https://github.com/IST-DASLab/sparsegpt) uses approximate second-order reconstruction.
- [SVD-LLM reference implementation](https://github.com/AIoT-MLSys-Lab/SVD-LLM) studies activation-aware low-rank compression.
- [SliceGPT reference implementation](https://github.com/microsoft/TransformerCompression) studies rotated dimension slicing.
- [PyTorch semi-structured sparsity tutorial](https://pytorch.org/tutorials/advanced/semi_structured_sparse.html) demonstrates a supported 2:4 execution route.
- [vLLM quantization compatibility](https://docs.vllm.ai/en/stable/features/quantization/) describes runtime hardware limits.
