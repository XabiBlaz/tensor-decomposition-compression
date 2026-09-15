# Quantization and serving status

The CPU rounding reference in `tn_compression.quantization` is symmetric grouped
round-to-nearest with floating-point weights. It measures quality damage and
does **not** produce packed INT4 storage or accelerated inference.

`examples/quantize_gptq.py` adapts the maintained
[llm-compressor 0.5.1 W4A16 recipe](https://github.com/vllm-project/llm-compressor/blob/0.5.1/examples/quantization_w4a16/llama3_example.py).
It pins the recipe version, uses only the configured calibration partition,
preserves `lm_head`, and saves a compressed-tensors checkpoint plus provenance.
Create a separate environment with `requirements/gptq.txt`; do not install it
over the tested core environment. The integration remains **unverified on GPU**.

The selected deployment target is vLLM 0.8.5, V0, FP16, single GPU, no prefix
caching. These are reproducibility pins, not claims about the latest versions.
The host has RTX 2080 Ti (SM 7.5) and driver 460.106.00. The selected
[compressed-tensors W4A16 path](https://docs.vllm.ai/en/v0.8.5/features/quantization/int4.html)
requires newer GPU capability; CUDA 12.x also needs a newer driver according to
[NVIDIA's compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
Some [other GPTQ kernels support Turing](https://docs.vllm.ai/en/v0.8.5/features/quantization/supported_hardware.html),
so this is a blocker for this chosen environment/format, not every GPTQ recipe.
Changing GPU drivers on this shared machine is outside this repository task.

## Commands for a compatible host

```sh
python examples/quantize_gptq.py --config examples/configs/qwen-pilot.yaml --output runs/qwen-gptq
docker build -f containers/serving.Dockerfile -t tn-serving:0.8.5 .
docker run --rm --gpus 'device=0' -p 127.0.0.1:8000:8000 --ipc=host \
  -v "$PWD/runs/qwen-gptq:/model:ro" tn-serving:0.8.5 \
  --model /model --served-model-name qwen-pilot --dtype half \
  --max-model-len 512 --gpu-memory-utilization 0.8 --enforce-eager
python -m tn_compression.serving --config examples/configs/serving-pilot.json --output runs/serving.json
```

For dense serving, use the pinned original Hugging Face model and pass both
`--revision` and `--tokenizer-revision` from `qwen-pilot.yaml`. Mount the HF cache.
Run the same workloads and precision for dense and compressed artifacts. Expand
the small example to recorded prompt/output lengths and concurrency levels.

The client records each streamed request, actual token counts from final usage,
failure outcomes, p50/p95, aggregate token throughput and exact timing definitions.
It measures client-observed content events; multi-token chunks prevent claiming
exact individual-token latency. Server hardware/version, weight/cache memory,
loading time and teacher-forced parity must be recorded separately. No live
server was available here, so there are no serving performance results.

The Dockerfiles have pinned base tags and dependencies but have not been built
and validated in this environment. Record resolved image digests after a build;
do not call them fully reproducible validated containers yet. Completion of
Milestone 4 still requires GPTQ save/reload, quality, backend parity, and measured
dense/quantized serving on a compatible host. Later CPU research can proceed.
