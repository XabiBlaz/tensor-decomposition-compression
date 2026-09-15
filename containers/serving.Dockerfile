FROM vllm/vllm-openai:v0.8.5
# V0 supports the chosen historical stack; record this engine choice in results.
ENV VLLM_USE_V1=0
