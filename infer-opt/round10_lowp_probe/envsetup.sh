# sourced by batch scripts
cd /root/lightly-train/infer-opt/round10_lowp_probe
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export LIGHTLY_TRAIN_EVENTS_DISABLED=1
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
export QUACK_CACHE_DIR=/root/autodl-tmp/infer-opt-tmp/round10_lowp_probe/quack_cache
export TORCH_CUDA_ARCH_LIST=12.0
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/infer-opt-tmp/round10_lowp_probe/build/torch_ext
mkdir -p $TORCH_EXTENSIONS_DIR $QUACK_CACHE_DIR
