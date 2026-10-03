# T2 trainer 组件镜像（L2）：FROM 共享 base（C4 · docker/Dockerfile.base）+ 代码层秒级重建。
# 构建上下文 = stack 根（fork 单独目录无法构建——path dep 代价，决策见 doc/topics/t2_nano_deepspeed.md §依赖接线）：
#   docker build -f docker/Dockerfile.base -t nano-stack/base:torch-cu13-py312 .  # 仅首次/torch 变更（root 线域）
#   docker build -f nano-deepspeed/Dockerfile -t nano-stack/train:t2 .            # 代码改动 ~1 min
#   minikube -p nano-stack image load nano-stack/base:torch-cu13-py312 nano-stack/train:t2
FROM nano-stack/base:torch-cu13-py312

COPY nano-model /opt/nano-model
COPY nano-deepspeed /opt/nano-deepspeed
RUN pip3 install --break-system-packages --no-deps --no-build-isolation \
    /opt/nano-model /opt/nano-deepspeed

WORKDIR /workspace
# 2-pod rendezvous（Indexed Job）：
#   torchrun --nnodes=2 --node-rank=$JOB_COMPLETION_INDEX --nproc-per-node=1 \
#     --master-addr=nano-train-0.nano-train-hs --master-port=29500 train_nano_model.py ...
ENTRYPOINT ["torchrun"]
