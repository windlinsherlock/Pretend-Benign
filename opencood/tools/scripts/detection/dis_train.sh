#!/bin/bash

# get the number of GPUs
NUM_PROC=$1

# get all arguments except the GPU count
PY_ARGS=${@:2}

# use a regular expression to extract the path immediately following --hypes_yaml from PY_ARGS
HYPES_YAML=$(echo "$PY_ARGS" | grep -oP '(?<=--hypes_yaml\s)\S+')

# 1. extract the part of the path after hypes_yaml/
RELATIVE_PATH=${HYPES_YAML#*hypes_yaml/}

# 2. generate a timestamp (format: YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# 3. build the timestamped log path
# example: logs/attack/opv2v/.../point_pillar_intermediate_mean_20260602_233000.log
LOG_FILE="../../logs/logs/${RELATIVE_PATH%.yaml}_dist_train_${TIMESTAMP}.log"

# 4. ensure the target directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# 5. execute the command
echo "正在使用 $NUM_PROC 张显卡进行训练..."
echo "配置文件: $HYPES_YAML"
echo "日志已输出至: $LOG_FILE"
echo "附加参数: $PY_ARGS"

# 6. execute with torchrun
nohup torchrun --nproc_per_node=$NUM_PROC detection/train.py ${PY_ARGS} > "$LOG_FILE" 2>&1 &