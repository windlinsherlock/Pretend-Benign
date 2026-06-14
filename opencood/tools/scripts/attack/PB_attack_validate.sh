#!/bin/bash

# 1. get all arguments
PY_ARGS=$@

# 2. extract the path immediately following --hypes_yaml from PY_ARGS
HYPES_YAML=$(echo "$PY_ARGS" | grep -oP '(?<=--hypes_yaml\s)\S+')

# 3. extract the part of the path after hypes_yaml/
RELATIVE_PATH=${HYPES_YAML#*hypes_yaml/}

# 4. generate a timestamp (format: YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# 5. build the log path: logs/ + path without .yaml + _timestamp.log
LOG_FILE="../../logs/logs/${RELATIVE_PATH%.yaml}_PB_attack_validate_${TIMESTAMP}.log"

# 6. ensure the target directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# 7. execute the command
echo "正在使用单卡推理..."
echo "日志将输出到: $LOG_FILE"

nohup python attack/PB_attack_validate.py ${PY_ARGS} > "$LOG_FILE" 2>&1 &