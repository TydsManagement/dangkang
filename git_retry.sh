#!/bin/bash

# 设置最大重试次数
MAX_RETRIES=10
# 当前重试次数
RETRY_COUNT=0

# 检查传入的参数，确定是执行 push 还是 pull
ACTION=$1

# 执行指定的操作，直到成功或达到最大重试次数
while true; do
    if [ "$ACTION" == "push" ]; then
        git push
    elif [ "$ACTION" == "pull" ]; then
        git pull
    else
        echo "Invalid action. Use 'push' or 'pull'."
        exit 1
    fi

    if [ $? -eq 0 ]; then
        echo "$ACTION succeeded!"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT+1))
        echo "$ACTION failed. Retry count: $RETRY_COUNT"
        if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
            echo "Reached max retries. Exiting."
            exit 1
        fi
        # 等待几秒钟后再尝试重试
        sleep 10
    fi
done