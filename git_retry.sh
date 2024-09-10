#!/bin/bash

# 设置最大重试次数
MAX_RETRIES=10
# 当前重试次数
RETRY_COUNT=0

# 执行 git push 操作，直到成功或达到最大重试次数
while true; do
    git push
    if [ $? -eq 0 ]; then
        echo "Push succeeded!"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT+1))
        echo "Push failed. Retry count: $RETRY_COUNT"
        if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
            echo "Reached max retries. Exiting."
            exit 1
        fi
        # 等待几秒钟后再尝试重试
        sleep 10
    fi
done
