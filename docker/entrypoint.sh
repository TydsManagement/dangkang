#!/bin/bash

# 启动Nginx服务
/usr/sbin/nginx

# 设置LD_LIBRARY_PATH环境变量，以便找到必要的库文件
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/

# 定义Python命令的别名
PY=python3

# 确保工作线程数WS不为空且大于等于1，否则默认为1
if [[ -z "$WS" || $WS -lt 1 ]]; then
  WS=1
fi

# 定义任务执行器函数，无限循环执行Python脚本
function task_exe(){
    while [ 1 -eq 1 ];do
      $PY rag/svr/task_executor.py ;
    done
}

# 根据工作线程数启动相应数量的任务执行器进程
for ((i=0;i<WS;i++))
do
  task_exe  &
done

# 启动API服务器，无限循环执行Python脚本
echo "Starting API server..."
echo "WS: $WS"
echo "API_PORT: $API_PORT"
echo "API_HOST: $API_HOST"
echo "API_TIMEOUT: $API_TIMEOUT"
echo "API_MAX_CONN: $API_MAX_CONN"
echo "API_MAX_RETRY: $API_MAX_RETRY"
echo "API_RETRY_INTERVAL: $API_RETRY_INTERVAL"
echo "API_RETRY_TIMEOUT: $API_RETRY_TIMEOUT"
echo "API_RETRY_MAX_CONN: $API_RETRY_MAX_CONN"
echo "API_RETRY_MAX_RETRY: $API_RETRY_MAX_RETRY"
echo "API_RETRY_RETRY_INTERVAL: $API_RETRY_RETRY_INTERVAL"
echo "API_RETRY_RETRY_TIMEOUT: $API_RETRY_RETRY_TIMEOUT"
echo "API_RETRY_MAX_CONN: $API_RETRY_MAX_CONN"
while [ 1 -eq 1 ];do
    $PY api/ragflow_server.py
done

# 等待所有子进程结束
wait;
