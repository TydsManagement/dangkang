#!/bin/bash

# 设置默认安装目录
INSTALL_DIR="/opt/dangkang"

# 设置版本号配置文件路径
VERSION_FILE="$INSTALL_DIR/ragflow_version.conf"

# 创建版本号配置文件（如果不存在）
if [ ! -f "$VERSION_FILE" ]; then
  echo "RAGFLOW_VERSION=1.0.1" | sudo tee $VERSION_FILE
fi

# 加载版本号
source $VERSION_FILE

# 创建 systemd 服务文件（如果不存在）
SERVICE_FILE="/etc/systemd/system/ragflow_release.service"

if [ ! -f "$SERVICE_FILE" ]; then
  echo "Creating systemd service file..."
  cat <<EOF | sudo tee $SERVICE_FILE
[Unit]
Description=Ragflow Service
After=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=$INSTALL_DIR
ExecStart=/bin/bash -c "source /home/hnty/anaconda3/bin/activate ragflow && /bin/bash $INSTALL_DIR/ragflow-start.sh"
Restart=always
Environment="WS=2" "API_PORT=8000" "API_HOST=0.0.0.0" "API_TIMEOUT=30" "API_MAX_CONN=100" "API_MAX_RETRY=3" "API_RETRY_INTERVAL=10" "API_RETRY_TIMEOUT=5" "API_RETRY_MAX_CONN=100" "API_RETRY_MAX_RETRY=3" "API_RETRY_RETRY_INTERVAL=10" "API_RETRY_RETRY_TIMEOUT=5"
Environment="STACK_VERSION=8.11.3" "ES_PORT=1200" "ELASTIC_PASSWORD=infini_rag_flow"
Environment="KIBANA_PORT=6601" "KIBANA_USER=rag_flow" "KIBANA_PASSWORD=infini_rag_flow"
Environment="MEM_LIMIT=8073741824" "MYSQL_PASSWORD=infini_rag_flow" "MYSQL_PORT=5455"
Environment="MINIO_CONSOLE_PORT=9001" "MINIO_PORT=9000" "MINIO_USER=rag_flow" "MINIO_PASSWORD=infini_rag_flow"
Environment="REDIS_PORT=6379" "REDIS_PASSWORD=infini_rag_flow" "SVR_HTTP_PORT=9527"
Environment="RAGFLOW_VERSION=$RAGFLOW_VERSION" "TIMEZONE='Asia/Shanghai'"

[Install]
WantedBy=multi-user.target
EOF
else
  echo "Service file already exists. Skipping creation."
fi

# 重新加载 systemd 配置
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# 启用并启动服务
echo "Enabling and starting Ragflow service..."
sudo systemctl enable ragflow.service
sudo systemctl restart ragflow.service

# 检查服务状态
echo "Checking service status..."
sudo systemctl status ragflow.service