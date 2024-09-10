#!/bin/bash

# 设置默认安装目录
INSTALL_DIR="/home/hnty/wrc/dangkang"

# 创建 systemd 服务文件
echo "Creating systemd service file..."
SERVICE_FILE="/etc/systemd/system/ragflow.service"

cat <<EOF | sudo tee $SERVICE_FILE
[Unit]
Description=Ragflow Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
ExecStart=/bin/bash -c "conda activate ragflow && /bin/bash $INSTALL_DIR/ragflow-start.sh"
Restart=always
Environment="WS=2" "API_PORT=8000" "API_HOST=0.0.0.0" "API_TIMEOUT=30" "API_MAX_CONN=100" "API_MAX_RETRY=3" "API_RETRY_INTERVAL=10" "API_RETRY_TIMEOUT=5" "API_RETRY_MAX_CONN=100" "API_RETRY_MAX_RETRY=3" "API_RETRY_RETRY_INTERVAL=10" "API_RETRY_RETRY_TIMEOUT=5"

[Install]
WantedBy=multi-user.target
EOF

# 重新加载 systemd 配置
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# 启用并启动服务
echo "Enabling and starting Ragflow service..."
sudo systemctl enable ragflow.service
sudo systemctl start ragflow.service

# 检查服务状态
echo "Checking service status..."
sudo systemctl status ragflow.service