#!/bin/bash

# 设置路径和日志文件
INSTALL_TMP_DIR="/tmp/DangkangAI_Install_bag"
RAGFLOW_INSTALL_DIR="/opt/ragflow"
LOGFILE="$INSTALL_TMP_DIR/install.log"

# 日志记录函数
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOGFILE"
}

# 错误处理函数
handle_error() {
    log "ERROR: $1"
    exit 1
}

# 解压主文件包
log "Unpacking the main installation package..."
mkdir -p "$INSTALL_TMP_DIR"
tar -xzvf DangkangAI_Install_bag.tar.gz -C /tmp/ || handle_error "Failed to extract the main installation package."

# 检查并安装 Docker
install_docker() {
    log "Starting Docker installation..."
    cd "$INSTALL_TMP_DIR/docker/rpm" || handle_error "Failed to change directory to docker/rpm"
    bash docker_rpm_install.sh || handle_error "Failed to install Docker"
    log "Docker installed successfully."
}

# 安装 Docker Compose
install_docker_compose() {
    log "Installing Docker Compose..."
    cd "$INSTALL_TMP_DIR/docker" || handle_error "Failed to change directory to docker"
    bash install_docker_compose.sh || handle_error "Failed to install Docker Compose"
    log "Docker Compose installed successfully."
}

# 导入 Docker 镜像
load_docker_images() {
    log "Loading Docker images..."
    cd "$INSTALL_TMP_DIR/docker/image" || handle_error "Failed to change directory to docker/image"
    bash load_image.sh || handle_error "Failed to load Docker images"
    log "Docker images loaded successfully."
}

# 启动 Docker 容器
start_docker_containers() {
    log "Starting Docker containers..."
    cd "$INSTALL_TMP_DIR/db" || handle_error "Failed to change directory to db"
    bash import_containers.sh || handle_error "Failed to start Docker containers"
    log "Docker containers started successfully."
}

import_container_data() {
    log "Importing container data..."

    # 确保 MySQL 数据文件在正确的位置
    MYSQL_SQL_FILE="/tmp/DangkangAI_Install_bag/db_cache/all-databases.sql"
    if [ ! -f "$MYSQL_SQL_FILE" ]; then
        handle_error "MySQL data file not found at $MYSQL_SQL_FILE"
    fi

    # 调用小脚本执行数据导入
    bash "$INSTALL_TMP_DIR/db_cache/data_import.sh" || handle_error "Failed to import container data"
    log "Container data imported successfully."
}
# 拷贝模型文件
copy_model_files() {
    log "Copying model files..."
    cp -r "$INSTALL_TMP_DIR/models/nltk_data" /root/ || handle_error "Failed to copy nltk_data"
    mkdir -p /root/.ragflow/ || handle_error "Failed to create /root/.ragflow/ directory"
    cp -r "$INSTALL_TMP_DIR/models/bge-*" /root/.ragflow/ || handle_error "Failed to copy bge models"
    log "Model files copied successfully."
}

# 安装 Miniconda
install_miniconda() {
    log "Installing Miniconda..."
    cd "$INSTALL_TMP_DIR/conda" || handle_error "Failed to change directory to conda"
    bash install_miniconda.sh || handle_error "Failed to install Miniconda"
    log "Miniconda installed successfully."
}

# 设置 Conda 环境
setup_conda_environment() {
    log "Setting up Conda environment..."
    mkdir -p /root/miniconda3/envs/py11 || handle_error "Failed to create Conda environment directory"
    tar -xzvf "$INSTALL_TMP_DIR/conda/py11.tar.gz" -C /root/miniconda3/envs/py11 || handle_error "Failed to extract py11 environment"
    source /root/miniconda3/bin/activate py11 || handle_error "Failed to activate py11 environment"
    log "Conda environment set up successfully."
}

# 安装并启动 Ollama
install_and_start_ollama() {
    log "Installing Ollama..."
    cd "$INSTALL_TMP_DIR/ollama" || handle_error "Failed to change directory to ollama"
    bash ollama_install.sh || handle_error "Failed to install Ollama"
    bash create_ollama_service.sh || handle_error "Failed to create Ollama service"
    ollama create llama3.1 -f Modelfile || handle_error "Failed to create Ollama model"
    log "Ollama installed and started successfully."
}

# 解压和启动 Ragflow
setup_and_start_ragflow() {
    log "Setting up Ragflow..."
    mkdir -p "$RAGFLOW_INSTALL_DIR" || handle_error "Failed to create Ragflow install directory"
    cd "$INSTALL_TMP_DIR/ragflow" || handle_error "Failed to change directory to ragflow"
    bash unpack_ragflow.sh "$RAGFLOW_INSTALL_DIR" || handle_error "Failed to unpack Ragflow"
    bash start_service.sh || handle_error "Failed to start Ragflow service"
    log "Ragflow setup and started successfully."
}

# 主安装函数
main_install() {
    log "Starting installation process..."

    install_docker
    install_docker_compose
    load_docker_images
    start_docker_containers
    import_container_data
    copy_model_files
    install_miniconda
    setup_conda_environment
    install_and_start_ollama
    setup_and_start_ragflow

    log "Installation completed successfully."
}

# 开始安装
main_install
