#!/bin/bash

# 导入 Docker 镜像
function import_image() {
  local image_file=$1

  echo "正在加载 ${image_file} 镜像..."
  docker load -i $image_file
}

# 初始化数据卷
function initialize_volume() {
  local volume_name=$1
  local backup_file=$2

  echo "正在初始化 ${volume_name} 数据卷..."
  docker run --rm -v $volume_name:/data -v $(pwd):/backup busybox tar xvf /backup/$backup_file
}

# 初始化 MySQL 数据
function initialize_mysql() {
  local container_name=$1
  local backup_file=$2

  echo "正在初始化 MySQL 数据库..."
  docker exec -i $container_name mysql -u <username> -p<password> < $backup_file
}

# 定义导入文件名
APP_IMAGE_FILE="ragflow_app_image.tar"
ES_IMAGE_FILE="es_image.tar"
MYSQL_IMAGE_FILE="mysql_image.tar"
MINIO_IMAGE_FILE="minio_image.tar"
REDIS_IMAGE_FILE="redis_image.tar"

ES_VOLUME="es_data"
MINIO_VOLUME="minio_data"
REDIS_VOLUME="redis_data"

MYSQL_BACKUP_FILE="mysql_backup.sql"
ES_DATA_FILE="es_data.tar"
MINIO_DATA_FILE="minio_data.tar"
REDIS_DATA_FILE="redis_data.tar"

# 导入 Docker 镜像
import_image $APP_IMAGE_FILE
import_image $ES_IMAGE_FILE
import_image $MYSQL_IMAGE_FILE
import_image $MINIO_IMAGE_FILE
import_image $REDIS_IMAGE_FILE

# 创建并启动容器

# 启动 Redis 容器
docker run -d --name ragflow-redis -v redis_data:/data redis_image

# 启动 Elasticsearch 容器
docker run -d --name ragflow-es-01 -v es_data:/usr/share/elasticsearch/data es_image

# 启动 MySQL 容器
docker run -d --name ragflow-mysql -e MYSQL_ROOT_PASSWORD=<password> mysql_image

# 启动 MinIO 容器
docker run -d --name ragflow-minio -v minio_data:/data minio_image

# 初始化数据卷
initialize_volume $ES_VOLUME $ES_DATA_FILE
initialize_volume $MINIO_VOLUME $MINIO_DATA_FILE
initialize_volume $REDIS_VOLUME $REDIS_DATA_FILE

# 初始化 MySQL 数据库
initialize_mysql ragflow-mysql $MYSQL_BACKUP_FILE

# 启动应用程序容器，并链接到相应的服务
docker run -d --name ragflow-server -p 80:80 -p 443:443 -p 9380:9380 --link ragflow-es-01 --link ragflow-mysql --link ragflow-minio --link ragflow-redis ragflow_app_image

echo "安装完成！RAGFlow 服务已成功启动并准备就绪。"
