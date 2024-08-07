#!/bin/bash

# 导出应用程序容器
function export_container() {
  local container_name=$1
  local image_name=$2
  local output_file=$3

  # 创建镜像
  echo "正在为 ${container_name} 创建镜像..."
  docker commit $container_name $image_name

  # 导出镜像为 tar 文件
  echo "正在保存 ${image_name} 镜像到 ${output_file}..."
  docker save -o $output_file $image_name
}

# 导出数据卷
function export_volume() {
  local volume_name=$1
  local output_file=$2

  echo "正在导出 ${volume_name} 数据卷到 ${output_file}..."
  docker run --rm -v $volume_name:/data -v $(pwd):/backup busybox tar cvf /backup/$output_file /data
}

# 导出 MySQL 数据
function export_mysql() {
  local container_name=$1
  local output_file=$2

  echo "正在导出 MySQL 数据库..."
  docker exec $container_name mysqldump -u <username> -p<password> --all-databases > $output_file
}

# 定义容器和卷名称
APP_CONTAINER="ragflow-server"
ES_CONTAINER="ragflow-es-01"
MYSQL_CONTAINER="ragflow-mysql"
MINIO_CONTAINER="ragflow-minio"
REDIS_CONTAINER="ragflow-redis"

# 定义导出文件名
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

# 导出应用程序和数据库镜像
export_container $APP_CONTAINER "ragflow_app_image" $APP_IMAGE_FILE
export_container $ES_CONTAINER "es_image" $ES_IMAGE_FILE
export_container $MYSQL_CONTAINER "mysql_image" $MYSQL_IMAGE_FILE
export_container $MINIO_CONTAINER "minio_image" $MINIO_IMAGE_FILE
export_container $REDIS_CONTAINER "redis_image" $REDIS_IMAGE_FILE

# 导出 MySQL 数据库
export_mysql $MYSQL_CONTAINER $MYSQL_BACKUP_FILE

# 导出数据卷
export_volume $ES_VOLUME $ES_DATA_FILE
export_volume $MINIO_VOLUME $MINIO_DATA_FILE
export_volume $REDIS_VOLUME $REDIS_DATA_FILE

# 打包所有导出的文件
echo "正在将所有文件打包为 ragflow_package.tar.gz..."
tar -czvf ragflow_package.tar.gz $APP_IMAGE_FILE $ES_IMAGE_FILE $MYSQL_IMAGE_FILE $MINIO_IMAGE_FILE $REDIS_IMAGE_FILE $MYSQL_BACKUP_FILE $ES_DATA_FILE $MINIO_DATA_FILE $REDIS_DATA_FILE

echo "导出完成！请将 ragflow_package.tar.gz 复制到目标机器以进行安装。"
