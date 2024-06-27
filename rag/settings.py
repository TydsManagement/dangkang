#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# 导入操作系统模块，用于路径拼接和环境变量获取
import os

# 导入工具模块，用于获取基础配置和解密数据库配置
from api.utils import get_base_config, decrypt_database_config
# 导入文件工具模块，用于获取项目基础目录
from api.utils.file_utils import get_project_base_directory
# 导入日志工具模块，用于日志记录
from api.utils.log_utils import LoggerFactory, getLogger

# 定义常量：RAG配置文件路径，位于项目基础目录的conf子目录下
# Server
RAG_CONF_PATH = os.path.join(get_project_base_directory(), "conf")
# 定义常量：子进程标准输出日志文件名
SUBPROCESS_STD_LOG_NAME = "std.log"

# 通过get_base_config获取ES（Elasticsearch）的基础配置，{}
ES = get_base_config("es", {})
# 通过decrypt_database_config解密并获取MinIO的数据库配置
MINIO = decrypt_database_config(name="minio")
# 尝试解密并获取Redis的数据库配置，如果失败则使用空字典{}
try:
    REDIS = decrypt_database_config(name="redis")
except Exception as e:
    REDIS = {}
    pass
# 定义常量：文档的最大大小，128MB
DOC_MAXIMUM_SIZE = 128 * 1024 * 1024

# 设置日志工厂的输出目录为项目基础目录的logs子目录下，并设定日志级别为30（WARNING级别）
# Logger
LoggerFactory.set_directory(
    os.path.join(
        get_project_base_directory(),
        "logs",
        "rag"))
LoggerFactory.LEVEL = 30

# 获取并定义不同模块的日志记录器，如ES、MinIO、Cron等
es_logger = getLogger("es")
minio_logger = getLogger("minio")
cron_logger = getLogger("cron_logger")
# 设置cron_logger的日志级别为20（INFO级别）
cron_logger.setLevel(20)
chunk_logger = getLogger("chunk_logger")
database_logger = getLogger("database")

# 定义常量：服务队列名称
SVR_QUEUE_NAME = "rag_flow_svr_queue"
# 定义常量：服务队列消息保留时间，1小时
SVR_QUEUE_RETENTION = 60*60
# 定义常量：服务队列最大长度，1024条消息
SVR_QUEUE_MAX_LEN = 1024
# 定义常量：服务消费者名称
SVR_CONSUMER_NAME = "rag_flow_svr_consumer"
# 定义常量：服务消费者组名称
SVR_CONSUMER_GROUP_NAME = "rag_flow_svr_consumer_group"
