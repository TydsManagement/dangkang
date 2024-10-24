# 导入必要的库，用于环境变量的管理和加载
import os
import dotenv
import typing
# 从api.utils.file_utils模块导入获取项目基础目录的函数
from api.utils.file_utils import get_project_base_directory

# 加载.env文件中的环境变量，并返回它们作为一个映射
def get_versions() -> typing.Mapping[str, typing.Any]:
    """
    加载并返回.env文件中的环境变量。
    
    返回:
        typing.Mapping[str, typing.Any]: 包含环境变量名称和值的映射。
    """
    # 加载.env文件中的环境变量
    dotenv.load_dotenv(dotenv.find_dotenv())
    # 返回加载的环境变量
    return dotenv.dotenv_values()


# 获取RAGFLOW_VERSION环境变量的值，如果未设置，则返回"dev"
def get_rag_version() -> typing.Optional[str]:
    """
    获取RAGFLOW_VERSION环境变量的值。
    
    如果环境变量未设置，则默认返回"dev"。
    
    返回:
        typing.Optional[str]: RAGFLOW_VERSION的值，如果未设置则为"dev"。
    """
    # 从系统环境变量中获取 RAGFLOW_VERSION 的值，如果不存在则使用 "dev" 作为默认值
    rag_version = os.getenv("RAGFLOW_VERSION", "dev")
    print(f"RAGFLOW_VERSION is {rag_version}")
    return os.getenv("RAGFLOW_VERSION", "dev")
