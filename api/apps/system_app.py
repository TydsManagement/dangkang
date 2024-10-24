from flask_login import login_required

from api.db.services.knowledgebase_service import KnowledgebaseService
from api.utils.api_utils import get_json_result
from api.versions import get_rag_version
from rag.settings import SVR_QUEUE_NAME
from rag.utils.es_conn import ELASTICSEARCH
from rag.utils.minio_conn import MINIO
from timeit import default_timer as timer

from rag.utils.redis_conn import REDIS_CONN


# 定义一个路由，处理对'/version'的GET请求
# 此装饰器表明该函数将处理特定的HTTP请求
@manager.route('/version', methods=['GET'])
# 要求用户登录后才能访问该路由
# 这个装饰器确保了只有登录的用户才能访问版本信息
@login_required
def version():
    """
    返回当前系统的版本信息

    该函数通过调用get_json_result函数，将系统版本信息封装成JSON格式的数据并返回
    确保了数据的格式一致性和易于解析
    """
    return get_json_result(data=get_rag_version())



# 定义一个路由，处理对'/status'的GET请求，且要求用户登录后才能访问
@manager.route('/status', methods=['GET'])
@login_required
def status():
    """
    检查系统中关键服务的健康状态，并以JSON格式返回结果。
    包括Elasticsearch、Minio、MySQL和Redis的服务状态检查。
    """
    # 初始化一个空字典，用于存储各服务的健康状态信息
    res = {}

    # 记录当前时间，用于计算Elasticsearch健康检查的耗时
    st = timer()
    try:
        # 尝试获取Elasticsearch的健康状态，并记录耗时
        res["es"] = ELASTICSEARCH.health()
        res["es"]["elapsed"] = "{:.1f}".format((timer() - st)*1000.)
    except Exception as e:
        # 如果发生异常，记录错误信息和耗时
        res["es"] = {"status": "red", "elapsed": "{:.1f}".format((timer() - st)*1000.), "error": str(e)}

    # 重复上述过程，检查Minio的健康状态
    st = timer()
    try:
        MINIO.health()
        res["minio"] = {"status": "green", "elapsed": "{:.1f}".format((timer() - st)*1000.)}
    except Exception as e:
        res["minio"] = {"status": "red", "elapsed": "{:.1f}".format((timer() - st)*1000.), "error": str(e)}

    # 检查MySQL数据库的连接状态
    st = timer()
    try:
        KnowledgebaseService.get_by_id("x")
        res["mysql"] = {"status": "green", "elapsed": "{:.1f}".format((timer() - st)*1000.)}
    except Exception as e:
        res["mysql"] = {"status": "red", "elapsed": "{:.1f}".format((timer() - st)*1000.), "error": str(e)}

    # 检查Redis的健康状态
    st = timer()
    try:
        if not REDIS_CONN.health():
            raise Exception("Lost connection!")
        res["redis"] = {"status": "green", "elapsed": "{:.1f}".format((timer() - st)*1000.)}
    except Exception as e:
        res["redis"] = {"status": "red", "elapsed": "{:.1f}".format((timer() - st)*1000.), "error": str(e)}

    # 返回包含所有服务健康状态的JSON响应
    return get_json_result(data=res)
