from flask_login import login_required

from api.db.services.knowledgebase_service import KnowledgebaseService
from api.utils.api_utils import get_json_result
from api.versions import get_rag_version
from rag.settings import SVR_QUEUE_NAME
from rag.utils.es_conn import ELASTICSEARCH
from rag.utils.minio_conn import MINIO
from timeit import default_timer as timer

from rag.utils.redis_conn import REDIS_CONN


@manager.route('/version', methods=['GET'])
@login_required
def version():
    return get_json_result(data=get_rag_version())


@manager.route('/status', methods=['GET'])
@login_required
def status():
    res = {}
    st = timer()
    try:
        res["es"] = ELASTICSEARCH.health()
        res["es"]["elapsed"] = "{:.1f}".format((timer() - st)*1000.)
    except Exception as e:
        res["es"] = {"status": "red", "elapsed": "{:.1f}".format((timer() - st)*1000.), "error": str(e)}

    st = timer()
    try:
        MINIO.health()
        res["minio"] = {"status": "green", "elapsed": "{:.1f}".format((timer() - st)*1000.)}
    except Exception as e:
        res["minio"] = {"status": "red", "elapsed": "{:.1f}".format((timer() - st)*1000.), "error": str(e)}

    st = timer()
    try:
        KnowledgebaseService.get_by_id("x")
        res["mysql"] = {"status": "green", "elapsed": "{:.1f}".format((timer() - st)*1000.)}
    except Exception as e:
        res["mysql"] = {"status": "red", "elapsed": "{:.1f}".format((timer() - st)*1000.), "error": str(e)}

    st = timer()
    try:
        if not REDIS_CONN.health():
            raise Exception("Lost connection!")
        res["redis"] = {"status": "green", "elapsed": "{:.1f}".format((timer() - st)*1000.)}
    except Exception as e:
        res["redis"] = {"status": "red", "elapsed": "{:.1f}".format((timer() - st)*1000.), "error": str(e)}

    return get_json_result(data=res)
