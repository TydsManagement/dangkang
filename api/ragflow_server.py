import logging
import os
import signal
import sys
import time
import traceback
import urllib3
from concurrent.futures import ThreadPoolExecutor
from werkzeug.serving import run_simple

from api import utils
from api.apps import app
from api.db.db_models import init_database_tables as init_web_db
from api.db.init_data import init_web_data
from api.db.runtime_config import RuntimeConfig
from api.db.services.document_service import DocumentService
from api.settings import (
    HOST, HTTP_PORT, access_logger, database_logger, stat_logger,
)
from api.versions import get_versions

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def update_progress():
    """
    不断循环以更新进度的函数。

    该函数旨在定期调用DocumentService.update_progress()方法来更新某些操作的进度。
    它通过捕获可能发生的异常并记录错误来确保程序的健壮性。

    注意：此函数会无限循环运行，直到外部干预停止循环或发生无法恢复的错误。
    """
    while True:  # 无限循环，以确保进度更新持续进行
        time.sleep(1)  # 每秒检查一次是否需要更新进度
        try:
            DocumentService.update_progress()  # 尝试更新文档处理进度
        except Exception as e:  # 捕获所有可能的异常，以避免进程崩溃
            stat_logger.error("update_progress exception:" + str(e))  # 记录异常信息，以便于问题追踪和诊断


if __name__ == '__main__':
    print("""
████████╗ ██████╗ ██╗   ██╗ ██████╗ ██╗   ██╗     █████╗ ██╗
╚══██╔══╝██╔═══██╗╚██╗ ██╔╝██╔═══██╗██║   ██║    ██╔══██╗██║
   ██║   ██║   ██║ ╚████╔╝ ██║   ██║██║   ██║    ███████║██║
   ██║   ██║   ██║  ╚██╔╝  ██║   ██║██║   ██║    ██╔══██║██║
   ██║   ╚██████╔╝   ██║   ╚██████╔╝╚██████╔╝    ██║  ██║██║
   ╚═╝    ╚═════╝    ╚═╝    ╚═════╝  ╚═════╝     ╚═╝  ╚═╝╚═╝
                                                            
""", flush=True)
    stat_logger.info(
        f'project base: {utils.file_utils.get_project_base_directory()}'
    )

    # init db
    init_web_db()
    init_web_data()
    # init runtime config
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default=False, help="rag flow version", action='store_true')
    parser.add_argument('--debug', default=False, help="debug mode", action='store_true')
    args = parser.parse_args()
    if args.version:
        print(get_versions())
        sys.exit(0)

    RuntimeConfig.DEBUG = args.debug
    if RuntimeConfig.DEBUG:
        stat_logger.info("run on debug mode")

    RuntimeConfig.init_env()
    RuntimeConfig.init_config(JOB_SERVER_HOST=HOST, HTTP_PORT=HTTP_PORT)

    peewee_logger = logging.getLogger('peewee')
    peewee_logger.propagate = False
    # rag_arch.common.log.ROpenHandler
    peewee_logger.addHandler(database_logger.handlers[0])
    peewee_logger.setLevel(database_logger.level)

    thr = ThreadPoolExecutor(max_workers=1)
    thr.submit(update_progress)

    # start http server
    try:
        stat_logger.info("RAG Flow http server start...")
        werkzeug_logger = logging.getLogger("werkzeug")
        for h in access_logger.handlers:
            werkzeug_logger.addHandler(h)
        run_simple(hostname=HOST, port=HTTP_PORT, application=app, threaded=True, use_reloader=RuntimeConfig.DEBUG, use_debugger=RuntimeConfig.DEBUG)
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGKILL)
