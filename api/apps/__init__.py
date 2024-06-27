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
import logging
import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from flask import Blueprint, Flask
from werkzeug.wrappers.request import Request
from flask_cors import CORS

from api.db import StatusEnum
from api.db.db_models import close_connection
from api.db.services import UserService
from api.utils import CustomJSONEncoder

from flask_session import Session
from flask_login import LoginManager
from api.settings import SECRET_KEY, stat_logger
from api.settings import API_VERSION, access_logger
from api.utils.api_utils import server_error_response
from itsdangerous.url_safe import URLSafeTimedSerializer as Serializer

__all__ = ['app']


logger = logging.getLogger('flask.app')
for h in access_logger.handlers:
    logger.addHandler(h)

Request.json = property(lambda self: self.get_json(force=True, silent=True))

app = Flask(__name__)
CORS(app, supports_credentials=True,max_age=2592000)
app.url_map.strict_slashes = False
app.json_encoder = CustomJSONEncoder
app.errorhandler(Exception)(server_error_response)


## convince for dev and debug
#app.config["LOGIN_DISABLED"] = True
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get("MAX_CONTENT_LENGTH", 128 * 1024 * 1024))

Session(app)
login_manager = LoginManager()
login_manager.init_app(app)



def search_pages_path(pages_dir):
    """
    在指定目录中搜索所有的_app.py和_api.py文件路径。

    参数:
    pages_dir: Pathlib.Path对象，表示要搜索的目录。

    返回:
    一个列表，包含所有找到的_app.py和_api.py文件的路径。
    """
    # 搜索目录中所有的_app.py文件，并过滤掉以.开头的隐藏文件
    app_path_list = [path for path in pages_dir.glob('*_app.py') if not path.name.startswith('.')]

    # 搜索目录中所有的_api.py文件，并过滤掉以.开头的隐藏文件
    api_path_list = [path for path in pages_dir.glob('*_api.py') if not path.name.startswith('.')]

    # 将_api.py文件路径列表合并到_app.py文件路径列表中
    app_path_list.extend(api_path_list)

    # 返回合并后的文件路径列表
    return app_path_list



def register_page(page_path):
    """
    注册一个页面路由。

    根据页面路径动态生成模块名和页面名，并根据路径中是否包含"_api"来决定URL的前缀。
    它通过读取页面文件生成模块，然后将该模块作为一个Blueprint注册到应用程序中。

    :param page_path: 页面文件的Path对象，用于获取模块名和页面名，以及注册Blueprint的路径。
    :return: 注册路由的URL前缀。
    """
    # 将page_path转换为字符串路径
    path = f'{page_path}'

    # 根据path中是否包含"_api"来决定page_name的处理方式
    page_name = page_path.stem.rstrip('_api') if "_api" in path else page_path.stem.rstrip('_app')
    # 构建模块名，基于页面路径中的"api"目录位置和page_name
    module_name = '.'.join(page_path.parts[page_path.parts.index('api'):-1] + (page_name,))

    # 从文件位置生成模块规范
    spec = spec_from_file_location(module_name, page_path)
    # 根据规范生成模块
    page = module_from_spec(spec)
    # 设置页面模块的app和manager属性
    page.app = app
    page.manager = Blueprint(page_name, module_name)
    # 将页面模块注册到系统模块中
    sys.modules[module_name] = page
    # 执行模块代码，初始化模块
    spec.loader.exec_module(page)
    # 如果页面模块中定义了page_name属性，则使用该属性值，否则使用默认的page_name
    page_name = getattr(page, 'page_name', page_name)
    # 根据path中是否包含"_api"来决定URL前缀的格式
    url_prefix = f'/api/{API_VERSION}/{page_name}' if "_api" in path else f'/{API_VERSION}/{page_name}'

    # 注册Blueprint，并设置URL前缀
    app.register_blueprint(page.manager, url_prefix=url_prefix)
    # 返回URL前缀
    return url_prefix


# 定义一个列表，包含项目中两个重要的页面目录
# 这两个目录分别是当前文件所在的目录和上级目录中的api/apps子目录
# 注释中提到的FIXME指示可能存在需要清理或优化的代码路径
pages_dir = [
    Path(__file__).parent,
    Path(__file__).parent.parent / 'api' / 'apps', # FIXME: ragflow/api/api/apps, can be remove?
]

client_urls_prefix = [
    register_page(path)
    for dir in pages_dir
    for path in search_pages_path(dir)
]


@login_manager.request_loader
def load_user(web_request):
    """
    登录管理器的请求加载器，用于在每个请求中尝试加载当前用户。

    @param web_request: 请求对象，从中获取认证信息。
    @return: 如果认证成功，返回用户对象；否则返回None。
    """
    # 初始化JWT序列化器，用于验证JWT令牌
    jwt = Serializer(secret_key=SECRET_KEY)
    # 从请求头中获取Authorization信息
    authorization = web_request.headers.get("Authorization")
    if authorization:
        try:
            # 解析Authorization信息中的JWT令牌
            access_token = str(jwt.loads(authorization))
            # 根据JWT令牌查询用户服务，验证令牌有效性
            user = UserService.query(access_token=access_token, status=StatusEnum.VALID.value)
            if user:
                # 如果用户存在且有效，返回用户对象
                return user[0]
            else:
                # 如果用户不存在或无效，返回None
                return None
        except Exception as e:
            # 记录JWT解析过程中的异常
            stat_logger.exception(e)
            # 异常情况下返回None
            return None
    else:
        # 如果请求中没有Authorization信息，返回None
        return None


# 定义一个在每次HTTP请求结束后调用的函数，用于关闭数据库连接
# 此函数作为装饰器应用于app的teardown_request方法，确保即使请求处理过程中发生异常，也能正确关闭数据库连接
@app.teardown_request
def _db_close(exc):
    # 调用close_connection函数来关闭数据库连接
    close_connection()
