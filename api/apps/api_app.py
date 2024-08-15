import json
import os
import re
from datetime import datetime, timedelta
from flask import request, Response
from flask_login import login_required, current_user

from api.db import FileType, ParserType, FileSource
from api.db.db_models import APIToken, API4Conversation, Task, File
from api.db.services import duplicate_name
from api.db.services.api_service import APITokenService, API4ConversationService
from api.db.services.dialog_service import DialogService, chat
from api.db.services.document_service import DocumentService
from api.db.services.file2document_service import File2DocumentService
from api.db.services.file_service import FileService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.task_service import queue_tasks, TaskService
from api.db.services.user_service import UserTenantService
from api.settings import RetCode, retrievaler
from api.utils import get_uuid, current_timestamp, datetime_format
from api.utils.api_utils import server_error_response, get_data_error_result, get_json_result, validate_request
from itsdangerous import URLSafeTimedSerializer

from api.utils.file_utils import filename_type, thumbnail
from rag.utils.minio_conn import MINIO


def generate_confirmation_token(tenent_id):
    """
    生成确认令牌。

    该函数为特定的租户生成一个安全的、带时间限制的确认令牌。令牌用于确认用户操作，如邮箱确认、密码重置等。

    参数:
    - tenent_id: 租户标识符，用于确保令牌的唯一性和与特定租户的关联。

    返回值:
    - 一个字符串，表示生成的确认令牌。该令牌包含租户标识符和一个安全的唯一标识符，用于确认操作。
    """
    # 初始化一个URL安全的、带时间限制的序列化器，用于生成安全的令牌
    serializer = URLSafeTimedSerializer(tenent_id)
    # 生成一个UUID，并使用租户ID作为盐进行序列化，然后返回序列化后的字符串的特定子串
    # 这里返回的子串是为了解析和验证令牌时所需，同时保证令牌的长度适中
    return "ragflow-" + serializer.dumps(get_uuid(), salt=tenent_id)[2:34]


@manager.route('/new_token', methods=['POST'])
@validate_request("dialog_id")
@login_required
def new_token():
    """
    创建新的对话令牌。

    本函数用于处理POST请求，路径为'/new_token'。它的目的是为当前用户创建一个新的对话令牌。
    请求体应包含一个'dialog_id'字段。函数将验证请求，确保用户已登录，并查询用户的租户信息。
    如果租户信息存在，将生成一个新的对话令牌，并将其与其他相关信息一起保存。
    如果保存成功，将返回新的令牌信息；否则，返回错误信息。

    :return: JSON响应，包含新生成的令牌信息或错误信息。
    """
    # 解析请求体中的JSON数据
    req = request.json
    try:
        # 查询当前用户所关联的租户信息
        tenants = UserTenantService.query(user_id=current_user.id)
        # 如果没有找到租户信息，返回错误响应
        if not tenants:
            return get_data_error_result(retmsg="Tenant not found!")

        # 获取第一个租户的租户ID
        tenant_id = tenants[0].tenant_id
        # 构建令牌对象，包含租户ID、生成的令牌、对话ID及其他创建时间信息
        obj = {"tenant_id": tenant_id, "token": generate_confirmation_token(tenant_id),
               "dialog_id": req["dialog_id"],
               "create_time": current_timestamp(),
               "create_date": datetime_format(datetime.now()),
               "update_time": None,
               "update_date": None
               }
        # 尝试保存令牌对象，如果保存失败，返回错误响应
        if not APITokenService.save(**obj):
            return get_data_error_result(retmsg="Fail to new a dialog!")

        # 如果保存成功，返回包含令牌信息的响应
        return get_json_result(data=obj)
    except Exception as e:
        # 如果发生异常，返回服务器错误响应
        return server_error_response(e)


@manager.route('/token_list', methods=['GET'])
@login_required
def token_list():
    """
    获取用户API令牌列表。

    本函数通过查询与用户关联的租户信息，以及该租户下的所有API令牌，
    并将这些令牌的信息以JSON格式返回给已登录的用户。

    :return: 包含API令牌信息的JSON数组，或错误信息的JSON对象。
    """
    try:
        # 查询当前用户所关联的租户信息
        tenants = UserTenantService.query(user_id=current_user.id)
        # 如果没有找到租户信息，返回错误信息
        if not tenants:
            return get_data_error_result(retmsg="Tenant not found!")

        # 根据第一个租户的ID查询对应的API令牌信息
        # 这里假设用户只关联了一个租户，这是一种简化处理
        objs = APITokenService.query(tenant_id=tenants[0].tenant_id, dialog_id=request.args["dialog_id"])
        # 将查询到的API令牌信息转换为JSON格式并返回
        return get_json_result(data=[o.to_dict() for o in objs])
    except Exception as e:
        # 如果发生异常，返回服务器错误信息
        return server_error_response(e)


@manager.route('/rm', methods=['POST'])
@validate_request("tokens", "tenant_id")
@login_required
def rm():
    """
    删除指定租户的API令牌。

    该路由通过POST请求来删除指定租户ID下的一个或多个API令牌。请求体应包含tokens数组和tenant_id字段。
    tokens数组列出了需要删除的API令牌的标识符。

    :return: 删除成功时返回一个包含成功标志的JSON对象；如果发生异常，则返回一个包含错误信息的JSON对象。
    """
    # 解析请求体中的JSON数据
    req = request.json
    try:
        # 遍历请求中指定的令牌列表，并批量删除这些令牌
        for token in req["tokens"]:
            APITokenService.filter_delete(
                [APIToken.tenant_id == req["tenant_id"], APIToken.token == token])
        # 返回一个表示操作成功的JSON响应
        return get_json_result(data=True)
    except Exception as e:
        # 如果在处理过程中发生异常，返回一个包含异常信息的JSON响应
        return server_error_response(e)


@manager.route('/stats', methods=['GET'])
@login_required
def stats():
    """
    查询统计信息的接口。

    需要登录才能访问，提供从特定租户开始的统计信息。
    统计信息包括页面访问量(pv)、独立访客数(uv)、响应速度、令牌使用量、对话轮次和点赞数。
    可以通过查询参数from_date和to_date指定统计的时间范围。
    """
    try:
        # 根据当前用户查询其关联的租户信息
        tenants = UserTenantService.query(user_id=current_user.id)

        # 如果没有找到租户信息，返回错误结果
        if not tenants:
            return get_data_error_result(retmsg="Tenant not found!")
        objs = API4ConversationService.stats(
            tenants[0].tenant_id,
            request.args.get(
                "from_date",
                (datetime.now() -
                 timedelta(
                    days=7)).strftime("%Y-%m-%d 24:00:00")),
            request.args.get(
                "to_date",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        res = {
            "pv": [(o["dt"], o["pv"]) for o in objs],
            "uv": [(o["dt"], o["uv"]) for o in objs],
            "speed": [(o["dt"], float(o["tokens"])/(float(o["duration"]+0.1))) for o in objs],
            "tokens": [(o["dt"], float(o["tokens"])/1000.) for o in objs],
            "round": [(o["dt"], o["round"]) for o in objs],
            "thumb_up": [(o["dt"], o["thumb_up"]) for o in objs]
        }

        # 返回统计结果的JSON响应
        return get_json_result(data=res)
    except Exception as e:
        # 如果发生异常，返回服务器错误响应
        return server_error_response(e)


@manager.route('/new_conversation', methods=['GET'])
def set_conversation():
    """
    创建新的对话。

    通过验证请求头中的Authorization token来验证用户身份，
    然后根据用户请求创建一个新的对话实例，并将其保存到数据库中。
    如果对话创建成功，返回新的对话数据；否则，返回相应的错误信息。

    :return: 新对话的数据或错误信息。
    """
    # 从请求头中获取Authorization token并进行分割以获取实际的token值
    token = request.headers.get('Authorization').split()[1]
    # 使用token查询API令牌数据库，以验证token的有效性
    objs = APIToken.query(token=token)
    # 如果查询结果为空，表示token无效，返回认证错误信息
    if not objs:
        return get_json_result(
            data=False, retmsg='Token is not valid!"', retcode=RetCode.AUTHENTICATION_ERROR)
    # 从请求体中获取JSON数据
    req = request.json
    try:
        # 根据查询到的dialog_id获取对话服务实例和对话配置
        e, dia = DialogService.get_by_id(objs[0].dialog_id)
        # 如果获取失败，表示对话不存在，返回错误信息
        if not e:
            return get_data_error_result(retmsg="Dialog not found")
        # 构建新的对话实例数据
        conv = {
            "id": get_uuid(),
            "dialog_id": dia.id,
            "user_id": request.args.get("user_id", ""),
            "message": [{"role": "assistant", "content": dia.prompt_config["prologue"]}]
        }
        # 将新的对话实例保存到数据库
        API4ConversationService.save(**conv)
        # 根据对话ID从数据库中获取新的对话实例
        e, conv = API4ConversationService.get_by_id(conv["id"])
        # 如果获取失败，表示对话创建失败，返回错误信息
        if not e:
            return get_data_error_result(retmsg="Fail to new a conversation!")
        # 将对话实例转换为字典格式并返回
        conv = conv.to_dict()
        return get_json_result(data=conv)
    except Exception as e:
        # 如果发生异常，返回服务器错误信息
        return server_error_response(e)


# 定义一个处理对话完成请求的路由
@manager.route('/completion', methods=['POST'])
@validate_request("conversation_id", "messages")
def completion():
    # 从请求头中获取授权令牌
    token = request.headers.get('Authorization').split()[1]
    # 校验令牌的有效性
    if not APIToken.query(token=token):
        return get_json_result(
            data=False, retmsg='Token is not valid!"', retcode=RetCode.AUTHENTICATION_ERROR)
    # 解析请求体中的数据
    req = request.json
    # 根据对话ID获取对话信息
    e, conv = API4ConversationService.get_by_id(req["conversation_id"])
    if not e:
        return get_data_error_result(retmsg="Conversation not found!")
    # 默认quote值为False
    if "quote" not in req: req["quote"] = False

    # 筛选出用户消息
    msg = []
    for m in req["messages"]:
        if m["role"] == "system":
            continue
        if m["role"] == "assistant" and not msg:
            continue
        msg.append({"role": m["role"], "content": m["content"]})

    try:
        # 将最后一条用户消息添加到对话中
        conv.message.append(msg[-1])
        # 根据对话ID获取对话数据
        e, dia = DialogService.get_by_id(conv.dialog_id)
        if not e:
            return get_data_error_result(retmsg="Dialog not found!")
        # 删除请求体中的对话ID和消息信息，因为不再需要
        del req["conversation_id"]
        del req["messages"]

        # 如果对话没有参考信息，则初始化为空列表
        if not conv.reference:
            conv.reference = []
        # 添加一条空的助手消息和参考信息，用于后续填充答案
        conv.message.append({"role": "assistant", "content": ""})
        conv.reference.append({"chunks": [], "doc_aggs": []})

        # 定义一个函数，用于填充对话的参考信息和答案
        def fillin_conv(ans):
            nonlocal conv
            if not conv.reference:
                conv.reference.append(ans["reference"])
            else: conv.reference[-1] = ans["reference"]
            conv.message[-1] = {"role": "assistant", "content": ans["answer"]}

        # 定义一个函数，用于将答案中的docnm_kwd字段重命名为doc_name
        def rename_field(ans):
            reference = ans['reference']
            if not isinstance(reference, dict):
                return
            for chunk_i in reference.get('chunks', []):
                if 'docnm_kwd' in chunk_i:
                    chunk_i['doc_name'] = chunk_i['docnm_kwd']
                    chunk_i.pop('docnm_kwd')

        # 定义一个生成器，用于实时返回对话答案
        def stream():
            nonlocal dia, msg, req, conv
            try:
                for ans in chat(dia, msg, True, **req):
                    fillin_conv(ans)
                    rename_field(ans)
                    yield "data:" + json.dumps({"retcode": 0, "retmsg": "", "data": ans}, ensure_ascii=False) + "\n\n"
                API4ConversationService.append_message(conv.id, conv.to_dict())
            except Exception as e:
                yield "data:" + json.dumps({"retcode": 500, "retmsg": str(e),
                                            "data": {"answer": "**ERROR**: "+str(e), "reference": []}},
                                           ensure_ascii=False) + "\n\n"
            yield "data:"+json.dumps({"retcode": 0, "retmsg": "", "data": True}, ensure_ascii=False) + "\n\n"

        # 根据请求中的stream参数决定是实时返回答案还是一次性返回最终答案
        if req.get("stream", True):
            resp = Response(stream(), mimetype="text/event-stream")
            resp.headers.add_header("Cache-control", "no-cache")
            resp.headers.add_header("Connection", "keep-alive")
            resp.headers.add_header("X-Accel-Buffering", "no")
            resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
            return resp
        else:
            answer = None
            for ans in chat(dia, msg, **req):
                answer = ans
                fillin_conv(ans)
                API4ConversationService.append_message(conv.id, conv.to_dict())
                break

            rename_field(answer)
            return get_json_result(data=answer)

    except Exception as e:
        # 返回服务器错误响应
        return server_error_response(e)


@manager.route('/conversation/<conversation_id>', methods=['GET'])
# @login_required
def get(conversation_id):
    """
    根据对话ID获取对话信息。

    此路由处理GET请求，用于通过对话ID从数据库中检索特定对话的信息。
    如果对话不存在，则返回错误信息。如果对话存在，将对话信息转换为JSON格式并返回。

    参数:
    - conversation_id: 对话的唯一标识符。

    返回:
    - 如果对话不存在，返回一个包含错误信息的JSON对象。
    - 如果对话存在，返回一个包含对话详细信息的JSON对象。
    """
    try:
        # 尝试通过ID获取对话及其相关信息。
        e, conv = API4ConversationService.get_by_id(conversation_id)
        # 如果获取过程中存在错误（例如对话不存在），则返回错误信息。
        if not e:
            return get_data_error_result(retmsg="Conversation not found!")

        # 将对话对象转换为字典格式，便于后续处理。
        conv = conv.to_dict()
        # 遍历对话中的引用信息，处理特定字段。
        for referenct_i in conv['reference']:
            if referenct_i is None or len(referenct_i) == 0:
                continue
            for chunk_i in referenct_i['chunks']:
                # 如果chunk中包含'docnm_kwd'字段，将其值赋给'doc_name'字段，并移除'docnm_kwd'字段。
                if 'docnm_kwd' in chunk_i.keys():
                    chunk_i['doc_name'] = chunk_i['docnm_kwd']
                    chunk_i.pop('docnm_kwd')
        # 返回处理后的对话信息。
        return get_json_result(data=conv)
    except Exception as e:
        # 如果在处理过程中发生异常，返回服务器错误信息。
        return server_error_response(e)


# 定义一个上传文档的路由，该路由仅支持POST方法
@manager.route('/document/upload', methods=['POST'])
@validate_request("kb_name")
def upload():
    # 从请求头中获取Authorization信息，分离出token
    token = request.headers.get('Authorization').split()[1]
    # 根据token查询APIToken表中的数据
    objs = APIToken.query(token=token)
    # 如果查询结果为空，表示token无效，返回认证错误信息
    if not objs:
        return get_json_result(
            data=False, retmsg='Token is not valid!"', retcode=RetCode.AUTHENTICATION_ERROR)

    # 从请求表单中获取kb_name，并去除首尾空格
    kb_name = request.form.get("kb_name").strip()
    # 获取查询结果中第一条数据的tenant_id
    tenant_id = objs[0].tenant_id

    try:
        # 根据kb_name和tenant_id查询知识库信息
        e, kb = KnowledgebaseService.get_by_name(kb_name, tenant_id)
        # 如果查询不到知识库，返回错误信息
        if not e:
            return get_data_error_result(
                retmsg="Can't find this knowledgebase!")
        # 获取知识库id
        kb_id = kb.id
    except Exception as e:
        # 如果查询过程中出现异常，返回服务器错误信息
        return server_error_response(e)

    # 检查请求中是否存在文件部分
    if 'file' not in request.files:
        return get_json_result(
            data=False, retmsg='No file part!', retcode=RetCode.ARGUMENT_ERROR)

    # 获取上传的文件对象
    file = request.files['file']
    # 如果文件名为空，返回错误信息
    if file.filename == '':
        return get_json_result(
            data=False, retmsg='No file selected!', retcode=RetCode.ARGUMENT_ERROR)

    # 根据tenant_id获取根文件夹id
    root_folder = FileService.get_root_folder(tenant_id)
    pf_id = root_folder["id"]
    # 初始化知识库文档目录
    FileService.init_knowledgebase_docs(pf_id, tenant_id)
    # 获取知识库根文件夹
    kb_root_folder = FileService.get_kb_folder(tenant_id)
    # 创建新的知识库文件夹
    kb_folder = FileService.new_a_file_from_kb(kb.tenant_id, kb.name, kb_root_folder["id"])

    try:
        # 检查知识库文档数量是否超过限制
        if DocumentService.get_doc_count(kb.tenant_id) >= int(os.environ.get('MAX_FILE_NUM_PER_USER', 8192)):
            return get_data_error_result(
                retmsg="Exceed the maximum file number of a free user!")

        # 生成不重复的文件名
        filename = duplicate_name(
            DocumentService.query,
            name=file.filename,
            kb_id=kb_id)
        # 获取文件类型
        filetype = filename_type(filename)
        # 如果不支持该文件类型，返回错误信息
        if not filetype:
            return get_data_error_result(
                retmsg="This type of file has not been supported yet!")

        # 生成文件在存储系统中的唯一路径
        location = filename
        while MINIO.obj_exist(kb_id, location):
            location += "_"
        # 读取文件内容并上传到存储系统
        blob = request.files['file'].read()
        MINIO.put(kb_id, location, blob)
        # 构建文档信息字典
        doc = {
            "id": get_uuid(),
            "kb_id": kb.id,
            "parser_id": kb.parser_id,
            "parser_config": kb.parser_config,
            "created_by": kb.tenant_id,
            "type": filetype,
            "name": filename,
            "location": location,
            "size": len(blob),
            "thumbnail": thumbnail(filename, blob)
        }

        # 处理解析器id
        form_data=request.form
        if "parser_id" in form_data.keys():
            if request.form.get("parser_id").strip() in list(vars(ParserType).values())[1:-3]:
                doc["parser_id"] = request.form.get("parser_id").strip()
        # 根据文件类型设置默认解析器
        if doc["type"] == FileType.VISUAL:
            doc["parser_id"] = ParserType.PICTURE.value
        if doc["type"] == FileType.AURAL:
            doc["parser_id"] = ParserType.AUDIO.value
        if re.search(r"\.(ppt|pptx|pages)$", filename):
            doc["parser_id"] = ParserType.PRESENTATION.value

        # 插入文档信息到数据库
        doc_result = DocumentService.insert(doc)
        # 在知识库文件夹下添加文件信息
        FileService.add_file_from_kb(doc, kb_folder["id"], kb.tenant_id)
    except Exception as e:
        # 如果处理过程中出现异常，返回服务器错误信息
        return server_error_response(e)

    # 如果请求中包含运行参数
    if "run" in form_data.keys():
        if request.form.get("run").strip() == "1":
            try:
                # 更新文档状态为运行中
                info = {"run": 1, "progress": 0}
                info["progress_msg"] = ""
                info["chunk_num"] = 0
                info["token_num"] = 0
                DocumentService.update_by_id(doc["id"], info)
                # 删除旧任务
                tenant_id = DocumentService.get_tenant_id(doc["id"])
                if not tenant_id:
                    return get_data_error_result(retmsg="Tenant not found!")
                TaskService.filter_delete([Task.doc_id == doc["id"]])
                # 创建并启动新任务
                e, doc = DocumentService.get_by_id(doc["id"])
                doc = doc.to_dict()
                doc["tenant_id"] = tenant_id
                bucket, name = File2DocumentService.get_minio_address(doc_id=doc["id"])
                queue_tasks(doc, bucket, name)
            except Exception as e:
                 return server_error_response(e)

    # 返回处理结果
    return get_json_result(data=doc_result.to_json())


@manager.route('/list_chunks', methods=['POST'])
# @login_required
def list_chunks():
    """
    列出文档的片段列表。

    通过POST请求调用，需要提供Authorization头部信息以验证token。
    请求体中应包含doc_name或doc_id来指定文档。

    返回文档片段的内容、文档名称和图片ID（如果有的话）。
    """
    # 从请求头中获取Authorization信息，并提取token
    token = request.headers.get('Authorization').split()[1]
    # 使用token查询API令牌的有效性
    objs = APIToken.query(token=token)
    # 如果查询结果为空，表示token无效，返回认证错误信息
    if not objs:
        return get_json_result(
            data=False, retmsg='Token is not valid!"', retcode=RetCode.AUTHENTICATION_ERROR)

    req = request.json
    try:
        # 检查请求体中是否包含doc_name，如果包含，通过doc_name获取tenant_id和doc_id
        if "doc_name" in req.keys():
            tenant_id = DocumentService.get_tenant_id_by_name(req['doc_name'])
            doc_id = DocumentService.get_doc_id_by_doc_name(req['doc_name'])
        # 如果请求体中包含doc_id，但不包含doc_name，直接使用doc_id获取tenant_id
        elif "doc_id" in req.keys():
            tenant_id = DocumentService.get_tenant_id(req['doc_id'])
            doc_id = req['doc_id']
        # 如果请求体中既不包含doc_name也不包含doc_id，返回错误信息
        else:
            return get_json_result(
                data=False, retmsg="Can't find doc_name or doc_id"
            )
        # 调用检索器获取文档片段列表，并格式化返回结果
        res = retrievaler.chunk_list(doc_id=doc_id, tenant_id=tenant_id)
        res = [
            {
                "content": res_item["content_with_weight"],
                "doc_name": res_item["docnm_kwd"],
                "img_id": res_item["img_id"]
            } for res_item in res
        ]
    except Exception as e:
        # 如果发生异常，返回服务器错误响应
        return server_error_response(e)
    # 返回格式化后的文档片段列表
    return get_json_result(data=res)


@manager.route('/list_kb_docs', methods=['POST'])
# @login_required
def list_kb_docs():
    """
    根据知识库名称获取知识库文档列表。

    本函数通过HTTP POST请求实现，需要提供授权令牌（Authorization header）以验证权限。
    请求体中应包含知识库名称（kb_name）以及其他可选参数，如分页信息和排序条件。
    返回结果为JSON格式，包含文档列表和总条数。

    :return: JSON格式的响应，包含文档列表和总条数。
    """
    # 从请求头中获取授权令牌
    token = request.headers.get('Authorization').split()[1]
    # 使用授权令牌查询API令牌信息
    objs = APIToken.query(token=token)
    # 如果查询结果为空，表示令牌无效，返回认证错误信息
    if not objs:
        return get_json_result(
            data=False, retmsg='Token is not valid!"', retcode=RetCode.AUTHENTICATION_ERROR)

    # 从请求体中获取参数
    req = request.json
    # 获取租户ID
    tenant_id = objs[0].tenant_id
    # 获取知识库名称，去除首尾空格
    kb_name = req.get("kb_name", "").strip()

    try:
        # 根据知识库名称和租户ID查询知识库信息
        e, kb = KnowledgebaseService.get_by_name(kb_name, tenant_id)
        # 如果查询失败，表示找不到该知识库，返回错误信息
        if not e:
            return get_data_error_result(
                retmsg="Can't find this knowledgebase!")
        # 获取知识库ID
        kb_id = kb.id

    except Exception as e:
        # 如果查询过程中出现异常，返回服务器错误信息
        return server_error_response(e)

    # 从请求体中获取分页参数
    page_number = int(req.get("page", 1))
    items_per_page = int(req.get("page_size", 15))
    orderby = req.get("orderby", "create_time")
    desc = req.get("desc", True)
    keywords = req.get("keywords", "")

    try:
        # 根据知识库ID、分页参数、排序条件和关键词查询文档列表
        docs, tol = DocumentService.get_by_kb_id(
            kb_id, page_number, items_per_page, orderby, desc, keywords)
        # 将文档列表格式化为指定的JSON格式
        docs = [{"doc_id": doc['id'], "doc_name": doc['name']} for doc in docs]

        # 返回查询结果，包含文档列表和总条数
        return get_json_result(data={"total": tol, "docs": docs})

    except Exception as e:
        # 如果查询过程中出现异常，返回服务器错误信息
        return server_error_response(e)


@manager.route('/document', methods=['DELETE'])
# @login_required
def document_rm():
    """
    删除文档接口。
    通过DELETE请求调用，使用Authorization头部信息验证token。
    请求体中包含待删除文档的名称或ID列表。
    如果删除成功，返回成功信息；如果删除过程中有错误，返回错误详情。
    """
    # 从请求头中获取Authorization信息，并提取token
    token = request.headers.get('Authorization').split()[1]
    # 根据token查询API令牌信息
    objs = APIToken.query(token=token)
    # 如果查询结果为空，表示token无效，返回认证错误信息
    if not objs:
        return get_json_result(
            data=False, retmsg='Token is not valid!"', retcode=RetCode.AUTHENTICATION_ERROR)

    # 获取查询到的API令牌的租户ID
    tenant_id = objs[0].tenant_id
    # 解析请求体中的JSON数据
    req = request.json
    # 初始化待删除文档的ID列表
    doc_ids = []
    try:
        # 根据文档名称获取文档ID，若存在则添加到doc_ids列表中
        doc_ids = [DocumentService.get_doc_id_by_doc_name(doc_name) for doc_name in req.get("doc_names", [])]
        # 将请求体中直接提供的文档ID添加到doc_ids列表中
        for doc_id in req.get("doc_ids", []):
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)

        # 如果doc_ids列表为空，表示未找到任何待删除的文档，返回错误信息
        if not doc_ids:
            return get_json_result(
                data=False, retmsg="Can't find doc_names or doc_ids"
            )

    except Exception as e:
        # 如果在处理过程中发生异常，返回服务器错误信息
        return server_error_response(e)

    # 获取租户的根文件夹ID
    root_folder = FileService.get_root_folder(tenant_id)
    pf_id = root_folder["id"]
    # 初始化知识库文档
    FileService.init_knowledgebase_docs(pf_id, tenant_id)

    # 初始化错误信息字符串
    errors = ""
    for doc_id in doc_ids:
        try:
            # 根据文档ID获取文档信息，如果获取失败，返回文档不存在的错误信息
            e, doc = DocumentService.get_by_id(doc_id)
            if not e:
                return get_data_error_result(retmsg="Document not found!")
            # 获取文档的租户ID，如果获取失败，返回租户不存在的错误信息
            tenant_id = DocumentService.get_tenant_id(doc_id)
            if not tenant_id:
                return get_data_error_result(retmsg="Tenant not found!")

            # 获取文档在MinIO中的存储地址
            b, n = File2DocumentService.get_minio_address(doc_id=doc_id)

            # 从数据库中删除文档，如果删除失败，返回数据库错误信息
            if not DocumentService.remove_document(doc, tenant_id):
                return get_data_error_result(
                    retmsg="Database error (Document removal)!")

            # 根据文档ID获取文件与文档关联信息，并删除关联的文件信息
            f2d = File2DocumentService.get_by_document_id(doc_id)
            FileService.filter_delete([File.source_type == FileSource.KNOWLEDGEBASE, File.id == f2d[0].file_id])
            # 删除文件与文档的关联信息
            File2DocumentService.delete_by_document_id(doc_id)

            # 从MinIO中删除文档文件
            MINIO.rm(b, n)
        except Exception as e:
            # 如果在处理单个文档的删除过程中发生异常，将异常信息添加到错误信息字符串中
            errors += str(e)

    # 如果存在错误信息，返回包含错误信息的响应
    if errors:
        return get_json_result(data=False, retmsg=errors, retcode=RetCode.SERVER_ERROR)

    # 如果所有文档都成功删除，返回删除成功的响应
    return get_json_result(data=True)


# 处理AI助手对话完成请求的路由
@manager.route('/completion_aibotk', methods=['POST'])
@validate_request("Authorization", "conversation_id", "word")
def completion_faq():
    # 导入base64模块用于处理图片数据
    import base64
    # 获取请求的JSON数据
    req = request.json

    # 提取请求中的认证令牌
    token = req["Authorization"]
    # 根据令牌查询API令牌信息
    objs = APIToken.query(token=token)
    # 如果没有找到有效令牌，返回认证错误响应
    if not objs:
        return get_json_result(
            data=False, retmsg='Token is not valid!"', retcode=RetCode.AUTHENTICATION_ERROR)

    # 根据对话ID查询对话信息
    e, conv = API4ConversationService.get_by_id(req["conversation_id"])
    # 如果没有找到对话信息，返回对话不存在的错误响应
    if not e:
        return get_data_error_result(retmsg="Conversation not found!")
    # 默认开启引用模式，如果请求中没有指定
    if "quote" not in req: req["quote"] = True

    # 构建用户消息
    msg = []
    msg.append({"role": "user", "content": req["word"]})

    try:
        conv.message.append(msg[-1])
        # 根据对话ID查询对话流程信息
        e, dia = DialogService.get_by_id(conv.dialog_id)
        # 如果没有找到对话流程信息，返回对话流程不存在的错误响应
        if not e:
            return get_data_error_result(retmsg="Dialog not found!")
        # 从请求中移除对话ID参数
        del req["conversation_id"]

        # 如果对话没有参考信息，则初始化为空列表
        if not conv.reference:
            conv.reference = []
        # 添加一个空的助手回复和参考信息到对话中
        conv.message.append({"role": "assistant", "content": ""})
        conv.reference.append({"chunks": [], "doc_aggs": []})

        # 定义一个内嵌函数，用于填充对话的参考信息和助手回复内容
        def fillin_conv(ans):
            nonlocal conv
            # 如果对话的参考信息为空，添加新的参考信息
            if not conv.reference:
                conv.reference.append(ans["reference"])
            # 否则，更新最后一个参考信息
            else: conv.reference[-1] = ans["reference"]
            # 更新助手的回复内容
            conv.message[-1] = {"role": "assistant", "content": ans["answer"]}

        # 初始化数据结构，用于存放对话回复中的图片信息
        data_type_picture = {
            "type": 3,
            "url": "base64 content"
        }
        # 初始化对话数据列表
        data = [
            {
                "type": 1,
                "content": ""
            }
        ]
        # 通过对话服务获取对话回复
        ans = ""
        for a in chat(dia, msg, stream=False, **req):
            ans = a
            break
        # 更新对话数据中的回复内容，移除特殊标记
        data[0]["content"] += re.sub(r'##\d\$\$', '', ans["answer"])
        # 填充对话的参考信息和助手回复内容
        fillin_conv(ans)
        # 将更新后的对话信息保存到数据库
        API4ConversationService.append_message(conv.id, conv.to_dict())

        # 从回复内容中提取图片信息的索引
        chunk_idxs = [int(match[2]) for match in re.findall(r'##\d\$\$', ans["answer"])]
        # 遍历图片信息索引，获取图片数据
        for chunk_idx in chunk_idxs[:1]:
            # 如果图片ID存在
            if ans["reference"]["chunks"][chunk_idx]["img_id"]:
                try:
                    # 从MINIO服务中获取图片数据
                    bkt, nm = ans["reference"]["chunks"][chunk_idx]["img_id"].split("-")
                    response = MINIO.get(bkt, nm)
                    # 将图片数据编码为base64格式，并添加到对话数据中
                    data_type_picture["url"] = base64.b64encode(response).decode('utf-8')
                    data.append(data_type_picture)
                    break
                except Exception as e:
                    # 如果获取图片数据失败，返回服务器错误响应
                    return server_error_response(e)

        # 构建并返回成功的响应数据
        response = {"code": 200, "msg": "success", "data": data}
        return response

    # 如果处理过程中发生异常，返回服务器错误响应
    except Exception as e:
        return server_error_response(e)


@manager.route('/retrieval', methods=['POST'])
@validate_request("kb_id", "question")
def retrieval():
    token = request.headers.get('Authorization').split()[1]
    objs = APIToken.query(token=token)
    if not objs:
        return get_json_result(
            data=False, retmsg='Token is not valid!"', retcode=RetCode.AUTHENTICATION_ERROR)

    req = request.json
    kb_id = req.get("kb_id")
    doc_ids = req.get("doc_ids", [])
    question = req.get("question")
    page = int(req.get("page", 1))
    size = int(req.get("size", 30))
    similarity_threshold = float(req.get("similarity_threshold", 0.2))
    vector_similarity_weight = float(req.get("vector_similarity_weight", 0.3))
    top = int(req.get("top_k", 1024))

    try:
        e, kb = KnowledgebaseService.get_by_id(kb_id)
        if not e:
            return get_data_error_result(retmsg="Knowledgebase not found!")

        embd_mdl = TenantLLMService.model_instance(
            kb.tenant_id, LLMType.EMBEDDING.value, llm_name=kb.embd_id)

        rerank_mdl = None
        if req.get("rerank_id"):
            rerank_mdl = TenantLLMService.model_instance(
                kb.tenant_id, LLMType.RERANK.value, llm_name=req["rerank_id"])

        if req.get("keyword", False):
            chat_mdl = TenantLLMService.model_instance(kb.tenant_id, LLMType.CHAT)
            question += keyword_extraction(chat_mdl, question)

        ranks = retrievaler.retrieval(question, embd_mdl, kb.tenant_id, [kb_id], page, size,
                                      similarity_threshold, vector_similarity_weight, top,
                                      doc_ids, rerank_mdl=rerank_mdl)
        for c in ranks["chunks"]:
            if "vector" in c:
                del c["vector"]

        return get_json_result(data=ranks)
    except Exception as e:
        if str(e).find("not_found") > 0:
            return get_json_result(data=False, retmsg=f'No chunk found! Check the chunk status please!',
                                   retcode=RetCode.DATA_ERROR)
        return server_error_response(e)
