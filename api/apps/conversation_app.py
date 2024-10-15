from copy import deepcopy
from flask import request, Response
from flask_login import login_required
from api.db.services.dialog_service import DialogService, ConversationService, chat
from api.utils.api_utils import server_error_response, get_data_error_result, validate_request
from api.utils import get_uuid
from api.utils.api_utils import get_json_result
import json


@manager.route('/set', methods=['POST'])
@login_required
def set_conversation():
    """
    处理设置对话的请求。

    如果请求中包含对话ID，则更新现有对话；
    否则，根据请求中的对话框ID创建新对话。

    :return: 更新或新建对话后的JSON响应。
    """
    # 解析请求中的JSON数据
    req = request.json
    # 获取对话ID，如果存在
    conv_id = req.get("conversation_id")

    if conv_id:
        # 从请求数据中删除对话ID，因为它将用于更新对话，而不是创建新对话
        del req["conversation_id"]
        try:
            # 尝试更新现有对话
            if not ConversationService.update_by_id(conv_id, req):
                # 如果找不到对话，则返回错误响应
                return get_data_error_result(retmsg="Conversation not found!")
            # 获取更新后的对话，以确保更新成功
            e, conv = ConversationService.get_by_id(conv_id)
            if not e:
                # 如果获取对话失败，则返回错误响应
                return get_data_error_result(
                    retmsg="Fail to update a conversation!")
            # 将对话转换为字典格式，准备返回给客户端
            conv = conv.to_dict()
            return get_json_result(data=conv)
        except Exception as e:
            # 如果发生异常，则返回服务器错误响应
            return server_error_response(e)

    try:
        # 获取请求中的对话框ID
        e, dia = DialogService.get_by_id(req["dialog_id"])
        if not e:
            # 如果找不到对话框，则返回错误响应
            return get_data_error_result(retmsg="Dialog not found")
        # 创建新对话的数据结构
        conv = {
            "id": get_uuid(),
            "dialog_id": req["dialog_id"],
            "name": req.get("name", "New conversation"),
            "message": [{"role": "assistant", "content": dia.prompt_config["prologue"]}]
        }
        # 保存新对话
        ConversationService.save(**conv)
        # 获取刚刚创建的对话，以确保创建成功
        e, conv = ConversationService.get_by_id(conv["id"])
        if not e:
            # 如果获取对话失败，则返回错误响应
            return get_data_error_result(retmsg="Fail to new a conversation!")
        # 将对话转换为字典格式，准备返回给客户端
        conv = conv.to_dict()
        return get_json_result(data=conv)
    except Exception as e:
        # 如果发生异常，则返回服务器错误响应
        return server_error_response(e)


@manager.route('/get', methods=['GET'])
@login_required
def get():
    """
    通过GET请求获取指定对话ID的信息。

    本函数旨在通过对话ID从数据库中检索对应的对话记录，并以JSON格式返回给客户端。
    使用了装饰器@login_required确保只有登录的用户才能访问此功能。

    Returns:
        JSON: 包含对话信息的JSON对象，如果对话不存在，则返回错误信息。
    """
    # 从请求参数中获取对话ID
    conv_id = request.args["conversation_id"]
    try:
        # 尝试通过对话ID获取对话记录及其相关信息
        e, conv = ConversationService.get_by_id(conv_id)
        # 如果获取失败，返回错误信息
        if not e:
            return get_data_error_result(retmsg="Conversation not found!")
        # 将对话对象转换为字典格式，以便于JSON序列化
        conv = conv.to_dict()
        # 返回成功的响应，包含对话信息
        return get_json_result(data=conv)
    except Exception as e:
        # 捕获任何异常，返回服务器错误响应
        return server_error_response(e)


@manager.route('/rm', methods=['POST'])
@login_required
def rm():
    """
    删除对话的接口。

    该接口通过POST请求调用，用于批量删除指定的对话。请求体中应包含一个名为"conversation_ids"的JSON字段，
    其中列出了要删除的对话的ID。

    参数:
    - conversation_ids: 一个包含要删除的对话ID的列表。

    返回:
    - 如果删除成功，返回一个表示成功的JSON对象。
    - 如果删除过程中出现异常，返回一个包含错误信息的JSON对象。
    """
    # 从请求体中获取要删除的对话ID列表
    conv_ids = request.json["conversation_ids"]
    try:
        # 遍历对话ID列表，逐个删除对话
        for cid in conv_ids:
            ConversationService.delete_by_id(cid)
        # 返回成功结果，指示对话删除成功
        return get_json_result(data=True)
    except Exception as e:
        # 如果删除过程中出现异常，返回错误信息
        return server_error_response(e)


@manager.route('/list', methods=['GET'])
@login_required
def list_conversation():
    """
    获取对话列表。

    本函数通过查询特定对话ID的对话记录，并以JSON格式返回查询结果。
    使用者必须先通过身份验证。

    参数:
    - dialog_id: 对话的唯一标识符，通过URL参数传递。

    返回:
    - 如果查询成功，返回包含对话记录的JSON数组。
    - 如果查询过程中发生异常，返回服务器错误响应。
    """
    # 从请求参数中获取对话ID
    dialog_id = request.args["dialog_id"]
    try:
        # 查询对话记录，按创建时间降序排列
        convs = ConversationService.query(
            dialog_id=dialog_id,
            order_by=ConversationService.model.create_time,
            reverse=True)
        # 将查询结果转换为字典列表格式
        convs = [d.to_dict() for d in convs]
        # 返回处理后的查询结果
        return get_json_result(data=convs)
    except Exception as e:
        # 处理查询过程中发生的异常
        return server_error_response(e)


@manager.route('/completion', methods=['POST'])
@login_required
def completion():
    """
    处理对话完成请求。

    该函数接收一个包含对话ID和消息的JSON请求，然后根据这些信息获取对话服务和对话，
    并根据请求的stream参数来决定是流式返回答案还是一次性返回答案。

    :return: 返回处理结果，可能是流式响应或JSON结果。
    """
    # 解析请求中的JSON数据
    req = request.json
    # 初始化消息列表，用于存储过滤后的消息
    msg = []
    # 过滤请求中的消息，排除系统消息和首个助手消息
    for m in req["messages"]:
        if m["role"] == "system":
            continue
        if m["role"] == "assistant" and not msg:
            continue
        msg.append({"role": m["role"], "content": m["content"]})
    # 根据对话ID获取对话对象和服务
    try:
        e, conv = ConversationService.get_by_id(req["conversation_id"])
        if not e:
            return get_data_error_result(retmsg="Conversation not found!")
        # 添加最新消息到对话中
        conv.message.append(deepcopy(msg[-1]))
        e, dia = DialogService.get_by_id(conv.dialog_id)
        if not e:
            return get_data_error_result(retmsg="Dialog not found!")
        # 准备更新对话对象的引用和消息
        del req["conversation_id"]
        del req["messages"]
        if not conv.reference:
            conv.reference = []
        conv.message.append({"role": "assistant", "content": ""})
        conv.reference.append({"chunks": [], "doc_aggs": []})
        # 定义填充对话函数，用于更新对话中的答案和引用
        def fillin_conv(ans):
            nonlocal conv
            if not conv.reference:
                conv.reference.append(ans["reference"])
            else:
                conv.reference[-1] = ans["reference"]
            conv.message[-1] = {"role": "assistant", "content": ans["answer"]}
        # 定义流式返回生成器
        def stream():
            nonlocal dia, msg, req, conv
            try:
                for ans in chat(dia, msg, True, **req):
                    fillin_conv(ans)
                    yield "data:"+json.dumps({"retcode": 0, "retmsg": "", "data": ans}, ensure_ascii=False) + "\n\n"
                ConversationService.update_by_id(conv.id, conv.to_dict())
            except Exception as e:
                yield "data:" + json.dumps({"retcode": 500, "retmsg": str(e),
                                            "data": {"answer": "**ERROR**: "+str(e), "reference": []}},
                                           ensure_ascii=False) + "\n\n"
            yield "data:"+json.dumps({"retcode": 0, "retmsg": "", "data": True}, ensure_ascii=False) + "\n\n"
        # 根据stream参数决定是流式返回还是一次性返回答案
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
                ConversationService.update_by_id(conv.id, conv.to_dict())
                break
            return get_json_result(data=answer)
    except Exception as e:
        return server_error_response(e)


@manager.route('/list_names', methods=['POST'])
@login_required
def list_conversation_names():
    """
    通过POST请求获取指定会话ID列表的会话名称。

    本函数接收一个包含多个对话ID的JSON请求体，通过这些ID从数据库中检索对应的会话名称，
    并以JSON格式返回。

    使用了装饰器@login_required确保只有登录的用户才能访问此功能。

    Returns:
        JSON: 包含会话名称列表的JSON对象，如果某个对话ID不存在，则返回错误信息。
    """
    try:
        # 从请求体中获取对话ID列表
        conv_ids = request.json.get("conversation_ids", [])

        if not conv_ids:
            return get_data_error_result(retmsg="No conversation IDs provided!")

        # 查询所有指定的对话记录
        convs = ConversationService.get_by_ids(conv_ids)

        # 创建一个字典，便于按顺序查找
        conv_dict = {conv.id: conv for conv in convs if conv is not None}

        # 按照传入的顺序构造结果列表
        results = [{"name": conv_dict[conv_id].name, "id": conv_id, "dialog_id": conv_dict[conv_id].dialog_id}
                   for conv_id in conv_ids if conv_id in conv_dict]

        # 返回成功的响应，包含对话名称和ID列表
        return get_json_result(data=results)
    except Exception as e:
        # 捕获任何异常，返回服务器错误响应
        return server_error_response(e)

