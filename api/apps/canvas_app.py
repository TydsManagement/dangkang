import json
from functools import partial

from flask import request, Response
from flask_login import login_required, current_user

from api.db.db_models import UserCanvas
from api.db.services.canvas_service import CanvasTemplateService, UserCanvasService
from api.utils import get_uuid
from api.utils.api_utils import get_json_result, server_error_response, validate_request
from graph.canvas import Canvas


@manager.route('/templates', methods=['GET'])
@login_required
def templates():
    """
    获取所有画布模板的列表，并以JSON格式返回。

    本函数通过查询CanvasTemplateService获取所有画布模板的信息，
    然后将这些信息转换为JSON格式，最后返回这个JSON对象。

    路由装饰器@manager.route定义了本函数处理的URL路径和HTTP方法。
    @login_required装饰器确保只有登录的用户才能访问本函数。

    返回:
        JSON对象，包含所有画布模板的详细信息。
    """
    # 调用CanvasTemplateService的get_all方法获取所有画布模板
    # 并将每个模板转换为字典格式，最后构建成一个列表
    template_list = [c.to_dict() for c in CanvasTemplateService.get_all()]

    # 调用get_json_result函数，将模板列表作为数据部分返回
    # get_json_result函数会将数据包装成标准的JSON响应返回给客户端
    return get_json_result(data=template_list)



@manager.route('/list', methods=['GET'])
@login_required
def canvas_list():
    """
    获取当前登录用户画布列表的JSON结果。

    本函数通过查询用户画布服务，获取当前登录用户的所有画布信息，
    并将这些信息转换为JSON格式返回。这样设计是为了支持API的GET请求，
    提供一个接口用于获取用户画布的列表。

    Returns:
        json: 包含用户画布信息的JSON对象列表。每个画布信息都是一个字典，
              字典的关键字和值对应画布的各种属性，如画布ID、名称等。
    """
    # 调用UserCanvasService的query方法查询当前用户的所有画布
    # 并将查询结果转换为字典列表，以便返回JSON格式的数据
    return get_json_result(data=sorted([c.to_dict() for c in \
                                        UserCanvasService.query(user_id=current_user.id)],
                                       key=lambda x: x["update_time"] * -1)
                           )


@manager.route('/rm', methods=['POST'])
@validate_request("canvas_ids")
@login_required
def rm():
    """
    删除用户画布。

    该路由处理POST请求，用于删除用户指定的画布。请求体中应包含一个名为"canvas_ids"的JSON数组，
    其中包含了待删除的画布的ID。该函数通过遍历这些ID，并调用UserCanvasService的delete_by_id方法来逐个删除画布。

    删除操作成功后，函数返回一个JSON对象，其中data字段为True，表示操作成功。
    """
    # 遍历请求体中的画布ID列表，并逐个删除这些画布
    for i in request.json["canvas_ids"]:
        UserCanvasService.delete_by_id(i)

    # 返回操作成功的提示信息
    return get_json_result(data=True)



@manager.route('/set', methods=['POST'])
@validate_request("dsl", "title")
@login_required
def save():
    """
    保存用户画布配置。

    该函数用于处理POST请求，用于创建或更新用户的画布配置。首先，它从请求中提取用户ID和DSL（领域特定语言）配置，
    并检查DSL是否已经是字符串，如果不是，则将其转换为JSON字符串。然后，它检查是否存在同名的画布，如果存在，则返回错误。
    如果不存在同名画布，它将生成一个唯一ID并尝试保存新的画布配置。如果更新已存在的画布配置，则直接更新该配置。

    :return: 返回保存结果的JSON响应。
    """
    # 解析请求中的JSON数据，并添加用户ID
    req = request.json
    req["user_id"] = current_user.id

    # 确保DSL参数是字符串格式
    if not isinstance(req["dsl"], str):
        req["dsl"] = json.dumps(req["dsl"], ensure_ascii=False)

    # 将DSL参数解析为Python对象，以便进一步处理
    req["dsl"] = json.loads(req["dsl"])

    # 如果是新画布，则检查标题是否重复，并生成唯一ID进行保存
    if "id" not in req:
        if UserCanvasService.query(user_id=current_user.id, title=req["title"].strip()):
            return server_error_response(ValueError("Duplicated title."))
        req["id"] = get_uuid()
        if not UserCanvasService.save(**req):
            return server_error_response("Fail to save canvas.")
    # 如果是更新画布，则直接根据ID进行更新
    else:
        UserCanvasService.update_by_id(req["id"], req)

    # 返回保存成功的响应，包含更新后的配置数据
    return get_json_result(data=req)


@manager.route('/get/<canvas_id>', methods=['GET'])
@login_required
def get(canvas_id):
    """
    根据canvas_id获取用户画布信息。

    本函数通过canvas_id查询用户画布服务，获取特定画布的信息。
    如果画布不存在，则返回错误响应；否则，将画布信息转换为JSON格式并返回。

    参数:
    - canvas_id: 画布的唯一标识符。

    返回:
    - 如果画布存在，返回画布的JSON表示。
    - 如果画布不存在，返回错误信息的JSON表示。
    """
    # 通过canvas_id查询用户画布服务，获取画布实体和可能的错误。
    e, c = UserCanvasService.get_by_id(canvas_id)

    # 如果查询存在错误，即画布不存在，则返回错误响应。
    if not e:
        return server_error_response("canvas not found.")

    # 如果查询成功，将画布实体转换为字典格式，并返回JSON结果。
    return get_json_result(data=c.to_dict())


@manager.route('/completion', methods=['POST'])
@validate_request("id")
@login_required
def run():
    """
    处理完成画布的请求。

    该函数接收一个画布ID，尝试找到对应的画布并执行它。如果请求中包含了消息，
    这些消息会被添加到画布的记录中。执行结果根据请求是否要求流式传输而有不同的处理方式。

    :return: 根据不同的情况，返回执行结果或者流式传输的响应。
    """
    # 解析请求中的JSON数据
    req = request.json
    # 获取请求中是否要求流式传输的标志
    stream = req.get("stream", True)

    # 根据画布ID尝试获取画布及其服务
    e, cvs = UserCanvasService.get_by_id(req["id"])
    # 如果没有找到画布，返回错误响应
    if not e:
        return server_error_response("canvas not found.")

    # 确保画布的DSL是字符串格式
    if not isinstance(cvs.dsl, str):
        cvs.dsl = json.dumps(cvs.dsl, ensure_ascii=False)

    # 初始化最终答案
    final_ans = {"reference": [], "content": ""}
    try:
        # 根据DSL和当前用户ID创建画布实例
        canvas = Canvas(cvs.dsl, current_user.id)
        # 如果请求中包含消息，添加到画布记录中
        if "message" in req:
            canvas.messages.append({"role": "user", "content": req["message"]})
            canvas.add_user_input(req["message"])
        # 运行画布，根据stream参数决定是否流式返回结果
        answer = canvas.run(stream=stream)
    except Exception as e:
        # 如果运行中出现异常，返回错误响应
        return server_error_response(e)

    assert answer is not None, "Nothing. Is it over?"
    # 如果要求流式传输结果
    if stream:
        # 确保答案是生成器类型
        assert isinstance(answer, partial), "Nothing. Is it over?"

        # 定义SSE（服务器发送事件）的生成器函数
        def sse():
            nonlocal answer, cvs
            try:
                # 遍历生成器答案，并更新最终答案
                for ans in answer():
                    for k in ans.keys():
                        final_ans[k] = ans[k]
                    # 将答案格式化为SSE的格式输出
                    ans = {"answer": ans["content"], "reference": ans.get("reference", [])}
                    yield "data:" + json.dumps({"retcode": 0, "retmsg": "", "data": ans}, ensure_ascii=False) + "\n\n"

                # 将最后的答案和参考添加到画布中，并更新画布的DSL
                canvas.messages.append({"role": "assistant", "content": final_ans["content"]})
                if final_ans.get("reference"):
                    canvas.reference.append(final_ans["reference"])
                cvs.dsl = json.loads(str(canvas))
                # 更新画布服务
                UserCanvasService.update_by_id(req["id"], cvs.to_dict())
            except Exception as e:
                # 如果在SSE过程中出现异常，输出错误信息
                yield "data:" + json.dumps({"retcode": 500, "retmsg": str(e),
                                            "data": {"answer": "**ERROR**: " + str(e), "reference": []}},
                                           ensure_ascii=False) + "\n\n"
            # 结束SSE传输
            yield "data:" + json.dumps({"retcode": 0, "retmsg": "", "data": True}, ensure_ascii=False) + "\n\n"

        # 构建并返回SSE响应
        resp = Response(sse(), mimetype="text/event-stream")
        # 设置响应头支持SSE
        resp.headers.add_header("Cache-control", "no-cache")
        resp.headers.add_header("Connection", "keep-alive")
        resp.headers.add_header("X-Accel-Buffering", "no")
        resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
        return resp

    final_ans["content"] = "\n".join(answer["content"]) if "content" in answer else ""
    # 如果不要求流式传输，将答案和参考添加到画布，更新画布服务，并返回画布DSL
    canvas.messages.append({"role": "assistant", "content": final_ans["content"]})
    if final_ans.get("reference"):
        canvas.reference.append(final_ans["reference"])
    cvs.dsl = json.loads(str(canvas))
    UserCanvasService.update_by_id(req["id"], cvs.to_dict())
    return get_json_result(data={"answer": final_ans["content"], "reference": final_ans.get("reference", [])})


@manager.route('/reset', methods=['POST'])
@validate_request("id")
@login_required
def reset():
    """
    重置画布的DSL（领域特定语言）。

    该接口用于接收POST请求，通过用户ID和画布ID来重置特定用户的画布DSL。
    它首先验证请求的合法性，然后尝试获取指定画布，接着对画布进行重置操作，
    最后更新画布的DSL并返回更新后的DSL内容。

    :return: JSON格式的结果，包含重置后的画布DSL。
    """
    # 解析请求中的JSON数据
    req = request.json
    try:
        # 根据canvas_id获取用户画布
        e, user_canvas = UserCanvasService.get_by_id(req["id"])
        if not e:
            return server_error_response("canvas not found.")
        # 创建一个新的画布实例，用于重置操作
        canvas = Canvas(json.dumps(user_canvas.dsl), current_user.id)
        # 重置画布
        canvas.reset()
        # 更新画布DSL至请求数据，准备更新数据库
        req["dsl"] = json.loads(str(canvas))
        # 更新数据库中的画布DSL
        UserCanvasService.update_by_id(req["id"], {"dsl": req["dsl"]})
        # 返回重置后的画布DSL
        return get_json_result(data=req["dsl"])
    except Exception as e:
        # 处理任何异常，并返回错误响应
        return server_error_response(e)



