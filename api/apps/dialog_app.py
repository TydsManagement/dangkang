from flask import request
from flask_login import login_required, current_user
from api.db.services.dialog_service import DialogService
from api.db import StatusEnum
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.user_service import TenantService
from api.utils.api_utils import server_error_response, get_data_error_result, validate_request
from api.utils import get_uuid
from api.utils.api_utils import get_json_result


@manager.route('/set', methods=['POST'])
@login_required
def set_dialog():
    """
    创建或更新对话框配置。

    该接口用于根据客户端传来的JSON数据，创建一个新的对话框或更新已存在的对话框配置。
    如果dialog_id不存在，则视为创建新对话框；如果dialog_id存在，则视为更新现有对话框。

    :return: 返回处理结果的JSON数据。
    """
    # 解析请求中的JSON数据
    req = request.json
    # 提取对话框ID，如果不存在则默认为空字符串
    dialog_id = req.get("dialog_id")
    # 提取对话框名称，如果不存在则默认为"New Dialog"
    name = req.get("name", "New Dialog")
    # 提取对话框描述，如果不存在则默认为"A helpful Dialog"
    description = req.get("description", "A helpful Dialog")
    # 提取对话框图标链接，如果不存在则默认为空字符串
    icon = req.get("icon", "")
    # 提取top_n值，表示对话生成时的候选项数量，默认为6
    top_n = req.get("top_n", 6)
    # 提取top_k值，表示对话生成时的考虑词汇量，默认为1024
    top_k = req.get("top_k", 1024)
    # 提取rerank_id，用于重新排序，默认为空字符串
    rerank_id = req.get("rerank_id", "")
    # 如果rerank_id为空，则显式设置为为空字符串
    if not rerank_id: req["rerank_id"] = ""
    # 提取相似度阈值，默认为0.1
    similarity_threshold = req.get("similarity_threshold", 0.1)
    # 提取向量相似度权重，默认为0.3
    vector_similarity_weight = req.get("vector_similarity_weight", 0.3)
    # 如果vector_similarity_weight为None，则默认为0.3
    if vector_similarity_weight is None: vector_similarity_weight = 0.3
    # 提取LLM设置，默认为空字典
    llm_setting = req.get("llm_setting", {})
    # 定义默认的prompt配置
    default_prompt = {
        "system": """你是一个智能助手，请总结知识库的内容来回答问题，请列举知识库中的数据详细回答。当所有知识库内容都与问题无关时，你的回答必须包括“知识库中未找到您要的答案！”这句话。回答需要考虑聊天历史。
以下是知识库：
{knowledge}
以上是知识库。""",
        "prologue": "您好，我是您的助手小樱，长得可爱又善良，can I help you?",
        "parameters": [
            {"key": "knowledge", "optional": False}
        ],
        "empty_response": "Sorry! 知识库中未找到相关内容！"
    }
    # 提取prompt_config，如果不存在则使用默认prompt配置
    prompt_config = req.get("prompt_config", default_prompt)
    # 如果prompt_config中的system为空，则使用默认的system配置
    if not prompt_config["system"]:
        prompt_config["system"] = default_prompt["system"]
    # 校验prompt_config中的参数是否有效
    for p in prompt_config["parameters"]:
        if p["optional"]:
            continue
        if prompt_config["system"].find("{%s}" % p["key"]) < 0:
            # 如果必需参数在system中不存在，则返回错误结果
            return get_data_error_result(
                retmsg="Parameter '{}' is not used".format(p["key"]))

    try:
        # 根据当前用户ID获取租户信息
        e, tenant = TenantService.get_by_id(current_user.id)
        if not e:
            # 如果租户信息不存在，则返回错误结果
            return get_data_error_result(retmsg="Tenant not found!")
        # 提取LLM ID，如果不存在则使用租户默认的LLM ID
        llm_id = req.get("llm_id", tenant.llm_id)
        # 如果dialog_id不存在，则视为创建新对话框
        if not dialog_id:
            # 如果未选择知识库，则返回错误结果
            if not req.get("kb_ids"):
                return get_data_error_result(
                    retmsg="Fail! Please select knowledgebase!")
            # 创建新的对话框配置
            dia = {
                "id": get_uuid(),
                "tenant_id": current_user.id,
                "name": name,
                "kb_ids": req["kb_ids"],
                "description": description,
                "llm_id": llm_id,
                "llm_setting": llm_setting,
                "prompt_config": prompt_config,
                "top_n": top_n,
                "top_k": top_k,
                "rerank_id": rerank_id,
                "similarity_threshold": similarity_threshold,
                "vector_similarity_weight": vector_similarity_weight,
                "icon": icon
            }
            # 保存新的对话框配置
            if not DialogService.save(**dia):
                # 如果保存失败，则返回错误结果
                return get_data_error_result(retmsg="Fail to new a dialog!")
            # 获取刚保存的对话框信息
            e, dia = DialogService.get_by_id(dia["id"])
            if not e:
                # 如果获取失败，则返回错误结果
                return get_data_error_result(retmsg="Fail to new a dialog!")
            # 返回新的对话框配置的JSON格式数据
            return get_json_result(data=dia.to_json())
        else:
            # 更新已存在的对话框配置
            del req["dialog_id"]
            if "kb_names" in req:
                del req["kb_names"]
            if not DialogService.update_by_id(dialog_id, req):
                # 如果更新失败，则返回错误结果
                return get_data_error_result(retmsg="Dialog not found!")
            # 获取更新后的对话框信息
            e, dia = DialogService.get_by_id(dialog_id)
            if not e:
                # 如果获取失败，则返回错误结果
                return get_data_error_result(retmsg="Fail to update a dialog!")
            # 处理对话框的knowledge IDs和names
            dia = dia.to_dict()
            dia["kb_ids"], dia["kb_names"] = get_kb_names(dia["kb_ids"])
            # 返回更新后的对话框配置的JSON格式数据
            return get_json_result(data=dia)
    except Exception as e:
        # 如果发生异常，则返回服务器错误响应
        return server_error_response(e)


@manager.route('/get', methods=['GET'])
@login_required
def get():
    """
    根据对话ID获取对话信息。

    本函数通过HTTP GET请求获取特定对话ID的对话记录，并返回该记录的详细信息。
    如果对话记录不存在，则返回错误信息。
    如果获取过程中发生异常，则返回服务器错误信息。

    :return: JSON格式的对话信息或错误信息。
    """
    # 从请求参数中获取对话ID
    dialog_id = request.args["dialog_id"]
    try:
        # 尝试根据对话ID获取对话记录及其相关信息
        e, dia = DialogService.get_by_id(dialog_id)
        # 如果获取失败，返回对话不存在的错误信息
        if not e:
            return get_data_error_result(retmsg="Dialog not found!")
        # 将对话记录转换为字典格式，并获取对话涉及的知识库名称
        dia = dia.to_dict()
        dia["kb_ids"], dia["kb_names"] = get_kb_names(dia["kb_ids"])
        # 返回获取成功的对话信息
        return get_json_result(data=dia)
    except Exception as e:
        # 如果获取过程中发生异常，返回服务器错误信息
        return server_error_response(e)


def get_kb_names(kb_ids):
    """
    根据知识库ID列表，获取对应知识库的名称列表。

    参数:
    kb_ids: 知识库ID的列表。

    返回:
    一个包含两个列表的元组，第一个列表是有效的知识库ID列表，第二个列表是相应的知识库名称列表。
    """
    # 初始化用于存储有效ID和名称的列表
    ids, nms = [], []

    # 遍历输入的知识库ID列表
    for kid in kb_ids:
        # 根据ID获取知识库实体和状态
        e, kb = KnowledgebaseService.get_by_id(kid)

        # 如果获取失败或知识库状态无效，则跳过当前循环
        if not e or kb.status != StatusEnum.VALID.value:
            continue

        # 如果知识库有效，则将ID和名称分别添加到对应的列表中
        ids.append(kid)
        nms.append(kb.name)

    # 返回有效的ID列表和名称列表
    return ids, nms


@manager.route('/list', methods=['GET'])
@login_required
def list_dialogs():
    """
    查询并返回对话列表的JSON结果。

    本函数提供了一个受保护的路由（需要登录），用于获取当前用户可用的对话列表。
    对话列表是根据创建时间倒序排列的，并且只包含有效状态的对话。

    返回:
        - 成功时，返回包含对话信息的JSON数组；
        - 失败时，返回包含错误信息的服务器错误响应。
    """
    try:
        # 查询有效的对话服务列表，按创建时间倒序排列
        diags = DialogService.query(
            tenant_id=current_user.id,
            status=StatusEnum.VALID.value,
            reverse=True,
            order_by=DialogService.model.create_time)

        # 将对话对象列表转换为字典列表，以便于JSON序列化
        diags = [d.to_dict() for d in diags]

        # 获取每个对话相关的知识库ID和名称，以便在返回的JSON中包含
        for d in diags:
            d["kb_ids"], d["kb_names"] = get_kb_names(d["kb_ids"])

        # 返回处理后的对话列表作为JSON结果
        return get_json_result(data=diags)
    except Exception as e:
        # 处理任何异常，并返回服务器错误响应
        return server_error_response(e)


@manager.route('/rm', methods=['POST'])
@login_required
@validate_request("dialog_ids")
def rm():
    """
    删除对话记录。

    该路由用于处理删除特定对话记录的请求。它首先验证请求中是否包含有效的对话ID列表，
    然后尝试更新这些对话的状态为无效，从而实现逻辑上的删除。

    请求方法:
    POST

    请求参数:
    - dialog_ids: 一个包含待删除对话ID的列表。

    返回值:
    - 如果删除成功，返回一个包含成功标记的JSON响应。
    - 如果删除过程中发生异常，返回一个包含错误信息的JSON响应。
    """
    # 解析请求中的JSON数据，其中应包含待删除的对话ID列表
    req = request.json
    try:
        # 更新指定对话ID的状态为无效，实现逻辑删除
        DialogService.update_many_by_id(
            [{"id": id, "status": StatusEnum.INVALID.value} for id in req["dialog_ids"]])
        # 返回成功响应，指示对话记录已成功标记为无效
        return get_json_result(data=True)
    except Exception as e:
        # 在发生异常时返回错误响应，包含异常信息
        return server_error_response(e)
