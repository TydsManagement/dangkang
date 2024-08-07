import datetime
import json
import traceback

from flask import request
from flask_login import login_required, current_user
from elasticsearch_dsl import Q

from rag.app.qa import rmPrefix, beAdoc
from rag.nlp import search, rag_tokenizer, keyword_extraction
from rag.utils.es_conn import ELASTICSEARCH
from rag.utils import rmSpace
from api.db import LLMType, ParserType
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import TenantLLMService
from api.db.services.user_service import UserTenantService
from api.utils.api_utils import server_error_response, get_data_error_result, validate_request
from api.db.services.document_service import DocumentService
from api.settings import RetCode, retrievaler, kg_retrievaler
from api.utils.api_utils import get_json_result
import hashlib
import re


@manager.route('/list', methods=['POST'])
@login_required
@validate_request("doc_id")
def list_chunk():
    """
    根据文档ID和查询条件，分页获取文档片段列表。

    此接口用于处理POST请求，通过文档ID、页码和每页大小等参数，
    从特定的租户索引中检索文档片段，并返回相应的搜索结果。

    :return: 搜索结果的JSON响应，包含总条数和文档片段信息。
    """
    # 解析请求中的JSON数据
    req = request.json
    # 提取文档ID
    doc_id = req["doc_id"]
    # 提取页码和每页大小，默认为1和30
    page = int(req.get("page", 1))
    size = int(req.get("size", 30))
    # 提取搜索关键词
    question = req.get("keywords", "")
    try:
        # 获取文档对应的租户ID
        tenant_id = DocumentService.get_tenant_id(req["doc_id"])
        # 如果租户ID不存在，返回错误结果
        if not tenant_id:
            return get_data_error_result(retmsg="Tenant not found!")
        # 根据文档ID获取文档信息
        e, doc = DocumentService.get_by_id(doc_id)
        # 如果文档不存在，返回错误结果
        if not e:
            return get_data_error_result(retmsg="Document not found!")
        # 构建查询条件
        query = {
            "doc_ids": [doc_id], "page": page, "size": size, "question": question, "sort": True
        }
        # 如果请求中包含可用性标志，添加到查询条件中
        if "available_int" in req:
            query["available_int"] = int(req["available_int"])
        # 执行搜索
        sres = retrievaler.search(query, search.index_name(tenant_id))
        # 初始化返回结果
        res = {"total": sres.total, "chunks": [], "doc": doc.to_dict()}
        # 遍历搜索结果，构建每个片段的信息
        for id in sres.ids:
            d = {
                "chunk_id": id,
                "content_with_weight": rmSpace(sres.highlight[id]) if question and id in sres.highlight else sres.field[
                    id].get(
                    "content_with_weight", ""),
                "doc_id": sres.field[id]["doc_id"],
                "docnm_kwd": sres.field[id]["docnm_kwd"],
                "important_kwd": sres.field[id].get("important_kwd", []),
                "img_id": sres.field[id].get("img_id", ""),
                "available_int": sres.field[id].get("available_int", 1),
                "positions": sres.field[id].get("position_int", "").split("\t")
            }
            # 如果位置信息长度符合要求，转换为列表格式
            if len(d["positions"]) % 5 == 0:
                poss = []
                for i in range(0, len(d["positions"]), 5):
                    poss.append([float(d["positions"][i]), float(d["positions"][i + 1]), float(d["positions"][i + 2]),
                                 float(d["positions"][i + 3]), float(d["positions"][i + 4])])
                d["positions"] = poss
            # 将片段信息添加到结果中
            res["chunks"].append(d)
        # 返回搜索结果
        return get_json_result(data=res)
    except Exception as e:
        # 如果异常信息包含"not_found"，返回文档未找到的错误结果
        if str(e).find("not_found") > 0:
            return get_json_result(data=False, retmsg=f'No chunk found!',
                                   retcode=RetCode.DATA_ERROR)
        # 其他异常情况，返回服务器错误响应
        return server_error_response(e)


@manager.route('/get', methods=['GET'])
@login_required
def get():
    """
    根据chunk_id获取特定chunk的信息。

    该路由处理GET请求，用于查询指定chunk_id对应的chunk数据。
    先尝试根据用户ID查询该用户所属的租户信息，然后根据租户信息和chunk_id从Elasticsearch中获取chunk数据。
    如果数据不存在，则返回相应的错误信息。

    Returns:
        json: 包含查询到的chunk数据的JSON响应，如果未找到数据，则返回错误信息。
    """
    # 从请求参数中获取chunk_id
    chunk_id = request.args["chunk_id"]
    try:
        # 查询当前用户所关联的租户信息
        tenants = UserTenantService.query(user_id=current_user.id)
        # 如果租户信息不存在，返回数据查询错误的响应
        if not tenants:
            return get_data_error_result(retmsg="Tenant not found!")

        # 根据租户信息和chunk_id从Elasticsearch中获取chunk数据
        res = ELASTICSEARCH.get(
            chunk_id, search.index_name(
                tenants[0].tenant_id))
        # 如果chunk数据未找到，返回错误响应
        if not res.get("found"):
            return server_error_response("Chunk not found")

        # 提取并处理chunk数据的ID和源信息
        id = res["_id"]
        res = res["_source"]
        res["chunk_id"] = id

        # 移除源信息中特定的字段，这些字段可能包含敏感或不需要的信息
        k = []
        for n in res.keys():
            if re.search(r"(_vec$|_sm_|_tks|_ltks)", n):
                k.append(n)
        for n in k:
            del res[n]

        # 返回处理后的chunk数据
        return get_json_result(data=res)
    except Exception as e:
        # 如果异常信息包含"NotFoundError"，返回数据查询错误的响应
        if str(e).find("NotFoundError") >= 0:
            return get_json_result(data=False, retmsg=f'Chunk not found!',
                                   retcode=RetCode.DATA_ERROR)
        # 对于其他异常，返回服务器错误的响应
        return server_error_response(e)



@manager.route('/set', methods=['POST'])
@login_required
@validate_request("doc_id", "chunk_id", "content_with_weight",
                  "important_kwd")
def set():
    """
    处理文档内容的设置请求。

    该函数接收一个POST请求，更新文档片段的内容和重要关键词信息。
    它首先解析请求中的JSON数据，然后对数据进行处理，包括分词和关键词提取。
    最后，它将更新的信息存储到搜索引擎中。

    :param doc_id: 文档的唯一标识符。
    :param chunk_id: 文档片段的唯一标识符。
    :param content_with_weight: 文档片段的内容，带权重信息。
    :param important_kwd: 文档片段的重要关键词。
    :return: 返回一个JSON结果，表示操作是否成功。
    """
    # 解析请求中的JSON数据
    req = request.json
    # 初始化文档片段的数据字典
    d = {
        "id": req["chunk_id"],
        "content_with_weight": req["content_with_weight"]}
    # 对内容进行分词处理
    d["content_ltks"] = rag_tokenizer.tokenize(req["content_with_weight"])
    # 对分词结果进行细粒度处理
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
    # 存储重要关键词
    d["important_kwd"] = req["important_kwd"]
    # 对重要关键词进行分词处理
    d["important_tks"] = rag_tokenizer.tokenize(" ".join(req["important_kwd"]))
    # 如果请求中包含可用整数信息，则存储该信息
    if "available_int" in req:
        d["available_int"] = req["available_int"]

    try:
        # 获取文档所属的租户ID
        tenant_id = DocumentService.get_tenant_id(req["doc_id"])
        # 如果租户ID不存在，返回错误信息
        if not tenant_id:
            return get_data_error_result(retmsg="Tenant not found!")

        # 获取文档的嵌入ID，并实例化相应的模型
        embd_id = DocumentService.get_embd_id(req["doc_id"])
        embd_mdl = TenantLLMService.model_instance(
            tenant_id, LLMType.EMBEDDING.value, embd_id)

        # 获取文档对象
        e, doc = DocumentService.get_by_id(req["doc_id"])
        # 如果文档不存在，返回错误信息
        if not e:
            return get_data_error_result(retmsg="Document not found!")

        # 如果文档解析类型为QA，则对内容进行特殊处理
        if doc.parser_id == ParserType.QA:
            # 分割内容为问题和答案
            arr = [
                t for t in re.split(
                    r"[\n\t]",
                    req["content_with_weight"]) if len(t) > 1]
            # 如果问题和答案数量不正确，返回错误信息
            if len(arr) != 2:
                return get_data_error_result(
                    retmsg="Q&A must be separated by TAB/ENTER key.")
            # 移除前缀
            q, a = rmPrefix(arr[0]), rmPrefix(arr[1])
            # 根据是否包含中文，对数据进行处理
            d = beAdoc(d, arr[0], arr[1], not any(
                [rag_tokenizer.is_chinese(t) for t in q + a]))

        # 对文档名称和内容进行编码，获取向量表示
        v, c = embd_mdl.encode([doc.name, req["content_with_weight"]])
        # 根据文档解析类型，调整向量的权重
        v = 0.1 * v[0] + 0.9 * v[1] if doc.parser_id != ParserType.QA else v[1]
        # 将向量存储到数据字典中
        d["q_%d_vec" % len(v)] = v.tolist()
        # 将更新后的文档片段信息存储到搜索引擎中
        ELASTICSEARCH.upsert([d], search.index_name(tenant_id))
        # 返回操作成功的JSON结果
        return get_json_result(data=True)
    except Exception as e:
        # 如果发生异常，返回错误的JSON结果
        return server_error_response(e)


@manager.route('/switch', methods=['POST'])
@login_required
@validate_request("chunk_ids", "available_int", "doc_id")
def switch():
    """
    根据提供的文档ID和块ID，切换指定块的可用性状态。

    该路由仅接受POST请求，并需要用户登录。请求体应包含doc_id（文档ID）、
    chunk_ids（需要切换状态的块ID列表）和available_int（表示可用性的整数值）。

    返回结果表示操作是否成功。
    """
    # 解析请求体中的JSON数据
    req = request.json
    try:
        # 通过文档ID获取租户ID，这用于确定块数据存储的索引
        tenant_id = DocumentService.get_tenant_id(req["doc_id"])
        # 如果租户ID不存在，则返回错误响应
        if not tenant_id:
            return get_data_error_result(retmsg="Tenant not found!")
        # 使用Elasticsearch客户端更新指定块的可用性状态
        # 如果更新失败，则返回错误响应
        if not ELASTICSEARCH.upsert([{"id": i, "available_int": int(req["available_int"])} for i in req["chunk_ids"]],
                                    search.index_name(tenant_id)):
            return get_data_error_result(retmsg="Index updating failure")
        # 如果更新成功，则返回成功响应
        return get_json_result(data=True)
    except Exception as e:
        # 如果处理过程中发生异常，则返回服务器错误响应
        return server_error_response(e)


@manager.route('/rm', methods=['POST'])
@login_required
@validate_request("chunk_ids", "doc_id")
def rm():
    """
    删除指定的chunk。

    该路由用于处理删除chunk的请求。首先，它验证请求中是否包含正确的chunk_ids参数，
    然后尝试通过Elasticsearch客户端删除这些chunk。如果删除操作成功，它将返回一个成功的结果；
    如果失败，它将返回一个错误消息，指示索引更新失败。

    :return: 删除操作的成功或失败结果。
    """
    # 解析请求中的JSON数据
    req = request.json
    try:
        # 使用Elasticsearch客户端的deleteByQuery方法删除指定的chunk
        # 如果删除失败，返回一个表示索引更新失败的错误结果
        if not ELASTICSEARCH.deleteByQuery(
                Q("ids", values=req["chunk_ids"]), search.index_name(current_user.id)):
            return get_data_error_result(retmsg="Index updating failure")
        e, doc = DocumentService.get_by_id(req["doc_id"])
        if not e:
            return get_data_error_result(retmsg="Document not found!")
        deleted_chunk_ids = req["chunk_ids"]
        chunk_number = len(deleted_chunk_ids)
        DocumentService.decrement_chunk_num(doc.id, doc.kb_id, 1, chunk_number, 0)
        # 如果删除成功，返回一个表示成功的结果
        return get_json_result(data=True)
    except Exception as e:
        # 如果在处理请求时发生异常，返回一个表示服务器错误的响应
        return server_error_response(e)


@manager.route('/create', methods=['POST'])
@login_required
@validate_request("doc_id", "content_with_weight")
def create():
    """
    创建文档块。

    该路由处理POST请求，用于根据提供的文档ID和内容创建文档块。
    它通过对内容和文档ID进行哈希来生成唯一的块ID，并对文档内容进行分词。
    同时，它还会更新文档的嵌入向量和相关统计信息。

    请求应包含JSON数据，其中至少包括doc_id和content_with_weight字段。

    :return: 返回创建的块ID或错误响应。
    """
    # 解析请求中的JSON数据
    req = request.json
    # 使用MD5哈希算法生成块ID
    md5 = hashlib.md5()
    md5.update((req["content_with_weight"] + req["doc_id"]).encode("utf-8"))
    chunck_id = md5.hexdigest()

    # 初始化块信息字典
    d = {"id": chunck_id, "content_ltks": rag_tokenizer.tokenize(req["content_with_weight"]),
         "content_with_weight": req["content_with_weight"]}
    # 对内容进行细粒度分词
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
    # 提取重要关键词
    d["important_kwd"] = req.get("important_kwd", [])
    d["important_tks"] = rag_tokenizer.tokenize(" ".join(req.get("important_kwd", [])))
    # 记录创建时间和戳
    d["create_time"] = str(datetime.datetime.now()).replace("T", " ")[:19]
    d["create_timestamp_flt"] = datetime.datetime.now().timestamp()

    try:
        # 根据doc_id获取文档信息
        e, doc = DocumentService.get_by_id(req["doc_id"])
        if not e:
            return get_data_error_result(retmsg="Document not found!")
        # 更新块信息字典中与文档相关的字段
        d["kb_id"] = [doc.kb_id]
        d["docnm_kwd"] = doc.name
        d["doc_id"] = doc.id

        # 获取文档所属的租户ID
        tenant_id = DocumentService.get_tenant_id(req["doc_id"])
        if not tenant_id:
            return get_data_error_result(retmsg="Tenant not found!")

        # 获取文档的嵌入模型和ID
        embd_id = DocumentService.get_embd_id(req["doc_id"])
        embd_mdl = TenantLLMService.model_instance(
            tenant_id, LLMType.EMBEDDING.value, embd_id)

        # 对文档名和内容进行编码，计算加权平均向量
        v, c = embd_mdl.encode([doc.name, req["content_with_weight"]])
        v = 0.1 * v[0] + 0.9 * v[1]
        d["q_%d_vec" % len(v)] = v.tolist()

        # 将块信息插入到搜索引擎中
        ELASTICSEARCH.upsert([d], search.index_name(tenant_id))

        # 更新文档的块数量和嵌入向量统计信息
        DocumentService.increment_chunk_num(
            doc.id, doc.kb_id, c, 1, 0)

        # 返回创建的块ID
        return get_json_result(data={"chunk_id": chunck_id})
    except Exception as e:
        # 返回服务器错误响应
        return server_error_response(e)

@manager.route('/retrieval_test', methods=['POST'])
@login_required
@validate_request("kb_id", "question")
def retrieval_test():
    """
    实现知识库检索测试功能。

    该路由处理POST请求，用于对特定知识库进行检索测试。接收包括页面信息、大小、问题、知识库ID等在内的请求数据，
    并根据这些数据进行检索操作，返回检索结果。

    :param kb_id: 知识库ID，用于指定检索的知识库。
    :param question: 用户提问，用于检索的问题。
    :param page: 检索结果的页码。
    :param size: 每页的结果数量。
    :param doc_ids: 指定的文档ID列表，用于限定检索范围。
    :param similarity_threshold: 相似度阈值，用于过滤检索结果。
    :param vector_similarity_weight: 向量相似度的权重，用于调整检索结果的排序。
    :param top: 检索结果返回的顶部K个文档。
    :return: 检索结果的JSON响应。
    """
    # 解析请求中的JSON数据
    req = request.json
    # 获取请求中的页面信息和大小参数，设置默认值
    page = int(req.get("page", 1))
    size = int(req.get("size", 30))
    # 获取问题和知识库ID
    question = req["question"]
    kb_id = req["kb_id"]
    # 获取可选的文档ID列表和其他参数
    doc_ids = req.get("doc_ids", [])
    similarity_threshold = float(req.get("similarity_threshold", 0.2))
    vector_similarity_weight = float(req.get("vector_similarity_weight", 0.3))
    top = int(req.get("top_k", 1024))
    try:
        # 根据知识库ID获取知识库实例和服务
        e, kb = KnowledgebaseService.get_by_id(kb_id)
        if not e:
            return get_data_error_result(retmsg="Knowledgebase not found!")
        # 获取嵌入模型实例
        embd_mdl = TenantLLMService.model_instance(
            kb.tenant_id, LLMType.EMBEDDING.value, llm_name=kb.embd_id)
        # 根据请求中是否提供rerank_id，获取rerank模型实例
        rerank_mdl = None
        if req.get("rerank_id"):
            rerank_mdl = TenantLLMService.model_instance(
                kb.tenant_id, LLMType.RERANK.value, llm_name=req["rerank_id"])

        if req.get("keyword", False):
            chat_mdl = TenantLLMService.model_instance(kb.tenant_id, LLMType.CHAT)
            question += keyword_extraction(chat_mdl, question)

        # 执行检索操作，传入相关参数
        retr = retrievaler if kb.parser_id != ParserType.KG else kg_retrievaler
        ranks = retr.retrieval(question, embd_mdl, kb.tenant_id, [kb_id], page, size,
                               similarity_threshold, vector_similarity_weight, top,
                               doc_ids, rerank_mdl=rerank_mdl)
        # 删除检索结果中不必要的vector信息
        for c in ranks["chunks"]:
            if "vector" in c:
                del c["vector"]
        # 返回处理后的检索结果
        return get_json_result(data=ranks)
    except Exception as e:
        # 如果异常信息包含"not_found"，返回特定错误消息
        if str(e).find("not_found") > 0:
            return get_json_result(data=False, retmsg=f'No chunk found! Check the chunk status please!',
                                   retcode=RetCode.DATA_ERROR)
        # 其他异常情况，返回服务器错误响应
        return server_error_response(e)


@manager.route('/knowledge_graph', methods=['GET'])
@login_required
def knowledge_graph():
    doc_id = request.args["doc_id"]
    req = {
        "doc_ids": [doc_id],
        "knowledge_graph_kwd": ["graph", "mind_map"]
    }
    tenant_id = DocumentService.get_tenant_id(doc_id)
    sres = retrievaler.search(req, search.index_name(tenant_id))
    obj = {"graph": {}, "mind_map": {}}
    for id in sres.ids[:2]:
        ty = sres.field[id]["knowledge_graph_kwd"]
        try:
            obj[ty] = json.loads(sres.field[id]["content_with_weight"])
        except Exception as e:
            print(traceback.format_exc(), flush=True)

    return get_json_result(data=obj)

