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
#  limitations under the License
#

import os
import pathlib
import re

import flask
from elasticsearch_dsl import Q
from flask import request
from flask_login import login_required, current_user

from api.db.db_models import Task, File
from api.db.services.file2document_service import File2DocumentService
from api.db.services.file_service import FileService
from api.db.services.task_service import TaskService, queue_tasks
from rag.nlp import search
from rag.utils.es_conn import ELASTICSEARCH
from api.db.services import duplicate_name
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.utils.api_utils import server_error_response, get_data_error_result, validate_request
from api.utils import get_uuid
from api.db import FileType, TaskStatus, ParserType, FileSource
from api.db.services.document_service import DocumentService
from api.settings import RetCode
from api.utils.api_utils import get_json_result
from rag.utils.minio_conn import MINIO
from api.utils.file_utils import filename_type, thumbnail
from api.utils.web_utils import html2pdf, is_valid_url
from api.utils.web_utils import html2pdf, is_valid_url


@manager.route('/upload', methods=['POST'])
@login_required
@validate_request("kb_id")
def upload():
    """
    上传文件到知识库。

    该接口用于用户上传文件到指定的知识库。首先验证请求中是否包含KB ID，然后检查上传的文件是否存在。
    接着，根据KB ID获取知识库信息，并为用户创建知识库的文件夹结构。随后，处理上传的每个文件，将文件存储到MINIO，
    并在数据库中记录文件信息。如果上传过程中有任何错误，将返回相应的错误信息。

    :return: 返回上传结果的JSON响应。
    """
    # 从请求中获取KB ID
    kb_id = request.form.get("kb_id")
    # 如果KB ID不存在，返回错误信息
    if not kb_id:
        return get_json_result(
            data=False, retmsg='Lack of "KB ID"', retcode=RetCode.ARGUMENT_ERROR)
    # 检查请求中是否包含文件部分
    if 'file' not in request.files:
        return get_json_result(
            data=False, retmsg='No file part!', retcode=RetCode.ARGUMENT_ERROR)

    file_objs = request.files.getlist('file')
    # 遍历文件列表，检查是否有未选择的文件
    for file_obj in file_objs:
        if file_obj.filename == '':
            return get_json_result(
                data=False, retmsg='No file selected!', retcode=RetCode.ARGUMENT_ERROR)

    # 根据KB ID获取知识库信息
    e, kb = KnowledgebaseService.get_by_id(kb_id)
    # 如果知识库不存在，抛出异常
    if not e:
        raise LookupError("Can't find this knowledgebase!")

    # 获取用户文件根目录，为用户在知识库中创建文件夹结构
    root_folder = FileService.get_root_folder(current_user.id)
    pf_id = root_folder["id"]
    FileService.init_knowledgebase_docs(pf_id, current_user.id)
    kb_root_folder = FileService.get_kb_folder(current_user.id)
    # 为知识库创建特定的文件夹
    kb_folder = FileService.new_a_file_from_kb(kb.tenant_id, kb.name, kb_root_folder["id"])

    err = []
    # 处理上传的每个文件
    for file in file_objs:
        try:
            # 检查是否超过用户可上传文件数量的限制
            MAX_FILE_NUM_PER_USER = int(os.environ.get('MAX_FILE_NUM_PER_USER', 0))
            if MAX_FILE_NUM_PER_USER > 0 and DocumentService.get_doc_count(kb.tenant_id) >= MAX_FILE_NUM_PER_USER:
                raise RuntimeError("Exceed the maximum file number of a free user!")

            # 为文件生成唯一名称，并确定文件类型
            filename = duplicate_name(
                DocumentService.query,
                name=file.filename,
                kb_id=kb.id)
            filetype = filename_type(filename)
            # 如果文件类型不受支持，抛出异常
            if filetype == FileType.OTHER.value:
                raise RuntimeError("This type of file has not been supported yet!")

            # 为文件生成存储位置，确保文件名的唯一性
            location = filename
            while MINIO.obj_exist(kb_id, location):
                location += "_"
            # 读取文件内容并存储到MINIO
            blob = file.read()
            MINIO.put(kb_id, location, blob)
            # 构建文件信息的文档，并根据文件类型调整解析器ID
            doc = {
                "id": get_uuid(),
                "kb_id": kb.id,
                "parser_id": kb.parser_id,
                "parser_config": kb.parser_config,
                "created_by": current_user.id,
                "type": filetype,
                "name": filename,
                "location": location,
                "size": len(blob),
                "thumbnail": thumbnail(filename, blob)
            }
            if doc["type"] == FileType.VISUAL:
                doc["parser_id"] = ParserType.PICTURE.value
            if doc["type"] == FileType.AURAL:
                doc["parser_id"] = ParserType.AUDIO.value
            if re.search(r"\.(ppt|pptx|pages)$", filename):
                doc["parser_id"] = ParserType.PRESENTATION.value
            # 将文件信息插入数据库
            DocumentService.insert(doc)

            # 在知识库文件夹中添加文件
            FileService.add_file_from_kb(doc, kb_folder["id"], kb.tenant_id)
        except Exception as e:
            # 记录文件上传过程中的错误
            err.append(file.filename + ": " + str(e))
    # 如果有错误发生，返回错误信息
    if err:
        return get_json_result(
            data=False, retmsg="\n".join(err), retcode=RetCode.SERVER_ERROR)
    # 上传成功，返回成功信息
    return get_json_result(data=True)


@manager.route('/web_crawl', methods=['POST'])
@login_required
@validate_request("kb_id", "name", "url")
def web_crawl():
    """
    实现网页抓取功能，将指定网页转换为PDF文件，并存储到对应的知识库中。
    该函数首先验证请求的合法性，然后获取网页内容并转换为PDF格式。
    最后，将转换后的PDF文件存储到用户的知识库文件夹中，并创建对应的文档记录。

    :return: 返回操作的结果，成功时返回True，失败时返回错误信息。
    """
    # 从请求中获取知识库ID
    kb_id = request.form.get("kb_id")
    # 如果没有提供知识库ID，返回错误信息
    if not kb_id:
        return get_json_result(
            data=False, retmsg='Lack of "KB ID"', retcode=RetCode.ARGUMENT_ERROR)
    name = request.form.get("name")
    url = request.form.get("url")
    # 验证URL的格式是否合法
    if not is_valid_url(url):
        return get_json_result(
            data=False, retmsg='The URL format is invalid', retcode=RetCode.ARGUMENT_ERROR)
    # 根据知识库ID获取知识库信息
    e, kb = KnowledgebaseService.get_by_id(kb_id)
    # 如果找不到对应的知识库，抛出异常
    if not e:
        raise LookupError("Can't find this knowledgebase!")

    # 将网页内容转换为PDF格式
    blob = html2pdf(url)
    # 如果转换失败，返回错误信息
    if not blob: return server_error_response(ValueError("Download failure."))

    # 获取当前用户的根文件夹ID
    root_folder = FileService.get_root_folder(current_user.id)
    pf_id = root_folder["id"]
    # 初始化知识库文档文件夹
    FileService.init_knowledgebase_docs(pf_id, current_user.id)
    # 获取知识库的根文件夹
    kb_root_folder = FileService.get_kb_folder(current_user.id)
    # 创建新的知识库文件夹
    kb_folder = FileService.new_a_file_from_kb(kb.tenant_id, kb.name, kb_root_folder["id"])

    try:
        # 生成不重复的文件名
        filename = duplicate_name(
            DocumentService.query,
            name=name+".pdf",
            kb_id=kb.id)
        # 获取文件类型
        filetype = filename_type(filename)
        # 如果文件类型不受支持，抛出异常
        if filetype == FileType.OTHER.value:
            raise RuntimeError("This type of file has not been supported yet!")

        # 生成文件在存储系统中的唯一位置
        location = filename
        while MINIO.obj_exist(kb_id, location):
            location += "_"
        # 将PDF内容存储到对应的位置
        MINIO.put(kb_id, location, blob)
        # 创建文档记录
        doc = {
            "id": get_uuid(),
            "kb_id": kb.id,
            "parser_id": kb.parser_id,
            "parser_config": kb.parser_config,
            "created_by": current_user.id,
            "type": filetype,
            "name": filename,
            "location": location,
            "size": len(blob),
            "thumbnail": thumbnail(filename, blob)
        }
        # 根据文件类型调整文档的解析器ID
        if doc["type"] == FileType.VISUAL:
            doc["parser_id"] = ParserType.PICTURE.value
        if doc["type"] == FileType.AURAL:
            doc["parser_id"] = ParserType.AUDIO.value
        if re.search(r"\.(ppt|pptx|pages)$", filename):
            doc["parser_id"] = ParserType.PRESENTATION.value
        # 插入文档记录
        DocumentService.insert(doc)
        # 在知识库文件夹中添加文件记录
        FileService.add_file_from_kb(doc, kb_folder["id"], kb.tenant_id)
    except Exception as e:
        # 如果发生异常，返回错误信息
        return server_error_response(e)
    # 如果操作成功，返回成功信息
    return get_json_result(data=True)


@manager.route('/create', methods=['POST'])
@login_required
@validate_request("name", "kb_id")
def create():
    """
    创建文档的接口。

    该接口用于在知识库中创建一个新的文档。接收POST请求，请求体中包含文档名称和知识库ID。
    如果请求合法，且知识库和文档名称不存在，则创建新文档并返回创建的文档信息。

    :return: 返回创建的文档信息或错误信息。
    """
    # 获取请求的JSON数据
    req = request.json
    # 提取请求中的知识库ID
    kb_id = req["kb_id"]

    # 检查是否提供了知识库ID
    if not kb_id:
        # 如果没有提供知识库ID，返回参数错误信息
        return get_json_result(
            data=False, retmsg='Lack of "KB ID"', retcode=RetCode.ARGUMENT_ERROR)

    try:
        # 根据知识库ID查询知识库信息
        e, kb = KnowledgebaseService.get_by_id(kb_id)
        # 如果查询出错，返回找不到知识库的错误信息
        if not e:
            return get_data_error_result(
                retmsg="Can't find this knowledgebase!")

        # 检查是否已存在同名文档
        if DocumentService.query(name=req["name"], kb_id=kb_id):
            # 如果已存在同名文档，返回重复文档名称的错误信息
            return get_data_error_result(
                retmsg="Duplicated document name in the same knowledgebase.")

        # 创建新文档并插入到数据库
        doc = DocumentService.insert({
            "id": get_uuid(),
            "kb_id": kb.id,
            "parser_id": kb.parser_id,
            "parser_config": kb.parser_config,
            "created_by": current_user.id,
            "type": FileType.VIRTUAL,
            "name": req["name"],
            "location": "",
            "size": 0
        })
        # 返回创建的文档信息
        return get_json_result(data=doc.to_json())
    except Exception as e:
        # 如果发生异常，返回服务器错误信息
        return server_error_response(e)


@manager.route('/list', methods=['GET'])
@login_required
def list_docs():
    """
    获取知识库文档列表。

    本函数提供了一个接口，用于根据KB ID（知识库标识符）和其他查询参数，
    获取指定知识库中的文档列表。支持分页和排序功能。

    :param kb_id: 知识库的ID，用于指定查询的知识库。
    :param keywords: 关键词，用于文档内容的搜索。
    :param page: 请求的页码，用于分页查询。
    :param page_size: 每页显示的文档数量，用于分页查询。
    :param orderby: 文档排序的字段，默认为创建时间。
    :param desc: 是否按降序排序，默认为True。

    :return: 返回一个JSON结果，包含文档列表和总数量。
    """
    # 从请求参数中获取知识库ID
    kb_id = request.args.get("kb_id")
    # 如果没有提供知识库ID，则返回错误响应
    if not kb_id:
        return get_json_result(
            data=False, retmsg='Lack of "KB ID"', retcode=RetCode.ARGUMENT_ERROR)
    # 从请求参数中获取关键词，默认为空字符串
    keywords = request.args.get("keywords", "")
    # 从请求参数中获取页码，默认为1
    page_number = int(request.args.get("page", 1))
    # 从请求参数中获取每页的文档数量，默认为15
    items_per_page = int(request.args.get("page_size", 15))
    # 从请求参数中获取排序字段，默认为创建时间
    orderby = request.args.get("orderby", "create_time")
    # 从请求参数中获取排序方式，默认为降序
    desc = request.args.get("desc", True)
    try:
        # 根据知识库ID和其他查询参数，获取文档列表和总数量
        docs, tol = DocumentService.get_by_kb_id(
            kb_id, page_number, items_per_page, orderby, desc, keywords)
        # 返回查询结果的JSON响应
        return get_json_result(data={"total": tol, "docs": docs})
    except Exception as e:
        # 如果发生异常，返回服务器错误响应
        return server_error_response(e)


@manager.route('/thumbnails', methods=['GET'])
@login_required
def thumbnails():
    """
    获取文档缩略图的接口。

    该接口用于根据提供的文档ID列表，获取对应文档的缩略图。需要用户登录。

    请求参数:
    - doc_ids: 通过查询参数传递，以逗号分隔的文档ID字符串。

    返回值:
    - 如果请求参数doc_ids为空，则返回错误信息，提示缺少文档ID。
    - 如果请求成功，返回一个JSON对象，其中键为文档ID，值为对应的缩略图URL。
    - 如果发生异常，返回服务器错误响应。
    """
    # 从请求参数中获取doc_ids，并按逗号分割成列表
    doc_ids = request.args.get("doc_ids").split(",")

    # 检查doc_ids是否为空，如果为空则返回缺少参数的错误响应
    if not doc_ids:
        return get_json_result(
            data=False, retmsg='Lack of "Document ID"', retcode=RetCode.ARGUMENT_ERROR)

    try:
        # 调用DocumentService的get_thumbnails方法获取缩略图，并以文档ID为键，缩略图URL为值构建字典
        docs = DocumentService.get_thumbnails(doc_ids)
        return get_json_result(data={d["id"]: d["thumbnail"] for d in docs})
    except Exception as e:
        # 如果发生异常，返回服务器错误响应
        return server_error_response(e)


@manager.route('/change_status', methods=['POST'])
@login_required
@validate_request("doc_id", "status")
def change_status():
    """
    根据提供的doc_id和status更新文档的状态。
    仅允许状态从0到1或从1到0的转变，并在更新文档状态后相应地更新Elasticsearch中的可用性标志。
    :return: 返回一个JSON结果，指示操作是否成功。
    """
    # 解析请求中的JSON数据
    req = request.json
    # 验证状态值是否为0或1
    if str(req["status"]) not in ["0", "1"]:
        # 如果状态值不合法，则返回错误响应
        get_json_result(
            data=False,
            retmsg='"Status" must be either 0 or 1!',
            retcode=RetCode.ARGUMENT_ERROR)

    try:
        # 根据doc_id获取文档对象
        e, doc = DocumentService.get_by_id(req["doc_id"])
        # 如果文档不存在，则返回错误响应
        if not e:
            return get_data_error_result(retmsg="Document not found!")
        # 根据文档关联的知识库ID获取知识库对象
        e, kb = KnowledgebaseService.get_by_id(doc.kb_id)
        # 如果知识库不存在，则返回错误响应
        if not e:
            return get_data_error_result(
                retmsg="Can't find this knowledgebase!")

        # 更新文档的状态
        if not DocumentService.update_by_id(
                req["doc_id"], {"status": str(req["status"])}):
            # 如果更新失败，则返回错误响应
            return get_data_error_result(
                retmsg="Database error (Document update)!")

        # 根据新的状态更新Elasticsearch中的文档可用性
        if str(req["status"]) == "0":
            # 如果状态为0，则将文档标记为不可用
            ELASTICSEARCH.updateScriptByQuery(Q("term", doc_id=req["doc_id"]),
                                              scripts="ctx._source.available_int=0;",
                                              idxnm=search.index_name(
                                                  kb.tenant_id)
                                              )
        else:
            # 如果状态为1，则将文档标记为可用
            ELASTICSEARCH.updateScriptByQuery(Q("term", doc_id=req["doc_id"]),
                                              scripts="ctx._source.available_int=1;",
                                              idxnm=search.index_name(
                                                  kb.tenant_id)
                                              )
        # 返回成功响应
        return get_json_result(data=True)
    except Exception as e:
        # 如果发生异常，则返回服务器错误响应
        return server_error_response(e)


@manager.route('/rm', methods=['POST'])
@login_required
@validate_request("doc_id")
def rm():
    """
    删除文档接口。

    该接口用于接收POST请求，删除指定的文档及其相关文件。首先，它会验证请求中是否包含有效的doc_id。
    然后，它会尝试删除这些文档，在删除过程中，如果遇到任何错误，都会记录下来，并在最后统一处理。

    :return: 删除操作的结果，成功返回True，失败返回错误信息。
    """
    # 解析请求中的JSON数据，获取doc_id
    req = request.json
    doc_ids = req["doc_id"]

    # 如果doc_ids是字符串，则转换为列表
    if isinstance(doc_ids, str): 
        doc_ids = [doc_ids]

    # 获取当前用户根目录的id
    root_folder = FileService.get_root_folder(current_user.id)
    pf_id = root_folder["id"]

    # 初始化知识库文档
    FileService.init_knowledgebase_docs(pf_id, current_user.id)

    # 用于记录删除过程中可能出现的错误
    errors = ""

    # 遍历待删除的文档id列表
    for doc_id in doc_ids:
        try:
            # 根据doc_id获取文档及其相关信息
            e, doc = DocumentService.get_by_id(doc_id)
            # 如果文档不存在，则返回错误信息
            if not e:
                return get_data_error_result(retmsg="Document not found!")

            # 获取文档所属的租户id
            tenant_id = DocumentService.get_tenant_id(doc_id)
            # 如果租户不存在，则返回错误信息
            if not tenant_id:
                return get_data_error_result(retmsg="Tenant not found!")

            # 获取文档在MinIO中的存储地址
            b, n = File2DocumentService.get_minio_address(doc_id=doc_id)

            # 从数据库中删除文档
            if not DocumentService.remove_document(doc, tenant_id):
                return get_data_error_result(
                    retmsg="Database error (Document removal)!")

            # 删除文档对应的文件关系记录
            f2d = File2DocumentService.get_by_document_id(doc_id)
            FileService.filter_delete([File.source_type == FileSource.KNOWLEDGEBASE, File.id == f2d[0].file_id])
            # 从数据库中删除文件关系记录
            File2DocumentService.delete_by_document_id(doc_id)

            # 从MinIO中删除文档对应的文件
            MINIO.rm(b, n)
        except Exception as e:
            # 记录删除过程中出现的异常
            errors += str(e)

    # 如果有错误发生，则返回错误信息
    if errors:
        return get_json_result(data=False, retmsg=errors, retcode=RetCode.SERVER_ERROR)

    # 删除成功，返回成功信息
    return get_json_result(data=True)



@manager.route('/run', methods=['POST'])
@login_required
@validate_request("doc_ids", "run")
def run():
    """
    处理文档的运行状态更新和相关操作。

    该函数接收一个POST请求，其中包含doc_ids和run参数。它主要用于更新文档的状态，
    删除搜索引擎中的文档，以及根据运行状态启动相应的任务队列。

    :return: 返回一个JSON结果，表示操作是否成功。
    """
    # 获取请求的JSON数据
    req = request.json
    try:
        # 遍历请求中指定的文档ID列表
        for id in req["doc_ids"]:
            # 构建用于更新文档状态的信息
            info = {"run": str(req["run"]), "progress": 0}
            # 如果运行状态为RUNNING，进一步设置进度信息
            if str(req["run"]) == TaskStatus.RUNNING.value:
                info["progress_msg"] = ""
                info["chunk_num"] = 0
                info["token_num"] = 0
            # 更新文档的状态
            DocumentService.update_by_id(id, info)
            # 如果运行状态为RUNNING，删除搜索引擎中的文档
            # if str(req["run"]) == TaskStatus.CANCEL.value:
            tenant_id = DocumentService.get_tenant_id(id)
            if not tenant_id:
                # 如果找不到租户ID，返回错误结果
                return get_data_error_result(retmsg="Tenant not found!")
            # 从搜索引擎中删除文档
            ELASTICSEARCH.deleteByQuery(
                Q("match", doc_id=id), idxnm=search.index_name(tenant_id))

            # 如果运行状态为RUNNING，清除相关任务并触发任务队列
            if str(req["run"]) == TaskStatus.RUNNING.value:
                # 清除相关任务
                TaskService.filter_delete([Task.doc_id == id])
                # 获取文档详情
                e, doc = DocumentService.get_by_id(id)
                doc = doc.to_dict()
                doc["tenant_id"] = tenant_id
                # 获取文档在MinIO中的存储地址
                bucket, name = File2DocumentService.get_minio_address(doc_id=doc["id"])
                # 将文档加入任务队列
                queue_tasks(doc, bucket, name)

        # 返回操作成功的JSON结果
        return get_json_result(data=True)
    except Exception as e:
        # 返回操作失败的JSON结果，包含错误信息
        return server_error_response(e)


@manager.route('/rename', methods=['POST'])
@login_required
@validate_request("doc_id", "name")
def rename():
    """
    处理文档重命名的请求。

    该接口仅支持POST方法，需要用户登录，并通过验证doc_id和name参数。
    主要逻辑包括：
    - 根据doc_id获取文档信息，判断文档是否存在。
    - 检查新名称是否与原文件后缀一致，防止文件类型被更改。
    - 检查新名称在相同知识库中是否唯一，防止重名。
    - 更新文档名称。
    - 如果文档关联了文件，同时更新文件名称。

    :return: 返回重命名操作的结果，成功或失败的原因。
    """
    # 解析请求中的JSON数据
    req = request.json
    try:
        # 根据doc_id获取文档对象，检查文档是否存在
        e, doc = DocumentService.get_by_id(req["doc_id"])
        if not e:
            return get_data_error_result(retmsg="Document not found!")

        # 检查新名称的文件后缀是否与原文件一致
        if pathlib.Path(req["name"].lower()).suffix != pathlib.Path(
                doc.name.lower()).suffix:
            return get_json_result(
                data=False,
                retmsg="The extension of file can't be changed",
                retcode=RetCode.ARGUMENT_ERROR)

        # 查询是否有同名文档存在于相同知识库中
        for d in DocumentService.query(name=req["name"], kb_id=doc.kb_id):
            if d.name == req["name"]:
                return get_data_error_result(
                    retmsg="Duplicated document name in the same knowledgebase.")

        # 更新文档名称
        if not DocumentService.update_by_id(
                req["doc_id"], {"name": req["name"]}):
            return get_data_error_result(
                retmsg="Database error (Document rename)!")

        # 如果文档关联了文件，更新文件名称
        informs = File2DocumentService.get_by_document_id(req["doc_id"])
        if informs:
            e, file = FileService.get_by_id(informs[0].file_id)
            FileService.update_by_id(file.id, {"name": req["name"]})

        # 返回重命名成功的响应
        return get_json_result(data=True)
    except Exception as e:
        # 返回服务器错误响应
        return server_error_response(e)



@manager.route('/get/<doc_id>', methods=['GET'])
# @login_required
def get(doc_id):
    """
    根据文档ID获取文档内容。

    本函数通过文档ID从数据库中检索文档，并根据文档类型返回相应的文档内容或图片。
    如果文档不存在，返回错误信息。
    如果文档存在，尝试从MinIO中获取文档的存储地址，并返回文档内容。

    参数:
    - doc_id: 文档的唯一标识符。

    返回:
    - 文档内容的HTTP响应，包含适当的Content-Type头。
    """
    try:
        # 尝试根据文档ID获取文档对象和文档内容。
        e, doc = DocumentService.get_by_id(doc_id)
        # 如果获取失败，返回文档不存在的错误信息。
        if not e:
            return get_data_error_result(retmsg="Document not found!")

        # 尝试获取文档在MinIO中的存储地址。
        b, n = File2DocumentService.get_minio_address(doc_id=doc_id)
        # 从MinIO中获取文档内容，并创建HTTP响应。
        response = flask.make_response(MINIO.get(b, n))

        # 通过正则表达式提取文档的文件扩展名。
        ext = re.search(r"\.([^.]+)$", doc.name)
        if ext:
            # 根据文档类型设置HTTP响应的Content-Type头。
            if doc.type == FileType.VISUAL.value:
                response.headers.set('Content-Type', 'image/%s' % ext.group(1))
            else:
                response.headers.set(
                    'Content-Type',
                    'application/%s' % ext.group(1))
        return response
    except Exception as e:
        # 如果发生异常，返回服务器错误的HTTP响应。
        return server_error_response(e)


@manager.route('/change_parser', methods=['POST'])
@login_required
@validate_request("doc_id", "parser_id")
def change_parser():
    """
    改变文档解析器的接口。

    该接口用于在文档服务中更改文档所使用的解析器。支持的请求方法为POST。
    请求体中需要包含doc_id和parser_id字段，用于指定要更改的文档和新的解析器。

    :return: 返回一个JSON对象，表示操作的结果。
    """
    # 获取请求的JSON数据
    req = request.json
    try:
        # 根据doc_id获取文档对象
        e, doc = DocumentService.get_by_id(req["doc_id"])
        # 如果文档不存在，返回错误信息
        if not e:
            return get_data_error_result(retmsg="Document not found!")

        # 检查当前解析器是否已经与请求的解析器相同
        if doc.parser_id.lower() == req["parser_id"].lower():
            # 如果解析器配置也相同，则直接返回成功结果
            if "parser_config" in req:
                if req["parser_config"] == doc.parser_config:
                    return get_json_result(data=True)
            else:
                return get_json_result(data=True)

        # 检查文档类型是否支持更改解析器
        if doc.type == FileType.VISUAL or re.search(
                r"\.(ppt|pptx|pages)$", doc.name):
            # 如果不支持，返回错误信息
            return get_data_error_result(retmsg="Not supported yet!")

        # 更新文档的解析器信息，并重置进度等相关字段
        e = DocumentService.update_by_id(doc.id,
                                         {"parser_id": req["parser_id"], "progress": 0, "progress_msg": "",
                                          "run": TaskStatus.UNSTART.value})
        # 如果更新失败，返回错误信息
        if not e:
            return get_data_error_result(retmsg="Document not found!")

        # 如果请求中包含parser_config，更新文档的解析器配置
        if "parser_config" in req:
            DocumentService.update_parser_config(doc.id, req["parser_config"])

        # 如果文档有令牌数，调整令牌数和块数，并更新处理持续时间
        if doc.token_num > 0:
            e = DocumentService.increment_chunk_num(doc.id, doc.kb_id, doc.token_num * -1, doc.chunk_num * -1,
                                                    doc.process_duation * -1)
            # 如果调整失败，返回错误信息
            if not e:
                return get_data_error_result(retmsg="Document not found!")
            # 获取租户ID，用于后续的Elasticsearch操作
            tenant_id = DocumentService.get_tenant_id(req["doc_id"])
            # 如果租户ID不存在，返回错误信息
            if not tenant_id:
                return get_data_error_result(retmsg="Tenant not found!")
            # 从Elasticsearch中删除文档数据
            ELASTICSEARCH.deleteByQuery(
                Q("match", doc_id=doc.id), idxnm=search.index_name(tenant_id))

        # 返回操作成功的结果
        return get_json_result(data=True)
    except Exception as e:
        # 如果发生异常，返回服务器错误信息
        return server_error_response(e)


@manager.route('/image/<image_id>', methods=['GET'])
# @login_required
def get_image(image_id):
    """
    根据图像ID获取图像资源。

    该函数通过图像ID从MinIO对象存储服务中检索图像资源，并将其作为响应返回给客户端。
    图像ID的格式是“存储桶名称-图像名称”，函数通过分割字符串来提取存储桶名称和图像名称。

    参数:
    - image_id: 图像资源的唯一标识符，包含存储桶名称和图像名称。

    返回:
    - 如果成功检索到图像资源，则返回一个包含图像数据和JPEG内容类型的响应。
    - 如果发生任何异常，则返回一个表示服务器错误的响应。
    """
    try:
        # 从图像ID中提取存储桶名称和图像名称
        bkt, nm = image_id.split("-")
        # 从MinIO服务中获取图像资源，并准备响应
        response = flask.make_response(MINIO.get(bkt, nm))
        # 设置响应的Content-Type为图像JPEG
        response.headers.set('Content-Type', 'image/JPEG')
        return response
    except Exception as e:
        # 处理任何异常，并返回服务器错误响应
        return server_error_response(e)
