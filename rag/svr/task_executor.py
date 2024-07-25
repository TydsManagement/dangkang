# 导入模块和类，用于处理各种功能，包括日期和时间操作、数据序列化和反序列化、日志记录、文件路径处理、
# 数据库操作、多进程、异常处理、任务调度、自然语言处理等。
import datetime
import json
import logging
import os
import hashlib
import copy
import re
import sys
import time
import traceback
from functools import partial

# 导入自定义服务类和设置，用于文件到文档的映射、数据库连接、检索器配置等。
from api.db.services.file2document_service import File2DocumentService
from api.settings import retrievaler
from rag.raptor import RecursiveAbstractiveProcessing4TreeOrganizedRetrieval as Raptor
from rag.utils.minio_conn import MINIO
from api.db.db_models import close_connection
from rag.settings import database_logger, SVR_QUEUE_NAME
from rag.settings import cron_logger, DOC_MAXIMUM_SIZE

# 导入多进程处理、numpy库、Elasticsearch DSL、自定义工具函数等，用于并行处理、数值计算、
# 数据检索和处理。
from multiprocessing import Pool
import numpy as np
from elasticsearch_dsl import Q, Search
from multiprocessing.context import TimeoutError
from api.db.services.task_service import TaskService
from rag.utils.es_conn import ELASTICSEARCH
from timeit import default_timer as timer
from rag.utils import rmSpace, findMaxTm, num_tokens_from_string

# 导入自然语言处理相关的模块和工具，用于文本分析和处理。
from rag.nlp import search, rag_tokenizer
from io import BytesIO
import pandas as pd

# 导入特定应用领域的处理模块，如法律、论文、演示文稿等。
from rag.app import laws, paper, presentation, manual, qa, table, book, resume, picture, naive, one, audio

# 导入数据库相关的枚举类型、文档服务和LLM服务类，用于数据库操作和大型语言模型的处理。
from api.db import LLMType, ParserType
from api.db.services.document_service import DocumentService
from api.db.services.llm_service import LLMBundle
from api.utils.file_utils import get_project_base_directory
from rag.utils.redis_conn import REDIS_CONN


# 定义批处理大小，用于指示每次处理的文档数量
BATCH_SIZE = 64

# 定义一个工厂字典，用于根据不同的解析器类型选择相应的解析函数
FACTORY = {
    "general": naive,
    ParserType.NAIVE.value: naive,
    ParserType.PAPER.value: paper,
    ParserType.BOOK.value: book,
    ParserType.PRESENTATION.value: presentation,
    ParserType.MANUAL.value: manual,
    ParserType.LAWS.value: laws,
    ParserType.QA.value: qa,
    ParserType.TABLE.value: table,
    ParserType.RESUME.value: resume,
    ParserType.PICTURE.value: picture,
    ParserType.ONE.value: one,
    ParserType.AUDIO.value: audio
}


def set_progress(task_id, from_page=0, to_page=-1,
                 prog=None, msg="Processing..."):
    """
    设置任务进度和消息。

    该函数用于更新给定任务的进度信息。如果任务被取消，则标记进度消息为已取消。
    如果指定了页面范围，则在进度消息中包含页面范围。

    :param task_id: 任务ID，用于标识要更新进度的任务。
    :param from_page: 任务处理的起始页面，默认为0。
    :param to_page: 任务处理的结束页面，默认为-1，表示处理到任务结束。
    :param prog: 任务的进度值，None表示未提供进度信息。
    :param msg: 任务的进度消息，默认为"Processing..."。
    """
    # 检查是否提供了进度值且小于0，如果是，则修改消息为错误消息。
    if prog is not None and prog < 0:
        msg = "[ERROR]" + msg
    # 检查任务是否被取消，并根据检查结果更新消息和进度值。
    cancel = TaskService.do_cancel(task_id)
    if cancel:
        msg += " [Canceled]"
        prog = -1
    # 如果指定了页面范围，更新进度消息以包含页面范围。
    if to_page > 0:
        if msg:
            msg = f"Page({from_page + 1}~{to_page + 1}): " + msg
    # 准备进度更新的数据。
    d = {"progress_msg": msg}
    if prog is not None:
        d["progress"] = prog
    # 尝试更新任务的进度信息，如果失败，则记录错误。
    try:
        TaskService.update_progress(task_id, d)
    except Exception as e:
        cron_logger.error("set_progress:({}), {}".format(task_id, str(e)))
    # 关闭数据库连接。
    close_connection()
    # 如果任务被取消，则退出程序。
    if cancel:
        sys.exit()


def collect():
    """
    从队列中消费任务信息，并处理这些任务。

    从特定队列中获取任务信息，如果获取失败或任务为空，则返回空的DataFrame。如果任务被取消，
    则记录日志并返回空的DataFrame。否则，处理任务并返回相应的DataFrame。

    返回:
        pd.DataFrame: 包含任务数据的DataFrame，如果无任务或处理失败则返回空的DataFrame。
    """
    try:
        # 从Redis队列中消费任务
        payload = REDIS_CONN.queue_consumer(SVR_QUEUE_NAME, "rag_flow_svr_task_broker", "rag_flow_svr_task_consumer")
        if not payload:
            # 如果没有获取到任务，则休眠1秒后返回空的DataFrame
            time.sleep(1)
            return pd.DataFrame()
    except Exception as e:
        # 记录从队列中获取任务时的异常
        cron_logger.error("Get task event from queue exception:" + str(e))
        return pd.DataFrame()

    # 获取消费的任务消息
    msg = payload.get_message()
    # 确认任务消费成功
    payload.ack()
    if not msg:
        # 如果消息为空，则返回空的DataFrame
        return pd.DataFrame()

    # 检查任务是否被取消
    if TaskService.do_cancel(msg["id"]):
        # 如果任务被取消，则记录日志并返回空的DataFrame
        cron_logger.info("Task {} has been canceled.".format(msg["id"]))
        return pd.DataFrame()
    # 获取任务详细信息
    tasks = TaskService.get_tasks(msg["id"])
    # 断言任务不为空，如果为空则抛出异常
    assert tasks, "{} empty task!".format(msg["id"])
    # 将任务信息转换为DataFrame
    tasks = pd.DataFrame(tasks)
    # 如果任务类型为'raptor'，则在DataFrame中添加'task_type'列，并设置值为'raptor'
    if msg.get("type", "") == "raptor":
        tasks["task_type"] = "raptor"
    return tasks


def get_minio_binary(bucket, name):
    """
    从MinIO存储中获取二进制文件。

    :param bucket: MinIO桶的名称。
    :param name: 文件在MinIO中的名称。
    :return: 文件的二进制数据。
    """
    return MINIO.get(bucket, name)


def build(row):
    """
    根据输入行数据，处理文档，包括切分文档和存储处理结果。

    :param row: 包含文档相关信息的字典，如文档ID、页面范围等。
    :return: 处理后的文档片段列表，如果处理失败则返回空列表。
    """
    # 检查文件大小是否超过限制
    if row["size"] > DOC_MAXIMUM_SIZE:
        # 如果文件过大，则记录错误信息并返回空列表
        set_progress(row["id"], prog=-1, msg="File size exceeds( <= %dMb )" % (int(DOC_MAXIMUM_SIZE / 1024 / 1024)))
        return []

    # 设置进度更新的回调函数
    callback = partial(
        set_progress,
        row["id"],
        row["from_page"],
        row["to_page"])
    # 根据解析器ID获取相应的文档切分器
    chunker = FACTORY[row["parser_id"].lower()]
    try:
        # 记录开始时间，用于性能监控
        st = timer()
        # 从MinIO中获取文档的存储地址
        bucket, name = File2DocumentService.get_minio_address(doc_id=row["doc_id"])
        # 从MinIO中获取文档二进制数据
        binary = get_minio_binary(bucket, name)
        # 记录从MinIO获取文档数据的时间
        cron_logger.info("From minio({}) {}/{}".format(timer() - st, row["location"], row["name"]))
        # 使用文档切分器切分文档
        cks = chunker.chunk(row["name"], binary=binary, from_page=row["from_page"],
                            to_page=row["to_page"], lang=row["language"], callback=callback,
                            kb_id=row["kb_id"], parser_config=row["parser_config"], tenant_id=row["tenant_id"])
        # 记录切分文档所需的时间
        cron_logger.info("Chunkking({}) {}/{}".format(timer() - st, row["location"], row["name"]))
    except TimeoutError as e:
        # 如果发生超时错误，更新进度并记录错误日志
        callback(-1, f"Internal server error: Fetch file timeout. Could you try it again.")
        cron_logger.error("Chunkking {}/{}: Fetch file timeout.".format(row["location"], row["name"]))
        return
    except Exception as e:
        # 处理其他异常情况
        if re.search("(No such file|not found)", str(e)):
            callback(-1, "Can not find file <%s>" % row["name"])
        else:
            callback(-1, f"Internal server error: %s" % str(e).replace("'", ""))
        traceback.print_exc()
        cron_logger.error("Chunkking {}/{}: {}".format(row["location"], row["name"], str(e)))
        return

    docs = []
    # 初始化文档基本结构
    doc = {
        "doc_id": row["doc_id"],
        "kb_id": [str(row["kb_id"])]
    }
    el = 0
    for ck in cks:
        d = copy.deepcopy(doc)
        d.update(ck)
        # 为每个文档片段生成唯一ID
        md5 = hashlib.md5()
        md5.update((ck["content_with_weight"] + str(d["doc_id"])).encode("utf-8"))
        d["_id"] = md5.hexdigest()
        # 设置文档的创建时间和戳
        d["create_time"] = str(datetime.datetime.now()).replace("T", " ")[:19]
        d["create_timestamp_flt"] = datetime.datetime.now().timestamp()
        if not d.get("image"):
            docs.append(d)
            continue

        # 处理文档中的图片数据
        output_buffer = BytesIO()
        if isinstance(d["image"], bytes):
            output_buffer = BytesIO(d["image"])
        else:
            d["image"].save(output_buffer, format='JPEG')

        # 上传处理后的图片到MinIO
        st = timer()
        MINIO.put(row["kb_id"], d["_id"], output_buffer.getvalue())
        el += timer() - st
        d["img_id"] = "{}-{}".format(row["kb_id"], d["_id"])
        del d["image"]
        docs.append(d)
    # 记录上传到MinIO所需的时间
    cron_logger.info("MINIO PUT({}):{}".format(row["name"], el))

    return docs


def init_kb(row):
    """
    初始化知识库索引。

    根据给定的租户信息，检查是否已存在相应的知识库索引，如果不存在，则创建新的知识库索引。

    参数:
    - row: 字典类型，包含租户信息，其中应至少包含 "tenant_id" 键。

    返回:
    - 如果索引已存在，则不返回任何内容。
    - 如果索引不存在，则返回创建索引的结果。
    """
    # 根据租户ID获取索引名称
    idxnm = search.index_name(row["tenant_id"])

    # 检查索引是否已存在，如果存在则不进行任何操作
    if ELASTICSEARCH.indexExist(idxnm):
        return

    # 读取映射配置文件，并基于配置文件内容创建新的索引
    return ELASTICSEARCH.createIdx(idxnm, json.load(
        open(os.path.join(get_project_base_directory(), "conf", "mapping.json"), "r")))


def embedding(docs, mdl, parser_config={}, callback=None):
    """
    对文档集合进行嵌入处理。

    参数:
    docs: 文档列表，每个文档包含标题和内容。
    mdl: 嵌入模型，用于将文本转换为向量。
    parser_config: 解析配置，用于配置文档内容的预处理方式。
    callback: 回调函数，用于在处理过程中提供进度反馈。

    返回:
    tk_count: 总的词汇数量，用于统计处理过程中的词汇计数。
    """
    # 定义批处理大小
    batch_size = 32

    # 处理文档标题，移除空格并转换为向量
    tts, cnts = [rmSpace(d["title_tks"]) for d in docs if d.get("title_tks")], [
        re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", d["content_with_weight"]) for d in docs]

    # 初始化词汇计数器
    tk_count = 0

    # 检查标题和内容的数量是否一致，如果一致，则进行向量嵌入处理
    if len(tts) == len(cnts):
        tts_ = np.array([])
        # 分批处理标题，进行嵌入并合并结果
        for i in range(0, len(tts), batch_size):
            vts, c = mdl.encode(tts[i: i + batch_size])
            if len(tts_) == 0:
                tts_ = vts
            else:
                tts_ = np.concatenate((tts_, vts), axis=0)
            tk_count += c
            # 调用回调函数更新处理进度
            callback(prog=0.6 + 0.1 * (i + 1) / len(tts), msg="")
        tts = tts_

    # 对文档内容进行向量嵌入处理
    cnts_ = np.array([])
    for i in range(0, len(cnts), batch_size):
        vts, c = mdl.encode(cnts[i: i + batch_size])
        if len(cnts_) == 0:
            cnts_ = vts
        else:
            cnts_ = np.concatenate((cnts_, vts), axis=0)
        tk_count += c
        # 调用回调函数更新处理进度
        callback(prog=0.7 + 0.2 * (i + 1) / len(cnts), msg="")
    cnts = cnts_

    # 根据配置文件设置标题和内容的权重
    title_w = float(parser_config.get("filename_embd_weight", 0.1))

    # 根据标题和内容的权重合并向量
    vects = (title_w * tts + (1 - title_w) *
             cnts) if len(tts) == len(cnts) else cnts

    # 确保处理后的向量数量与原始文档数量一致
    assert len(vects) == len(docs)

    # 将向量结果存储到文档中
    for i, d in enumerate(docs):
        v = vects[i].tolist()
        d["q_%d_vec" % len(v)] = v

    # 返回总的词汇数量
    return tk_count


def run_raptor(row, chat_mdl, embd_mdl, callback=None):
    """
    使用Raptor算法处理文档。

    :param row: 包含文档ID、租户ID、解析配置等信息的字典。
    :param chat_mdl: 聊天模型，用于生成回复。
    :param embd_mdl: 嵌入模型，用于将文本编码为向量。
    :param callback: 回调函数，用于在处理过程中进行回调。
    :return: 处理后的文档列表和总词数。
    """
    # 初始化，通过编码器获取"ok"的向量表示，用于后续向量名称的生成
    vts, _ = embd_mdl.encode(["ok"])
    # 生成向量名称
    vctr_nm = "q_%d_vec"%len(vts[0])

    # 通过检索器分块加载文档内容和对应的向量表示
    chunks = []
    for d in retrievaler.chunk_list(row["doc_id"], row["tenant_id"], fields=["content_with_weight", vctr_nm]):
        chunks.append((d["content_with_weight"], np.array(d[vctr_nm])))

    # 初始化Raptor对象
    raptor = Raptor(
        row["parser_config"]["raptor"].get("max_cluster", 64),
        chat_mdl,
        embd_mdl,
        row["parser_config"]["raptor"]["prompt"],
        row["parser_config"]["raptor"]["max_token"],
        row["parser_config"]["raptor"]["threshold"]
    )
    # 记录原始块数量
    original_length = len(chunks)
    # 使用Raptor算法处理文档块
    raptor(chunks, row["parser_config"]["raptor"]["random_seed"], callback)

    # 初始化文档基础信息
    doc = {
        "doc_id": row["doc_id"],
        "kb_id": [str(row["kb_id"])],
        "docnm_kwd": row["name"],
        "title_tks": rag_tokenizer.tokenize(row["name"])
    }
    # 初始化结果列表和词数计数器
    res = []
    tk_count = 0
    # 处理Raptor处理后的文档块
    for content, vctr in chunks[original_length:]:
        # 复制基础文档信息
        d = copy.deepcopy(doc)
        # 生成文档的唯一ID
        md5 = hashlib.md5()
        md5.update((content + str(d["doc_id"])).encode("utf-8"))
        d["_id"] = md5.hexdigest()
        # 设置文档的创建时间和戳
        d["create_time"] = str(datetime.datetime.now()).replace("T", " ")[:19]
        d["create_timestamp_flt"] = datetime.datetime.now().timestamp()
        # 设置文档的向量表示和其他文本处理结果
        d[vctr_nm] = vctr.tolist()
        d["content_with_weight"] = content
        d["content_ltks"] = rag_tokenizer.tokenize(content)
        d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
        # 将处理后的文档信息添加到结果列表
        res.append(d)
        # 累加词数
        tk_count += num_tokens_from_string(content)
    # 返回处理后的文档列表和总词数
    return res, tk_count


def main():
    """
    主函数，负责执行文档处理流程。
    这包括收集文档信息、初始化模型、处理文档、嵌入文档内容和索引文档。
    如果过程中遇到错误，会记录错误并尝试恢复或终止操作。
    """
    # 收集文档数据
    rows = collect()
    # 如果没有收集到数据，则直接返回
    if len(rows) == 0:
        return

    # 遍历每条文档数据
    for _, r in rows.iterrows():
        # 设置进度回调函数
        callback = partial(set_progress, r["id"], r["from_page"], r["to_page"])
        try:
            # 初始化嵌入模型
            embd_mdl = LLMBundle(r["tenant_id"], LLMType.EMBEDDING, llm_name=r["embd_id"], lang=r["language"])
        except Exception as e:
            # 如果初始化失败，记录错误并继续处理下一条文档
            callback(-1, msg=str(e))
            cron_logger.error(str(e))
            continue

        # 判断文档处理类型
        if r.get("task_type", "") == "raptor":
            try:
                # 初始化聊天模型
                chat_mdl = LLMBundle(r["tenant_id"], LLMType.CHAT, llm_name=r["llm_id"], lang=r["language"])
                # 运行Raptor处理流程
                cks, tk_count = run_raptor(r, chat_mdl, embd_mdl, callback)
            except Exception as e:
                # 如果处理失败，记录错误并继续处理下一条文档
                callback(-1, msg=str(e))
                cron_logger.error(str(e))
                continue
        else:
            # 记录开始时间
            st = timer()
            # 构建文档块
            cks = build(r)
            # 如果构建失败或没有生成块，跳过当前文档
            cron_logger.info("Build chunks({}): {}".format(r["name"], timer() - st))
            if cks is None:
                continue
            if not cks:
                callback(1., "No chunk! Done!")
                continue
            # 设置进度并记录构建耗时
            callback(
                msg="Finished slicing files(%d). Start to embedding the content." %
                    len(cks))
            st = timer()
            try:
                # 进行内容嵌入
                tk_count = embedding(cks, embd_mdl, r["parser_config"], callback)
            except Exception as e:
                # 如果嵌入失败，记录错误并继续处理下一条文档
                callback(-1, "Embedding error:{}".format(str(e)))
                cron_logger.error(str(e))
                tk_count = 0
            # 记录嵌入耗时
            cron_logger.info("Embedding elapsed({}): {:.2f}".format(r["name"], timer() - st))
            # 设置进度并准备索引文档
            callback(msg="Finished embedding({:.2f})! Start to build index!".format(timer() - st))

        # 初始化知识库
        init_kb(r)
        # 计算唯一文档块数量
        chunk_count = len(set([c["_id"] for c in cks]))
        # 记录开始时间
        st = timer()
        # 分批索引文档块
        es_r = ""
        es_bulk_size = 16
        for b in range(0, len(cks), es_bulk_size):
            es_r = ELASTICSEARCH.bulk(cks[b:b + es_bulk_size], search.index_name(r["tenant_id"]))
            # 每128个批次更新一次进度回调
            if b % 128 == 0:
                callback(prog=0.8 + 0.1 * (b + 1) / len(cks), msg="")
        # 记录索引耗时
        cron_logger.info("Indexing elapsed({}): {:.2f}".format(r["name"], timer() - st))
        # 如果索引失败，尝试恢复或删除失败的文档
        if es_r:
            callback(-1, "Index failure!")
            ELASTICSEARCH.deleteByQuery(
                Q("match", doc_id=r["doc_id"]), idxnm=search.index_name(r["tenant_id"]))
            cron_logger.error(str(es_r))
        else:
            # 检查是否需要取消任务
            if TaskService.do_cancel(r["id"]):
                ELASTICSEARCH.deleteByQuery(
                    Q("match", doc_id=r["doc_id"]), idxnm=search.index_name(r["tenant_id"]))
                continue
            # 完成任务处理，更新文档统计信息
            callback(1., "Done!")
            DocumentService.increment_chunk_num(
                r["doc_id"], r["kb_id"], tk_count, chunk_count, 0)
            # 记录任务完成信息
            cron_logger.info(
                "Chunk doc({}), token({}), chunks({}), elapsed:{:.2f}".format(
                    r["id"], tk_count, len(cks), timer() - st))


# 当模块作为主程序运行时，执行以下代码
if __name__ == "__main__":
    # 初始化peewee日志系统，确保日志不会被多次传播
    peewee_logger = logging.getLogger('peewee')
    peewee_logger.propagate = False
    # 将数据库日志处理器添加到peewee日志系统中
    peewee_logger.addHandler(database_logger.handlers[0])
    # 设置peewee日志系统的日志级别与数据库日志系统相同
    peewee_logger.setLevel(database_logger.level)

    # 无限循环，直到程序主动退出
    while True:
        # 主程序入口
        main()
