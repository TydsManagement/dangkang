# 导入正则表达式模块，用于后续的字符串匹配操作
import re
# 导入JSON模块，用于处理JSON格式的数据
import json
# 导入时间模块，用于处理时间相关功能
import time
# 导入复制模块，用于深拷贝对象
import copy

# 导入Elasticsearch客户端库
import elasticsearch
# 从elastic_transport模块导入ConnectionTimeout异常，用于处理连接超时错误
from elastic_transport import ConnectionTimeout
# 从Elasticsearch模块导入Elasticsearch类，用于创建Elasticsearch客户端实例
from elasticsearch import Elasticsearch
# 从elasticsearch_dsl模块导入UpdateByQuery、Search和Index类，用于DSL查询和更新操作
from elasticsearch_dsl import UpdateByQuery, Search, Index
# 从rag.settings模块导入es_logger，用于记录日志
from rag.settings import es_logger
# 导入rag的设置模块
from rag import settings
# 导入singleton装饰器，用于实现单例模式
from rag.utils import singleton

# 记录Elasticsearch库的版本信息
es_logger.info("Elasticsearch version: "+str(elasticsearch.__version__))


@singleton
class ESConnection:
    def __init__(self):
        self.info = {}
        self.conn()
        self.idxnm = settings.ES.get("index_name", "")
        if not self.es.ping():
            raise Exception("Can't connect to ES cluster")

    def conn(self):
        for _ in range(10):
            try:
                self.es = Elasticsearch(
                    settings.ES["hosts"].split(","),
                    basic_auth=(settings.ES["username"], settings.ES["password"]) if "username" in settings.ES and "password" in settings.ES else None,
                    verify_certs=False,
                    timeout=600
                )
                if self.es:
                    self.info = self.es.info()
                    es_logger.info("Connect to es.")
                    break
            except Exception as e:
                es_logger.error("Fail to connect to es: " + str(e))
                time.sleep(1)

    def version(self):
        v = self.info.get("version", {"number": "5.6"})
        v = v["number"].split(".")[0]
        return int(v) >= 7

    def health(self):
        return dict(self.es.cluster.health())

    def upsert(self, df, idxnm=""):
        """
        尝试更新或插入给定DataFrame中的文档到Elasticsearch。

        :param df: 包含要更新或插入的文档数据的DataFrame。
        :param idxnm: 可选参数，指定要操作的Elasticsearch索引名称。
        :return: 如果所有文档都成功更新/插入，则返回True；否则返回False。
        """
        res = []
        for d in df:
            # 提取文档ID，并从文档数据中删除ID字段，因为ID字段在Elasticsearch中用于唯一标识文档。
            id = d["id"]
            del d["id"]
            # 将文档数据包装为Elasticsearch更新操作所需的格式。
            d = {"doc": d, "doc_as_upsert": "true"}
            T = False
            for _ in range(10):
                try:
                    # 根据Elasticsearch实例的版本，选择合适的更新方法。
                    if not self.version():
                        r = self.es.update(
                            index=(
                                self.idxnm if not idxnm else idxnm),
                            body=d,
                            id=id,
                            doc_type="doc",
                            refresh=True,
                            retry_on_conflict=100)
                    else:
                        r = self.es.update(
                            index=(
                                self.idxnm if not idxnm else idxnm),
                            body=d,
                            id=id,
                            refresh=True,
                            retry_on_conflict=100)
                    # 记录成功更新的日志。
                    es_logger.info("Successfully upsert: %s" % id)
                    T = True
                    break
                except Exception as e:
                    # 记录更新失败的日志，并根据错误类型决定是否重试。
                    es_logger.warning("Fail to index: " +
                                      json.dumps(d, ensure_ascii=False) + str(e))
                    if re.search(r"(Timeout|time out)", str(e), re.IGNORECASE):
                        time.sleep(3)
                        continue
                    self.conn()
                    T = False

            if not T:
                # 将未成功更新/插入的文档添加到结果列表中，并记录错误日志。
                res.append(d)
                es_logger.error(
                    "Fail to index: " +
                    re.sub(
                        "[\r\n]",
                        "",
                        json.dumps(
                            d,
                            ensure_ascii=False)))
                d["id"] = id
                d["_index"] = self.idxnm

        # 如果所有文档都成功更新/插入，则返回True，否则返回False。
        if not res:
            return True
        return False

    def bulk(self, df, idx_nm=None):
        """
        批量处理文档更新或插入。

        :param df: 包含待处理文档的数据框，每个文档应包含"id"字段。
        :param idx_nm: 可选参数，指定索引名称，如果未提供，则使用默认索引名称。
        :return: 一个列表，包含处理过程中遇到的错误信息。
        """
        # 初始化用于存储文档id和文档内容的字典，以及用于存储批量操作动作的列表
        ids, acts = {}, []

        # 遍历数据框中的每个文档
        for d in df:
            # 优先使用"id"字段，如果不存在则使用"_id"字段作为文档的唯一标识
            id = d["id"] if "id" in d else d["_id"]
            # 深拷贝文档内容，并添加索引名称
            ids[id] = copy.deepcopy(d)
            ids[id]["_index"] = self.idxnm if not idx_nm else idx_nm
            # 删除文档中的"id"和"_id"字段，因为这些字段在ES中由其他方式指定
            if "id" in d:
                del d["id"]
            if "_id" in d:
                del d["_id"]
            # 构建更新文档的动作，包括文档的"id"和索引名称
            acts.append(
                {"update": {"_id": id, "_index": ids[id]["_index"]}, "retry_on_conflict": 100})
            # 构建文档内容，设置为更新操作时若文档不存在则插入
            acts.append({"doc": d, "doc_as_upsert": "true"})

        # 初始化用于存储错误信息的列表
        res = []
        # 尝试进行批量操作，最多尝试100次
        for _ in range(100):
            try:
                # 根据Elasticsearch版本选择合适的批量操作方法
                if elasticsearch.__version__[0] < 8:
                    r = self.es.bulk(
                        index=(
                            self.idxnm if not idx_nm else idx_nm),
                        body=acts,
                        refresh=False,
                        timeout="600s")
                else:
                    r = self.es.bulk(index=(self.idxnm if not idx_nm else
                                            idx_nm), operations=acts,
                                     refresh=False, timeout="600s")
                # 如果操作中有错误，则返回已处理的错误信息
                if re.search(r"False", str(r["errors"]), re.IGNORECASE):
                    return res

                # 遍历操作结果，收集遇到的错误信息
                for it in r["items"]:
                    if "error" in it["update"]:
                        res.append(str(it["update"]["_id"]) +
                                   ":" + str(it["update"]["error"]))

                # 如果没有错误，返回空列表表示所有操作成功
                return res
            except Exception as e:
                # 记录异常信息
                es_logger.warn("Fail to bulk: " + str(e))
                # 如果是超时异常，则等待3秒后继续尝试
                if re.search(r"(Timeout|time out)", str(e), re.IGNORECASE):
                    time.sleep(3)
                    continue
                # 如果是其他异常，则重新连接Elasticsearch后继续尝试
                self.conn()

        # 如果所有尝试都失败，返回错误信息列表
        return res

    def bulk4script(self, df):
        """
        批量处理文档更新。

        通过给定的DataFrame，批量更新Elasticsearch中的文档。首先，将文档的id和原始内容映射起来，
        并构建一个包含更新操作和脚本的列表。然后，尝试多次向Elasticsearch发送批量更新请求，
        如果遇到错误或超时，将等待一段时间后重试。

        参数:
        df: 包含待更新文档的DataFrame，每个文档应包含"id"和"script"字段。

        返回:
        一个列表，包含在批量更新过程中出现错误的文档id。
        """
        # 初始化用于存储文档id和原始内容的字典，以及包含更新操作的列表
        ids, acts = {}, []

        # 遍历DataFrame中的每个文档，准备更新操作
        for d in df:
            id = d["id"]
            ids[id] = copy.deepcopy(d["raw"])
            acts.append({"update": {"_id": id, "_index": self.idxnm}})
            acts.append(d["script"])
            es_logger.info("bulk upsert: %s" % id)

        # 初始化结果列表，用于存储出现错误的文档id
        res = []

        # 尝试多次发送批量更新请求
        for _ in range(10):
            try:
                # 根据Elasticsearch版本决定是否需要指定doc_type
                if not self.version():
                    r = self.es.bulk(
                        index=self.idxnm,
                        body=acts,
                        refresh=False,
                        timeout="600s",
                        doc_type="doc")
                else:
                    r = self.es.bulk(
                        index=self.idxnm,
                        body=acts,
                        refresh=False,
                        timeout="600s")

                # 如果请求中有错误，返回已处理的文档id
                if re.search(r"False", str(r["errors"]), re.IGNORECASE):
                    return res

                # 遍历响应中的每个项，如果更新操作出错，将文档id添加到结果列表
                for it in r["items"]:
                    if "error" in it["update"]:
                        res.append(str(it["update"]["_id"]))

                # 如果所有文档都更新成功，返回空列表
                return res
            except Exception as e:
                # 记录批量更新失败的日志，并根据错误类型决定是否重试
                es_logger.warning("Fail to bulk: " + str(e))
                if re.search(r"(Timeout|time out)", str(e), re.IGNORECASE):
                    time.sleep(3)
                    continue
                self.conn()

        # 如果多次尝试后仍有错误，返回结果列表
        return res

    def rm(self, d):
        """
        尝试删除指定文档。

        :param d: 包含文档ID的字典。
        :return: 删除操作是否成功的布尔值。
        """
        # 尝试10次删除文档
        for _ in range(10):
            try:
                # 根据ES版本决定使用哪种方式删除文档
                if not self.version():
                    # 删除操作针对的是旧版本的Elasticsearch
                    r = self.es.delete(
                        index=self.idxnm,
                        id=d["id"],
                        doc_type="doc",
                        refresh=True)
                else:
                    # 删除操作针对的是新版本的Elasticsearch
                    r = self.es.delete(
                        index=self.idxnm,
                        id=d["id"],
                        refresh=True,
                        doc_type="_doc")

                # 记录删除操作的日志
                es_logger.info("Remove %s" % d["id"])
                return True
            except Exception as e:
                # 记录删除失败的日志
                es_logger.warn("Fail to delete: " + str(d) + str(e))
                # 遇到超时错误时，休眠3秒后继续尝试
                if re.search(r"(Timeout|time out)", str(e), re.IGNORECASE):
                    time.sleep(3)
                    continue
                # 如果文档不存在，则认为删除操作成功
                if re.search(r"(not_found)", str(e), re.IGNORECASE):
                    return True
                # 如果遇到其他错误，重新连接Elasticsearch后继续尝试
                self.conn()

        # 如果10次尝试都失败，则记录错误日志并返回False
        es_logger.error("Fail to delete: " + str(d))

        return False

    def search(self, q, idxnm=None, src=False, timeout="2s"):
        """
        对 Elasticsearch 进行查询搜索。

        参数:
        q -- 查询字符串或查询字典。
        idxnm -- 指定的索引名称，可选。
        src -- 是否返回_source字段，可选。
        timeout -- 查询超时时间，可选。

        返回:
        Elasticsearch 查询结果。

        异常:
        当查询超时或出现其他错误时，抛出异常。
        """
        # 检查查询参数类型，如果不是字典，则构造查询字典
        if not isinstance(q, dict):
            q = Search().query(q).to_dict()

        # 尝试查询三次，以应对临时的错误或超时
        for i in range(3):
            try:
                # 执行 Elasticsearch 查询
                res = self.es.search(index=(self.idxnm if not idxnm else idxnm),
                                     body=q,
                                     timeout=timeout,
                                     # search_type="dfs_query_then_fetch",
                                     track_total_hits=True,
                                     _source=src)
                # 检查查询是否超时，如果超时，则抛出异常
                if str(res.get("timed_out", "")).lower() == "true":
                    raise Exception("Es Timeout.")
                # 如果查询成功，返回结果
                return res
            except Exception as e:
                # 记录查询异常日志
                es_logger.error(
                    "ES search exception: " +
                    str(e) +
                    "【Q】：" +
                    str(q))
                # 如果异常是超时导致的，则重试查询
                if str(e).find("Timeout") > 0:
                    continue
                # 如果是其他异常，抛出异常
                raise e
        # 如果三次查询都失败，记录错误日志并抛出异常
        es_logger.error("ES search timeout for 3 times!")
        raise Exception("ES search timeout.")

    def sql(self, sql, fetch_size=128, format="json", timeout="2s"):
        """
        执行SQL查询并返回结果。

        本函数尝试多次执行SQL查询，以应对可能的连接超时问题。它首先尝试执行查询，
        如果失败，则根据异常类型进行处理。对于连接超时异常，它将记录错误并尝试重新查询；
        对于其他异常，它将直接抛出。

        参数:
        sql (str): 要执行的SQL查询语句。
        fetch_size (int): 每次查询返回的记录数，默认为128。
        format (str): 返回结果的格式，默认为"json"。
        timeout (str): 请求超时时间，默认为"2s"。

        返回:
        查询结果，格式由format参数指定。

        抛出:
        ConnectionTimeout: 如果查询在多次尝试后仍超时。
        """
        # 尝试多次执行查询，以应对可能的超时问题
        for i in range(3):
            try:
                # 执行SQL查询，指定查询体、结果格式和超时时间
                res = self.es.sql.query(body={"query": sql, "fetch_size": fetch_size}, format=format, request_timeout=timeout)
                return res
            except ConnectionTimeout as e:
                # 记录连接超时异常，并尝试再次查询
                es_logger.error("Timeout【Q】：" + sql)
                continue
            except Exception as e:
                # 对于其他异常，直接抛出
                raise e
        # 如果多次尝试均失败，记录错误并抛出连接超时异常
        es_logger.error("ES search timeout for 3 times!")
        raise ConnectionTimeout()

    def get(self, doc_id, idxnm=None):
        """
        从Elasticsearch中获取文档。

        尝试多次从Elasticsearch索引中获取指定ID的文档。如果请求超时或遇到其他异常，则重试，
        直到成功获取文档或达到最大重试次数。

        参数:
        - doc_id: 需要获取的文档的ID。
        - idxnm: 可选参数，指定文档所在的索引名称。如果不提供，则使用类实例中预设的索引名称。

        返回:
        - Elasticsearch响应对象，包含获取的文档数据。

        抛出:
        - Exception: 如果所有重试都失败，或者文档获取超时，则抛出异常。
        """
        # 尝试多次获取文档，以应对临时的Elasticsearch服务不可用或超时的情况
        for i in range(3):
            try:
                # 根据提供的索引名称和文档ID，尝试获取文档
                res = self.es.get(index=(self.idxnm if not idxnm else idxnm),
                                     id=doc_id)
                # 检查Elasticsearch响应是否表明请求超时
                if str(res.get("timed_out", "")).lower() == "true":
                    raise Exception("Es Timeout.")
                return res
            except Exception as e:
                # 记录获取文档过程中遇到的异常
                es_logger.error(
                    "ES get exception: " +
                    str(e) +
                    "【Q】：" +
                    doc_id)
                # 如果异常是由于超时引起，尝试再次获取文档
                if str(e).find("Timeout") > 0:
                    continue
                raise e
        # 如果所有重试都失败，则记录错误并抛出异常
        es_logger.error("ES search timeout for 3 times!")
        raise Exception("ES search timeout.")

    def updateByQuery(self, q, d):
        """
        根据查询条件更新索引中的文档。

        :param q: 查询条件，用于筛选需要更新的文档。
        :param d: 更新文档的内容，是一个键值对的字典，键是字段名，值是新的字段值。
        :return: 如果更新成功返回True，否则返回False。
        """
        # 初始化UpdateByQuery请求，指定索引和查询条件
        ubq = UpdateByQuery(index=self.idxnm).using(self.es).query(q)

        # 构建脚本，用于更新文档的字段值
        scripts = ""
        for k, v in d.items():
            scripts += "ctx._source.%s = params.%s;" % (str(k), str(k))

        # 设置脚本内容和参数，并指定更新时不立即刷新索引
        ubq = ubq.script(source=scripts, params=d)
        ubq = ubq.params(refresh=False)

        # 设置更新操作的切片数，用于并行处理
        ubq = ubq.params(slices=5)

        # 设置处理版本冲突的方式为继续执行
        ubq = ubq.params(conflicts="proceed")

        # 尝试执行更新操作，最多尝试3次
        for i in range(3):
            try:
                r = ubq.execute()
                return True
            except Exception as e:
                # 记录更新操作异常信息
                es_logger.error("ES updateByQuery exception: " +
                                str(e) + "【Q】：" + str(q.to_dict()))
                # 如果异常是超时或版本冲突，则继续尝试
                if str(e).find("Timeout") > 0 or str(e).find("Conflict") > 0:
                    continue
                # 如果异常不是超时或版本冲突，则重新连接ES
                self.conn()

        return False

    def updateScriptByQuery(self, q, scripts, idxnm=None):
        """
        使用查询语句更新索引中的文档脚本。

        :param q: 查询语句，用于指定需要更新的文档。
        :param scripts: 脚本内容，指定更新操作的具体逻辑。
        :param idxnm: 可选参数，指定需要更新的索引名称。如果未提供，则使用类实例中预设的索引名称。
        :return: 如果更新操作成功执行，则返回True；如果尝试多次仍无法成功执行，则返回False。
        """
        # 根据提供的索引名称（如果有的话）和查询语句构造UpdateByQuery对象
        ubq = UpdateByQuery(
            index=self.idxnm if not idxnm else idxnm).using(
            self.es).query(q)

        # 设置更新操作的脚本源
        ubq = ubq.script(source=scripts)

        # 设置参数，要求更新操作立即刷新，以使更改可见
        ubq = ubq.params(refresh=True)

        # 设置并行处理切片的数量，以提高更新效率
        ubq = ubq.params(slices=5)

        # 设置处理版本冲突的策略，允许操作继续，即使存在版本冲突
        ubq = ubq.params(conflicts="proceed")

        # 尝试多次执行更新操作
        for i in range(3):
            try:
                # 执行更新操作
                r = ubq.execute()
                # 如果操作成功，返回True
                return True
            except Exception as e:
                # 记录更新操作异常信息
                es_logger.error("ES updateByQuery exception: " +
                                str(e) + "【Q】：" + str(q.to_dict()))
                # 如果异常信息包含"Timeout"或"Conflict"，则继续尝试
                if str(e).find("Timeout") > 0 or str(e).find("Conflict") > 0:
                    continue
                # 如果异常不是由于超时或版本冲突引起，尝试重新连接ES
                self.conn()

        # 如果多次尝试后仍无法成功执行更新操作，返回False
        return False

    def deleteByQuery(self, query, idxnm=""):
        """
        根据查询条件删除 Elasticsearch 中的文档。

        :param query: 查询条件，用于指定要删除的文档。
        :param idxnm: 可选参数，指定要操作的索引名称。如果不提供，则使用默认索引。
        :return: 如果删除成功，则返回 True；如果尝试3次后仍无法删除，则返回 False。
        """
        # 尝试3次删除操作，以应对可能的临时错误或冲突。
        for i in range(3):
            try:
                # 执行删除操作。如果idxnm未提供，则使用默认索引。
                r = self.es.delete_by_query(
                    index=idxnm if idxnm else self.idxnm,
                    refresh=True,  # 立即刷新索引，使删除操作可见。
                    body=Search().query(query).to_dict())
                return True  # 删除成功，返回True。
            except Exception as e:
                # 记录删除操作失败的错误日志。
                es_logger.error("ES updateByQuery deleteByQuery: " +
                                str(e) + "【Q】：" + str(query.to_dict()))
                # 如果错误是 NotFoundError，则认为删除操作成功，尽管实际上没有找到文档。
                if str(e).find("NotFoundError") > 0:
                    return True
                # 如果错误是 Timeout 或 Conflict，则重试删除操作。
                if str(e).find("Timeout") > 0 or str(e).find("Conflict") > 0:
                    continue
        return False  # 3次尝试后仍无法删除，返回False。

    def update(self, id, script, routing=None):
        """
        尝试多次更新Elasticsearch中的文档。

        参数:
        id -- 文档的ID。
        script -- 更新文档的脚本。
        routing -- 可选参数，用于指定文档的路由值。

        返回:
        如果更新成功，返回True；如果所有尝试都失败，返回False。
        """
        # 尝试更新文档三次
        for i in range(3):
            try:
                # 检查Elasticsearch版本，根据不同版本使用不同的API调用方式
                if not self.version():
                    # 对于旧版本的Elasticsearch，使用doc类型进行更新
                    r = self.es.update(
                        index=self.idxnm,
                        id=id,
                        body=json.dumps(
                            script,
                            ensure_ascii=False),
                        doc_type="doc",
                        routing=routing,
                        refresh=False)
                else:
                    # 对于新版本的Elasticsearch，使用_doc类型进行更新
                    r = self.es.update(index=self.idxnm, id=id, body=json.dumps(script, ensure_ascii=False),
                                       routing=routing, refresh=False)  # , doc_type="_doc")
                # 更新成功，返回True
                return True
            except Exception as e:
                # 记录更新过程中的异常信息
                es_logger.error(
                    "ES update exception: " + str(e) + " id：" + str(id) + ", version:" + str(self.version()) +
                    json.dumps(script, ensure_ascii=False))
                # 如果异常是由于超时引起，继续尝试更新
                if str(e).find("Timeout") > 0:
                    continue

        # 所有尝试都失败，返回False
        return False

    def indexExist(self, idxnm):
        """
        检查指定的索引是否存在于Elasticsearch中。

        参数:
        idxnm: str - 要检查的索引名称。如果未提供，则使用类级别的默认索引名称。

        返回:
        bool - 如果索引存在，则返回True；如果经过多次尝试后索引仍不存在，则返回False。
        """
        # 根据传入的索引名称或默认索引名称初始化索引对象
        s = Index(idxnm if idxnm else self.idxnm, self.es)

        # 尝试多次检查索引是否存在
        for i in range(3):
            try:
                # 检查索引是否存在并立即返回结果
                return s.exists()
            except Exception as e:
                # 记录异常信息，特别关注超时和冲突异常
                es_logger.error("ES updateByQuery indexExist: " + str(e))
                # 如果异常是超时或冲突，则继续尝试，否则返回False
                if str(e).find("Timeout") > 0 or str(e).find("Conflict") > 0:
                    continue

        # 经过多次尝试后，如果索引仍不存在，则返回False
        return False

    def docExist(self, docid, idxnm=None):
        """
        检查文档是否存在于Elasticsearch中。

        尝试多次查询文档是否存在，如果多次查询失败则返回False。
        这个方法处理了Elasticsearch查询可能遇到的超时和冲突异常。

        参数:
        docid (str): 要查询的文档ID。
        idxnm (str, 可选): 指定的索引名称。如果未指定，则使用类实例中的默认索引名称。

        返回:
        bool: 如果文档存在，则返回True；否则返回False。
        """
        # 尝试多次查询文档是否存在
        for i in range(3):
            try:
                # 根据提供的索引名称和文档ID检查文档是否存在
                return self.es.exists(index=(idxnm if idxnm else self.idxnm),
                                      id=docid)
            except Exception as e:
                # 记录查询过程中遇到的异常
                es_logger.error("ES Doc Exist: " + str(e))
                # 如果异常信息包含"Timeout"或"Conflict"，则继续尝试
                if str(e).find("Timeout") > 0 or str(e).find("Conflict") > 0:
                    continue
        # 如果多次尝试都失败，则返回False
        return False

    def createIdx(self, idxnm, mapping):
        """
        根据给定的索引名称和映射创建Elasticsearch索引。

        如果Elasticsearch版本低于8，则使用旧的方法创建索引；
        如果版本等于或高于8，则使用新方法创建索引，需要分别指定settings和mappings。

        参数:
        idxnm (str): 索引的名称。
        mapping (dict): 索引的映射定义，包括settings和mappings。

        返回:
        创建索引的响应。

        异常:
        如果创建索引时发生错误，将记录错误日志。
        """
        try:
            # 根据Elasticsearch版本选择创建索引的方法
            if elasticsearch.__version__[0] < 8:
                # Elasticsearch版本低于8时，直接创建索引
                return self.es.indices.create(idxnm, body=mapping)
            else:
                # Elasticsearch版本等于或高于8时，使用IndicesClient创建索引
                from elasticsearch.client import IndicesClient
                return IndicesClient(self.es).create(index=idxnm,
                                                     settings=mapping["settings"],
                                                     mappings=mapping["mappings"])
        except Exception as e:
            # 记录创建索引时的异常信息
            es_logger.error("ES create index error %s ----%s" % (idxnm, str(e)))

    def deleteIdx(self, idxnm):
        """
        删除Elasticsearch中的索引。

        此方法尝试删除指定的Elasticsearch索引。如果索引不存在，它将允许这种行为而不抛出错误。
        如果删除操作成功，它将返回相关的操作结果。如果出现异常，它将记录错误日志。

        参数:
        idxnm (str): 需要删除的索引的名称。

        返回:
        删除操作的结果，通常是一个包含操作细节的字典。

        异常:
        可能会抛出各种异常，具体取决于Elasticsearch客户端库的实现和Elasticsearch服务的状态。
        """
        try:
            # 尝试删除指定的索引，allow_no_indices=True允许索引不存在时不抛出错误
            return self.es.indices.delete(idxnm, allow_no_indices=True)
        except Exception as e:
            # 记录删除索引时发生的任何异常
            es_logger.error("ES delete index error %s ----%s" % (idxnm, str(e)))

    def getTotal(self, res):
        """
        获取搜索结果的总数量。

        由于Elasticsearch的不同版本在返回搜索结果时，“total”字段的结构可能有所不同，
        本函数旨在兼容处理这些不同结构，以统一的方式返回搜索结果的总数。

        参数:
        res (dict): 搜索结果的字典表示，其中包含“hits”字段，后者又包含“total”字段。

        返回:
        int 或 dict: 如果“total”字段是一个包含“value”键的字典，则返回“value”的值（int类型）；
                     否则，直接返回“total”字段的值（dict类型）。
        """
        # 检查“total”字段是否是一个字典，如果是，则返回其“value”值
        if isinstance(res["hits"]["total"], type({})):
            return res["hits"]["total"]["value"]
        # 如果“total”字段不是字典，直接返回该字段的值
        return res["hits"]["total"]

    def getDocIds(self, res):
        """
        获取文档ID列表。

        从搜索结果中提取并返回所有文档的ID列表。

        参数:
        res (dict): 包含搜索结果的字典，其中"hits"字段包含了搜索到的文档信息。

        返回:
        list: 包含所有文档ID的列表。
        """
        # 通过列表推导式从搜索结果中提取文档ID
        return [d["_id"] for d in res["hits"]["hits"]]

    def getSource(self, res):
        """
        从搜索结果中提取源数据。

        该方法遍历搜索结果中的每一项，将项的_id和_score添加到_source中，
        然后将修改后的_source添加到结果列表中。这样做是为了方便后续处理，
        将必要的字段集中到_source中，避免在处理数据时多次查找。

        参数:
        res (dict): 搜索引擎返回的搜索结果，包含hits字段，其中hits又包含hits列表，
                    列表中的每一项都是一个搜索结果项。

        返回:
        list: 包含所有搜索结果项的_source字段的列表，每个_source都已添加了_id和_score字段。
        """
        # 初始化一个空列表，用于存储处理后的源数据
        rr = []

        # 遍历搜索结果中的每一项
        for d in res["hits"]["hits"]:
            # 将_id和_score字段添加到_source中
            d["_source"]["id"] = d["_id"]
            d["_source"]["_score"] = d["_score"]
            # 将修改后的_source添加到结果列表中
            rr.append(d["_source"])

        # 返回处理后的源数据列表
        return rr

    def scrollIter(self, pagesize=100, scroll_time='2m', q={
        "query": {"match_all": {}}, "sort": [{"updated_at": {"order": "desc"}}]}):
        """
        使用Elasticsearch滚动API遍历索引中的所有文档。

        参数:
        pagesize: 每次滚动请求返回的文档数量，默认为100。
        scroll_time: 滚动上下文的持续时间，默认为'2m'（2分钟）。
        q: 搜索的查询字典，默认匹配所有文档并按'updated_at'降序排序。

        函数返回的生成器将分批输出文档。
        """
        # 尝试初始化滚动搜索，处理异常并在必要时重试。
        for _ in range(100):
            try:
                page = self.es.search(
                    index=self.idxnm,
                    scroll=scroll_time,
                    size=pagesize,
                    body=q,
                    _source=None
                )
                break
            except Exception as e:
                es_logger.error("ES scrolling fail. " + str(e))
                time.sleep(3)

        # 获取滚动ID和将要返回的总文档数。
        sid = page['_scroll_id']
        scroll_size = page['hits']['total']["value"]
        es_logger.info("[TOTAL]%d" % scroll_size)
        # 开始滚动并分批输出文档，直到没有更多文档为止。
        # 开始滚动
        while scroll_size > 0:
            yield page["hits"]["hits"]
            # 尝试继续滚动搜索，处理异常并在必要时重试。
            for _ in range(100):
                try:
                    page = self.es.scroll(scroll_id=sid, scroll=scroll_time)
                    break
                except Exception as e:
                    es_logger.error("ES scrolling fail. " + str(e))
                    time.sleep(3)

            # 更新滚动ID以供下一次滚动请求使用。
            sid = page['_scroll_id']
            # 获取上一次滚动返回的文档数量。
            scroll_size = len(page['hits']['hits'])


ELASTICSEARCH = ESConnection()
