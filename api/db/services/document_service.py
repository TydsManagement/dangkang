import random
from datetime import datetime
from elasticsearch_dsl import Q
from peewee import fn

from api.db.db_utils import bulk_insert_into_db
from api.settings import stat_logger
from api.utils import current_timestamp, get_format_time, get_uuid
from rag.settings import SVR_QUEUE_NAME
from rag.utils.es_conn import ELASTICSEARCH
from rag.utils.minio_conn import MINIO
from rag.nlp import search

from api.db import FileType, TaskStatus
from api.db.db_models import DB, Knowledgebase, Tenant, Task
from api.db.db_models import Document
from api.db.services.common_service import CommonService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db import StatusEnum
from rag.utils.redis_conn import REDIS_CONN


class DocumentService(CommonService):
    model = Document

    @classmethod
    @DB.connection_context()
    def get_by_kb_id(cls, kb_id, page_number, items_per_page,
                     orderby, desc, keywords):
        if keywords:
            docs = cls.model.select().where(
                (cls.model.kb_id == kb_id),
                (fn.LOWER(cls.model.name).contains(keywords.lower()))
            )
        else:
            docs = cls.model.select().where(cls.model.kb_id == kb_id)
        count = docs.count()
        if desc:
            docs = docs.order_by(cls.model.getter_by(orderby).desc())
        else:
            docs = docs.order_by(cls.model.getter_by(orderby).asc())

        docs = docs.paginate(page_number, items_per_page)

        return list(docs.dicts()), count

    @classmethod
    @DB.connection_context()
    def list_documents_in_dataset(cls, dataset_id, offset, count, order_by, descend, keywords):
        if keywords:
            docs = cls.model.select().where(
                (cls.model.kb_id == dataset_id),
                (fn.LOWER(cls.model.name).contains(keywords.lower()))
            )
        else:
            docs = cls.model.select().where(cls.model.kb_id == dataset_id)

        total = docs.count()

        if descend == 'True':
            docs = docs.order_by(cls.model.getter_by(order_by).desc())
        if descend == 'False':
            docs = docs.order_by(cls.model.getter_by(order_by).asc())

        docs = list(docs.dicts())
        docs_length = len(docs)

        if offset < 0 or offset > docs_length:
            raise IndexError("Offset is out of the valid range.")

        if count == -1:
            return docs[offset:], total

        return docs[offset:offset + count], total

    @classmethod
    @DB.connection_context()
    def insert(cls, doc):
        if not cls.save(**doc):
            raise RuntimeError("Database error (Document)!")
        e, doc = cls.get_by_id(doc["id"])
        if not e:
            raise RuntimeError("Database error (Document retrieval)!")
        e, kb = KnowledgebaseService.get_by_id(doc.kb_id)
        if not KnowledgebaseService.update_by_id(
                kb.id, {"doc_num": kb.doc_num + 1}):
            raise RuntimeError("Database error (Knowledgebase)!")
        return doc

    @classmethod
    @DB.connection_context()
    def remove_document(cls, doc, tenant_id):
        ELASTICSEARCH.deleteByQuery(
                Q("match", doc_id=doc.id), idxnm=search.index_name(tenant_id))
        cls.clear_chunk_num(doc.id)
        return cls.delete_by_id(doc.id)

    @classmethod
    @DB.connection_context()
    def get_newly_uploaded(cls):
        fields = [
            cls.model.id,
            cls.model.kb_id,
            cls.model.parser_id,
            cls.model.parser_config,
            cls.model.name,
            cls.model.type,
            cls.model.location,
            cls.model.size,
            Knowledgebase.tenant_id,
            Tenant.embd_id,
            Tenant.img2txt_id,
            Tenant.asr_id,
            cls.model.update_time]
        docs = cls.model.select(*fields) \
            .join(Knowledgebase, on=(cls.model.kb_id == Knowledgebase.id)) \
            .join(Tenant, on=(Knowledgebase.tenant_id == Tenant.id))\
            .where(
                cls.model.status == StatusEnum.VALID.value,
                ~(cls.model.type == FileType.VIRTUAL.value),
                cls.model.progress == 0,
                cls.model.update_time >= current_timestamp() - 1000 * 600,
                cls.model.run == TaskStatus.RUNNING.value)\
            .order_by(cls.model.update_time.asc())
        return list(docs.dicts())

    @classmethod
    @DB.connection_context()
    def get_unfinished_docs(cls):
        fields = [cls.model.id, cls.model.process_begin_at, cls.model.parser_config, cls.model.progress_msg, cls.model.run]
        docs = cls.model.select(*fields) \
            .where(
                cls.model.status == StatusEnum.VALID.value,
                ~(cls.model.type == FileType.VIRTUAL.value),
                cls.model.progress < 1,
                cls.model.progress > 0)
        return list(docs.dicts())

    @classmethod
    @DB.connection_context()
    def increment_chunk_num(cls, doc_id, kb_id, token_num, chunk_num, duation):
        num = cls.model.update(token_num=cls.model.token_num + token_num,
                               chunk_num=cls.model.chunk_num + chunk_num,
                               process_duation=cls.model.process_duation + duation).where(
            cls.model.id == doc_id).execute()
        if num == 0:
            raise LookupError(
                "Document not found which is supposed to be there")
        num = Knowledgebase.update(
            token_num=Knowledgebase.token_num +
            token_num,
            chunk_num=Knowledgebase.chunk_num +
            chunk_num).where(
            Knowledgebase.id == kb_id).execute()
        return num

    @classmethod
    @DB.connection_context()
    def decrement_chunk_num(cls, doc_id, kb_id, token_num, chunk_num, duation):
        num = cls.model.update(token_num=cls.model.token_num - token_num,
                               chunk_num=cls.model.chunk_num - chunk_num,
                               process_duation=cls.model.process_duation + duation).where(
            cls.model.id == doc_id).execute()
        if num == 0:
            raise LookupError(
                "Document not found which is supposed to be there")
        num = Knowledgebase.update(
            token_num=Knowledgebase.token_num -
            token_num,
            chunk_num=Knowledgebase.chunk_num -
            chunk_num
        ).where(
            Knowledgebase.id == kb_id).execute()
        return num

    @classmethod
    @DB.connection_context()
    def clear_chunk_num(cls, doc_id):
        doc = cls.model.get_by_id(doc_id)
        assert doc, "Can't fine document in database."

        num = Knowledgebase.update(
            token_num=Knowledgebase.token_num -
            doc.token_num,
            chunk_num=Knowledgebase.chunk_num -
            doc.chunk_num,
            doc_num=Knowledgebase.doc_num-1
        ).where(
            Knowledgebase.id == doc.kb_id).execute()
        return num

    @classmethod
    @DB.connection_context()
    def get_tenant_id(cls, doc_id):
        docs = cls.model.select(
            Knowledgebase.tenant_id).join(
            Knowledgebase, on=(
                Knowledgebase.id == cls.model.kb_id)).where(
                cls.model.id == doc_id, Knowledgebase.status == StatusEnum.VALID.value)
        docs = docs.dicts()
        if not docs:
            return
        return docs[0]["tenant_id"]

    @classmethod
    @DB.connection_context()
    def get_tenant_id_by_name(cls, name):
        docs = cls.model.select(
            Knowledgebase.tenant_id).join(
            Knowledgebase, on=(
                    Knowledgebase.id == cls.model.kb_id)).where(
            cls.model.name == name, Knowledgebase.status == StatusEnum.VALID.value)
        docs = docs.dicts()
        if not docs:
            return
        return docs[0]["tenant_id"]

    @classmethod
    @DB.connection_context()
    def get_embd_id(cls, doc_id):
        docs = cls.model.select(
            Knowledgebase.embd_id).join(
            Knowledgebase, on=(
                Knowledgebase.id == cls.model.kb_id)).where(
                cls.model.id == doc_id, Knowledgebase.status == StatusEnum.VALID.value)
        docs = docs.dicts()
        if not docs:
            return
        return docs[0]["embd_id"]
    
    @classmethod
    @DB.connection_context()
    def get_doc_id_by_doc_name(cls, doc_name):
        fields = [cls.model.id]
        doc_id = cls.model.select(*fields) \
            .where(cls.model.name == doc_name)
        doc_id = doc_id.dicts()
        if not doc_id:
            return
        return doc_id[0]["id"]

    @classmethod
    @DB.connection_context()
    def get_thumbnails(cls, docids):
        fields = [cls.model.id, cls.model.thumbnail]
        return list(cls.model.select(
            *fields).where(cls.model.id.in_(docids)).dicts())

    @classmethod
    @DB.connection_context()
    def update_parser_config(cls, id, config):
        e, d = cls.get_by_id(id)
        if not e:
            raise LookupError(f"Document({id}) not found.")

        def dfs_update(old, new):
            for k, v in new.items():
                if k not in old:
                    old[k] = v
                    continue
                if isinstance(v, dict):
                    assert isinstance(old[k], dict)
                    dfs_update(old[k], v)
                else:
                    old[k] = v
        dfs_update(d.parser_config, config)
        cls.update_by_id(id, {"parser_config": d.parser_config})

    @classmethod
    @DB.connection_context()
    def get_doc_count(cls, tenant_id):
        docs = cls.model.select(cls.model.id).join(Knowledgebase,
                                                   on=(Knowledgebase.id == cls.model.kb_id)).where(
            Knowledgebase.tenant_id == tenant_id)
        return len(docs)

    @classmethod
    @DB.connection_context()
    def begin2parse(cls, docid):
        cls.update_by_id(
            docid, {"progress": random.random() * 1 / 100.,
                    "progress_msg": "Task dispatched...",
                    "process_begin_at": get_format_time()
                    })

    @classmethod
    @DB.connection_context()
    def update_progress(cls):
        docs = cls.get_unfinished_docs()
        for d in docs:
            try:
                tsks = Task.query(doc_id=d["id"], order_by=Task.create_time)
                if not tsks:
                    continue
                msg = []
                prg = 0
                finished = True
                bad = 0
                e, doc = DocumentService.get_by_id(d["id"])
                status = doc.run    # TaskStatus.RUNNING.value
                for t in tsks:
                    if 0 <= t.progress < 1:
                        finished = False
                    prg += t.progress if t.progress >= 0 else 0
                    msg.append(t.progress_msg)
                    if t.progress == -1:
                        bad += 1
                prg /= len(tsks)
                if finished and bad:
                    prg = -1
                    status = TaskStatus.FAIL.value
                elif finished:
                    if d["parser_config"].get("raptor", {}).get("use_raptor") and d["progress_msg"].lower().find(" raptor")<0:
                        queue_raptor_tasks(d)
                        prg *= 0.98
                        msg.append("------ RAPTOR -------")
                    else:
                        status = TaskStatus.DONE.value

                msg = "\n".join(msg)
                info = {
                    "process_duation": datetime.timestamp(
                        datetime.now()) -
                                       d["process_begin_at"].timestamp(),
                    "run": status}
                if prg != 0:
                    info["progress"] = prg
                if msg:
                    info["progress_msg"] = msg
                cls.update_by_id(d["id"], info)
            except Exception as e:
                stat_logger.error("fetch task exception:" + str(e))

    @classmethod
    @DB.connection_context()
    def get_kb_doc_count(cls, kb_id):
        return len(cls.model.select(cls.model.id).where(
            cls.model.kb_id == kb_id).dicts())


    @classmethod
    @DB.connection_context()
    def do_cancel(cls, doc_id):
        try:
            _, doc = DocumentService.get_by_id(doc_id)
            return doc.run == TaskStatus.CANCEL.value or doc.progress < 0
        except Exception as e:
            pass
        return False


def queue_raptor_tasks(doc):
    def new_task():
        nonlocal doc
        return {
            "id": get_uuid(),
            "doc_id": doc["id"],
            "from_page": 0,
            "to_page": -1,
            "progress_msg": "Start to do RAPTOR (Recursive Abstractive Processing For Tree-Organized Retrieval)."
        }

    task = new_task()
    bulk_insert_into_db(Task, [task], True)
    task["type"] = "raptor"
    assert REDIS_CONN.queue_product(SVR_QUEUE_NAME, message=task), "Can't access Redis. Please check the Redis' status."
