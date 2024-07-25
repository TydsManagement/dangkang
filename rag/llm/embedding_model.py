# 导入正则表达式模块，用于后续的文本处理和模式匹配
import re
# 导入类型注解模块，用于函数参数和返回值的类型标注
from typing import Optional
# 导入线程模块，用于多线程编程
import threading
# 导入requests模块，用于发送HTTP请求
import requests
# 导入huggingface_hub库的snapshot_download函数，用于下载模型快照
from huggingface_hub import snapshot_download
# 导入AzureOpenAI类，用于与Azure OpenAI服务交互
from openai.lib.azure import AzureOpenAI
# 导入ZhipuAI类，用于与智谱AI服务交互
from zhipuai import ZhipuAI
# 导入操作系统模块，用于操作系统的接口
import os
# 导入ABC模块，用于定义抽象基类
from abc import ABC
# 导入Client类，用于与Ollama服务交互
from ollama import Client
# 导入dashscope模块，用于与DashScope服务交互
import dashscope
# 导入OpenAI模块，用于与OpenAI服务交互
from openai import OpenAI
# 导入FlagModel类，用于处理Flag Embedding模型
from FlagEmbedding import FlagModel
# 导入PyTorch模块，用于深度学习
import torch
# 导入NumPy模块，用于科学计算
import numpy as np
# 导入asyncio模块，用于异步编程
import asyncio
# 导入file_utils模块的get_home_cache_dir函数，用于获取缓存目录
from api.utils.file_utils import get_home_cache_dir
# 导入rag.utils模块的num_tokens_from_string和truncate函数，用于处理文本
from rag.utils import num_tokens_from_string, truncate
import google.generativeai as genai

class Base(ABC):
    def __init__(self, key, model_name):
        pass

    def encode(self, texts: list, batch_size=32):
        raise NotImplementedError("Please implement encode method!")

    def encode_queries(self, text: str):
        raise NotImplementedError("Please implement encode method!")


class DefaultEmbedding(Base):
    _model = None
    _model_lock = threading.Lock()
    def __init__(self, key, model_name, **kwargs):
        """
        If you have trouble downloading HuggingFace models, -_^ this might help!!

        For Linux:
        export HF_ENDPOINT=https://hf-mirror.com

        For Windows:
        Good luck
        ^_-

        """
        if not DefaultEmbedding._model:
            with DefaultEmbedding._model_lock:
                if not DefaultEmbedding._model:
                    try:
                        DefaultEmbedding._model = FlagModel(os.path.join(get_home_cache_dir(), re.sub(r"^[a-zA-Z]+/", "", model_name)),
                                                            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                                            use_fp16=torch.cuda.is_available())
                    except Exception as e:
                        model_dir = snapshot_download(repo_id="BAAI/bge-large-zh-v1.5",
                                                      local_dir=os.path.join(get_home_cache_dir(), re.sub(r"^[a-zA-Z]+/", "", model_name)),
                                                      local_dir_use_symlinks=False)
                        DefaultEmbedding._model = FlagModel(model_dir,
                                                            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                                            use_fp16=torch.cuda.is_available())
        self._model = DefaultEmbedding._model

    def encode(self, texts: list, batch_size=32):
        """
        对给定的文本列表进行编码。

        将输入的文本列表编码为数值序列，以便于进一步的处理和模型输入。此方法还计算了所有文本的总令牌数。

        参数:
        texts: 文本列表，每个元素是一个字符串。
        batch_size: 批处理大小，决定每次向模型发送多少个文本进行编码。

        返回:
        np.array: 所有文本编码的numpy数组。
        int: 所有文本的令牌总数。
        """
        # 对输入的文本进行截断，确保每个文本的长度不超过2048个字符
        texts = [truncate(t, 2048) for t in texts]

        # 初始化令牌计数器，用于累计所有文本中的令牌总数
        token_count = 0

        # 遍历处理后的文本列表，累加每个文本的令牌数
        for t in texts:
            token_count += num_tokens_from_string(t)

        # 初始化结果列表，用于存储编码后的序列
        res = []

        # 通过批处理方式对文本进行编码
        for i in range(0, len(texts), batch_size):
            # 将当前批处理的文本编码追加到结果列表中
            res.extend(self._model.encode(texts[i:i + batch_size]).tolist())

        # 将结果列表转换为numpy数组并返回
        # 同时返回所有文本的令牌总数
        return np.array(res), token_count

    def encode_queries(self, text: str):
        """
        对给定的文本进行编码。

        使用模型将文本转换为特定编码格式，同时计算文本中令牌（token）的数量。

        参数:
        text (str): 需要编码的文本。

        返回:
        tuple: 包含两个元素的元组。第一个元素是文本的编码表示，是一个列表；第二个元素是文本中令牌的数量。
        """
        # 计算文本中的令牌数量
        token_count = num_tokens_from_string(text)
        # 使用模型对文本进行编码，并将编码结果转换为列表格式
        return self._model.encode_queries([text]).tolist()[0], token_count


class OpenAIEmbed(Base):
    def __init__(self, key, model_name="text-embedding-ada-002",
                 base_url="https://api.openai.com/v1"):
        """
        初始化 OpenAI 客户端对象。

        该构造函数用于创建一个 OpenAI 客户端实例，用于与 OpenAI API 交互，特别是进行文本嵌入操作。

        参数:
            key (str): OpenAI 的 API 密钥，用于授权 API 调用。
            model_name (str, 可选): 模型的名称，指定要使用的文本嵌入模型。默认值为 "text-embedding-ada-002"。
            base_url (str, 可选): API 的基础URL。默认值为 "https://api.openai.com/v1"，除非有特殊需求，一般不需要更改。

        返回:
            None
        """
        # 检查并设置 base_url，如果未提供，则使用默认值
        if not base_url:
            base_url = "https://api.openai.com/v1"
        # 初始化 OpenAI 客户端，使用提供的 API 密钥和基础 URL
        self.client = OpenAI(api_key=key, base_url=base_url)
        # 设置模型名称，用于后续的文本嵌入操作
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        """
        对给定的文本列表进行编码。

        使用指定的模型对每个文本进行处理，截断过长的文本，并返回编码后的向量数组以及处理的总令牌数。

        参数:
        texts: list - 待编码的文本列表。
        batch_size: int - 批处理的大小，默认为32。

        返回:
        np.array - 编码后的向量数组。
        int - 处理的总令牌数。
        """
        # 对输入的文本进行截断，确保它们的长度不超过8196个字符
        texts = [truncate(t, 8196) for t in texts]

        # 调用客户端的embeddings.create方法，使用指定的模型对截断后的文本进行编码
        res = self.client.embeddings.create(input=texts,
                                            model=self.model_name)

        # 提取编码结果中的向量数据，并转换为numpy数组
        return np.array([d.embedding for d in res.data]
                        ), res.usage.total_tokens

    def encode_queries(self, text):
        """
        对给定的文本进行编码。

        使用客户端调用嵌入服务，将文本转换为向量表示。此方法限制了输入文本的长度，
        并返回编码后的向量以及使用的令牌总数。

        参数:
        text (str): 需要编码的文本。

        返回:
        np.array: 编码后的向量。
        int: 总共使用的令牌数。
        """
        # 调用客户端的embeddings方法创建文本嵌入，限制输入文本长度不超过8196个字符
        res = self.client.embeddings.create(input=[truncate(text, 8196)],
                                            model=self.model_name)
        # 返回编码后的向量和使用的令牌总数
        return np.array(res.data[0].embedding), res.usage.total_tokens


class LocalAIEmbed(Base):
    def __init__(self, key, model_name, base_url):
        self.base_url = base_url + "/embeddings"
        self.headers = {
            "Content-Type": "application/json",
        }
        self.model_name = model_name.split("___")[0]

    def encode(self, texts: list, batch_size=None):
        data = {"model": self.model_name, "input": texts, "encoding_type": "float"}
        res = requests.post(self.base_url, headers=self.headers, json=data).json()

        return np.array([d["embedding"] for d in res["data"]]), 1024

    def encode_queries(self, text):
        embds, cnt = self.encode([text])
        return np.array(embds[0]), cnt

class AzureEmbed(OpenAIEmbed):
    def __init__(self, key, model_name, **kwargs):
        # 初始化Azure OpenAI客户端，使用提供的API密钥和端点URL
        self.client = AzureOpenAI(api_key=key, azure_endpoint=kwargs["base_url"], api_version="2024-02-01")
        # 设置要使用的模型名称
        self.model_name = model_name


class BaiChuanEmbed(OpenAIEmbed):
    def __init__(self, key,
                 model_name='Baichuan-Text-Embedding',
                 base_url='https://api.baichuan-ai.com/v1'):
        """
        初始化BaichuanClient类的实例。

        参数:
        key: 用户的API密钥，用于身份验证。
        model_name: 模型名称，指定使用的文本嵌入模型，默认为'Baichuan-Text-Embedding'。
        base_url: API的基URL，默认为'https://api.baichuan-ai.com/v1'，可由用户指定以访问不同环境的API。

        返回:
        无
        """
        # 检查base_url是否为空，如果为空，则使用默认的API地址
        if not base_url:
            base_url = "https://api.baichuan-ai.com/v1"
        # 调用父类的初始化方法，传入key、model_name和base_url
        super().__init__(key, model_name, base_url)


class QWenEmbed(Base):
    def __init__(self, key, model_name="text_embedding_v2", **kwargs):
        dashscope.api_key = key
        self.model_name = model_name

    def encode(self, texts: list, batch_size=10):
        import dashscope
        try:
            res = []
            token_count = 0
            texts = [truncate(t, 2048) for t in texts]
            for i in range(0, len(texts), batch_size):
                resp = dashscope.TextEmbedding.call(
                    model=self.model_name,
                    input=texts[i:i + batch_size],
                    text_type="document"
                )
                embds = [[] for _ in range(len(resp["output"]["embeddings"]))]
                for e in resp["output"]["embeddings"]:
                    embds[e["text_index"]] = e["embedding"]
                res.extend(embds)
                token_count += resp["usage"]["total_tokens"]
            return np.array(res), token_count
        except Exception as e:
            raise Exception("Account abnormal. Please ensure it's on good standing to use QWen's "+self.model_name)
        return np.array([]), 0

    def encode_queries(self, text):
        try:
            resp = dashscope.TextEmbedding.call(
                model=self.model_name,
                input=text[:2048],
                text_type="query"
            )
            return np.array(resp["output"]["embeddings"][0]
                            ["embedding"]), resp["usage"]["total_tokens"]
        except Exception as e:
            raise Exception("Account abnormal. Please ensure it's on good standing to use QWen's "+self.model_name)
        return np.array([]), 0


class ZhipuEmbed(Base):
    def __init__(self, key, model_name="embedding-2", **kwargs):
        self.client = ZhipuAI(api_key=key)
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        arr = []
        tks_num = 0
        for txt in texts:
            res = self.client.embeddings.create(input=txt,
                                                model=self.model_name)
            arr.append(res.data[0].embedding)
            tks_num += res.usage.total_tokens
        return np.array(arr), tks_num

    def encode_queries(self, text):
        res = self.client.embeddings.create(input=text,
                                            model=self.model_name)
        return np.array(res.data[0].embedding), res.usage.total_tokens


class OllamaEmbed(Base):
    def __init__(self, key, model_name, **kwargs):
        """
                初始化模型客户端。

                参数:
                - key: API密钥。
                - model_name: 模型名称。
                - **kwargs: 额外的参数，其中应包含基础URL（base_url）。
                """
        # 初始化客户端，使用kwargs中的base_url配置
        self.client = Client(host=kwargs["base_url"])
        # 存储模型名称
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        """
        对给定的文本列表进行编码。

        使用预训练模型对每个文本进行嵌入处理，将文本转换为向量表示。

        参数:
        texts: list - 需要编码的文本列表。
        batch_size: int - 每次处理的文本数量。默认为32。

        返回:
        np.array, int - 返回一个numpy数组，其中包含所有文本的嵌入向量，以及处理过的文本总词汇数。
        """
        # 初始化一个列表，用于存储每个文本的嵌入向量
        arr = []
        # 初始化一个变量，用于累计处理过的文本的总词汇数
        tks_num = 0
        # 遍历文本列表
        for txt in texts:
            # 调用客户端的embeddings方法，对当前文本进行嵌入处理
            # 使用预训练模型名称self.model_name作为模型参数
            res = self.client.embeddings(prompt=txt,
                                         model=self.model_name)
            # 将处理得到的嵌入向量添加到列表中
            arr.append(res["embedding"])
            # 累加处理过的文本的总词汇数，这里假设每个文本的平均长度为128
            tks_num += 128
        # 将嵌入向量列表转换为numpy数组，并返回总词汇数
        return np.array(arr), tks_num

    def encode_queries(self, text):
        """
        对给定的文本进行编码查询。

        使用客户端的嵌入函数对文本进行处理，获取文本的嵌入表示。
        这里的嵌入表示是一种将文本转换为数值向量的表示方法，用于后续的计算或比较。

        参数:
        text (str): 需要进行编码查询的文本。

        返回:
        np.array: 文本的嵌入表示，是一个数值数组。
        int: 嵌入向量的维度，这里固定为128。
        """
        # 使用客户端的embeddings方法对文本进行编码
        res = self.client.embeddings(prompt=text,
                                     model=self.model_name)
        # 将获取的嵌入表示转换为numpy数组，并返回向量的维度
        return np.array(res["embedding"]), 128


class FastEmbed(Base):
    _model = None

    def __init__(
            self,
            key: Optional[str] = None,
            model_name: str = "BAAI/bge-small-en-v1.5",
            cache_dir: Optional[str] = None,
            threads: Optional[int] = None,
            **kwargs,
    ):
        from fastembed import TextEmbedding
        if not FastEmbed._model:
            self._model = TextEmbedding(model_name, cache_dir, threads, **kwargs)

    def encode(self, texts: list, batch_size=32):
        # Using the internal tokenizer to encode the texts and get the total
        # number of tokens
        encodings = self._model.model.tokenizer.encode_batch(texts)
        total_tokens = sum(len(e) for e in encodings)

        embeddings = [e.tolist() for e in self._model.embed(texts, batch_size)]

        return np.array(embeddings), total_tokens

    def encode_queries(self, text: str):
        # Using the internal tokenizer to encode the texts and get the total
        # number of tokens
        encoding = self._model.model.tokenizer.encode(text)
        embedding = next(self._model.query_embed(text)).tolist()

        return np.array(embedding), len(encoding.ids)


class XinferenceEmbed(Base):
    def __init__(self, key, model_name="", base_url=""):
        self.client = OpenAI(api_key="xxx", base_url=base_url)
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        res = self.client.embeddings.create(input=texts,
                                            model=self.model_name)
        return np.array([d.embedding for d in res.data]
                        ), res.usage.total_tokens

    def encode_queries(self, text):
        res = self.client.embeddings.create(input=[text],
                                            model=self.model_name)
        return np.array(res.data[0].embedding), res.usage.total_tokens


class YoudaoEmbed(Base):
    _client = None

    def __init__(self, key=None, model_name="maidalun1020/bce-embedding-base_v1", **kwargs):
        from BCEmbedding import EmbeddingModel as qanthing
        if not YoudaoEmbed._client:
            try:
                print("LOADING BCE...")
                YoudaoEmbed._client = qanthing(model_name_or_path=os.path.join(
                    get_home_cache_dir(),
                    "bce-embedding-base_v1"))
            except Exception as e:
                YoudaoEmbed._client = qanthing(
                    model_name_or_path=model_name.replace(
                        "maidalun1020", "InfiniFlow"))

    def encode(self, texts: list, batch_size=10):
        res = []
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        for i in range(0, len(texts), batch_size):
            embds = YoudaoEmbed._client.encode(texts[i:i + batch_size])
            res.extend(embds)
        return np.array(res), token_count

    def encode_queries(self, text):
        embds = YoudaoEmbed._client.encode([text])
        return np.array(embds[0]), num_tokens_from_string(text)


class JinaEmbed(Base):
    """
    该类初始化了一个用于与Jina AI Embedding API进行交互的客户端。

    参数:
    key: str
        访问API所需的授权令牌。
    model_name: str, 可选
        指定使用的模型名称，默认为"jina-embeddings-v2-base-zh"。
    base_url: str, 可选
        API的基URL，默认为"https://api.jina.ai/v1/embeddings"。
    """

    def __init__(self, key, model_name="jina-embeddings-v2-base-zh",
                 base_url="https://api.jina.ai/v1/embeddings"):
        """
        初始化客户端实例，设置API访问的基URL、请求头和模型名称。
        """
        # 设置API的基URL
        self.base_url = "https://api.jina.ai/v1/embeddings"
        # 初始化请求头，包含内容类型和授权令牌
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }
        # 设置使用的模型名称
        self.model_name = model_name

    def encode(self, texts: list, batch_size=None):
        """
        对给定的文本列表进行编码。

        使用预先训练的模型，将文本列表转换为对应的嵌入向量数组。
        此方法主要处理文本预处理、请求构建和响应解析。

        参数:
        texts: list - 待编码的文本列表。
        batch_size: int - 可选参数，用于指定批次大小。未使用此参数。

        返回:
        np.array, int - 返回一个numpy数组，其中包含每个文本的嵌入向量，以及总令牌数。
        """
        # 对输入的文本进行截断，确保它们的长度不超过8196个字符
        texts = [truncate(t, 8196) for t in texts]

        # 构建请求的JSON数据，指定模型名称、输入文本和编码类型
        data = {
            "model": self.model_name,
            "input": texts,
            'encoding_type': 'float'
        }

        # 发送POST请求，使用预先训练的模型对文本进行编码
        res = requests.post(self.base_url, headers=self.headers, json=data).json()

        # 解析响应，提取嵌入向量和总令牌数
        # 返回嵌入向量数组和总令牌数，用于进一步的处理或分析
        return np.array([d["embedding"] for d in res["data"]]), res["usage"]["total_tokens"]

    def encode_queries(self, text):
        """
        对查询文本进行编码。

        该方法将输入的文本转换为对应的嵌入表示，并返回这种表示形式的数组以及编码文本的数量。

        参数:
        text (str): 需要编码的查询文本。

        返回:
        np.array: 查询文本的嵌入表示数组。
        int: 编码后的文本数量。
        """
        # 使用self.encode方法将文本编码为嵌入表示和计数
        embds, cnt = self.encode([text])
        # 将嵌入表示转换为numpy数组并返回
        return np.array(embds[0]), cnt


class InfinityEmbed(Base):
    _model = None

    def __init__(
            self,
            model_names: list[str] = ("BAAI/bge-small-en-v1.5",),
            engine_kwargs: dict = {},
            key = None,
    ):
        """
        初始化InfinityEngine类的实例。

        这个类负责管理多个无限嵌入（infinity embedding）模型的异步引擎数组。它通过指定模型名称和可选的引擎参数来初始化。

        参数:
            model_names (list[str]): 模型名称的列表。默认为一个包含"BAAI/bge-small-en-v1.5"的元组。
            engine_kwargs (dict): 传递给引擎构造函数的额外参数字典。默认为空字典。
            key: 保留参数，用于未来功能。目前未使用。

        引擎参数是一个关键字参数字典，可以包含任何被无限嵌入引擎接受的参数。
        """
        # 导入EngineArgs类和AsyncEngineArray类，用于配置和创建异步引擎数组
        from infinity_emb import EngineArgs
        from infinity_emb.engine import AsyncEngineArray

        # 设置默认模型名称为模型名称列表的第一个元素
        self._default_model = model_names[0]
        # 根据模型名称列表和引擎参数创建并初始化异步引擎数组
        self.engine_array = AsyncEngineArray.from_args([EngineArgs(model_name_or_path=model_name, **engine_kwargs) for model_name in model_names])

    async def _embed(self, sentences: list[str], model_name: str = ""):
        """
        异步地将句子列表嵌入到指定的模型中。

        如果没有提供模型名称，则使用默认的模型。此方法首先检查模型是否正在运行，
        如果没有，它将启动模型。在嵌入句子后，如果模型之前没有在运行，则停止模型。

        参数:
        sentences: 待嵌入的句子列表。
        model_name: 模型的名称，可选，默认为空字符串。

        返回:
        embeddings: 句子的嵌入表示。
        usage: 模型的使用情况信息。
        """
        # 检查是否提供了模型名称，如果没有，则使用默认模型
        if not model_name:
            model_name = self._default_model
        # 从引擎数组中获取指定模型的引擎
        engine = self.engine_array[model_name]
        # 检查引擎是否正在运行
        was_already_running = engine.is_running
        # 如果引擎没有在运行，则启动它
        if not was_already_running:
            await engine.astart()
        # 使用引擎嵌入句子，并获取嵌入表示和使用情况信息
        embeddings, usage = await engine.embed(sentences=sentences)
        # 如果引擎之前没有在运行，则停止它
        if not was_already_running:
            await engine.astop()
        # 返回嵌入表示和使用情况信息
        return embeddings, usage

    def encode(self, texts: list[str], model_name: str = "") -> tuple[np.ndarray, int]:
        """
        使用指定模型将文本列表编码为数值嵌入。

        此函数异步调用 `_embed` 方法执行编码，提高效率。
        返回一个元组，其中包含编码后的嵌入向量和模型使用次数。

        参数:
        texts: 待编码的字符串列表。
        model_name: 用于编码的模型名称，默认为空字符串。

        返回:
        包含嵌入向量的numpy数组和模型使用次数的元组。
        """

        # 异步运行嵌入过程，可以提高编码过程的效率
        embeddings, usage = asyncio.run(self._embed(texts, model_name))
        # 将嵌入列表转换为numpy数组，便于后续处理，并返回模型的使用次数
        return np.array(embeddings), usage


    def encode_queries(self, text: str) -> tuple[np.ndarray, int]:
        """
        对给定的文本进行编码，使其适合模型输入。

        此方法接收一个文本字符串作为输入，使用内部定义的编码方法处理文本，
        并返回一个元组，其中包含编码后的文本（作为numpy数组）和总词数。

        参数:
            text (str): 需要编码的文本。

        返回:
            tuple[np.ndarray, int]: 包含编码后文本和总词数的元组。
                编码后的文本是一个numpy数组，总词数是一个整数。
        """
        # 使用内部的分词器对文本进行编码并获取总词数
        return self.encode([text])


class MistralEmbed(Base):
    def __init__(self, key, model_name="mistral-embed", base_url=None):
        """
        初始化MistralClient实例。

        使用提供的API密钥初始化MistralClient对象，并设置默认的模型名称。
        这允许用户在与Mistral服务的交互中指定特定的模型进行操作。

        参数:
            key (str): 用于身份验证的API密钥。
            model_name (str): 模型的名称，默认为"mistral-embed"。
            base_url (str): Mistral服务的基础URL，默认为None，表示使用默认URL。

        返回:
            None
        """
        from mistralai.client import MistralClient
        self.client = MistralClient(api_key=key)
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        """
        对给定的文本列表进行编码。

        使用指定的模型对每个文本进行嵌入处理，并返回嵌入向量数组以及处理过的总令牌数。

        参数:
        texts: list - 待编码的文本列表。
        batch_size: int - 批处理大小，默认为32。

        返回:
        numpy.array - 文本的嵌入向量数组。
        int - 处理过的总令牌数。
        """
        # 对输入的文本进行截断处理，确保每个文本的长度不超过8196个字符
        texts = [truncate(t, 8196) for t in texts]

        # 使用客户端和指定的模型对处理后的文本进行嵌入处理
        res = self.client.embeddings(input=texts,
                                            model=self.model_name)

        # 将嵌入处理的结果转换为numpy数组，并返回嵌入向量数组以及总令牌数
        return np.array([d.embedding for d in res.data]
                        ), res.usage.total_tokens

    def encode_queries(self, text):
        """
        对给定的文本进行编码。

        使用客户端调用预训练模型，将文本转换为向量表示。
        这里的主要目的是为了处理和编码查询文本，以便后续进行相似性搜索或其他自然语言处理任务。

        参数:
        text (str): 需要编码的文本。

        返回:
        np.array: 文本的向量表示。
        int: 使用的令牌总数，用于计量或后续处理。
        """
        # 调用客户端的embeddings方法对文本进行编码
        # 注意：这里对输入文本的长度进行了限制，确保不超过8196个字符
        res = self.client.embeddings(input=[truncate(text, 8196)],
                                            model=self.model_name)
        # 返回编码后的向量和使用的令牌总数
        return np.array(res.data[0].embedding), res.usage.total_tokens


class BedrockEmbed(Base):
    def __init__(self, key, model_name,
                 **kwargs):
        """
        初始化 Bedrock 模型运行时客户端。

        该构造函数用于创建一个 Bedrock 模型运行时客户端实例，它使用 AWS Boto3 库来建立与 Bedrock 服务的连接。
        这种初始化方法特别适用于需要从提供的 `key` 参数中动态获取 AWS 访问凭证和区域信息的场景。

        参数:
        key (str): 包含 Bedrock 访问密钥（AK）、Bedrock 私有密钥（SK）和 Bedrock 区域信息的字符串表达式。
        model_name (str): 模型的名称，用于标识将要交互的特定模型。
        **kwargs: 其他关键字参数，用于未来扩展。

        注意:
        - 使用 `eval` 函数从字符串表达式中动态获取密钥和区域信息是一种灵活但潜在不安全的做法。
        - 构造函数内部导入 `boto3` 是一种接受的做法，尽管在某些情况下，将导入语句放在文件顶部可能更符合常规。
        """
        import boto3

        # 从key参数中动态获取bedrock访问密钥，如果没有，则默认为空字符串
        self.bedrock_ak = eval(key).get('bedrock_ak', '')
        # 从key参数中动态获取bedrock私有密钥，如果没有，则默认为空字符串
        self.bedrock_sk = eval(key).get('bedrock_sk', '')
        # 从key参数中动态获取bedrock区域信息，如果没有，则默认为空字符串
        self.bedrock_region = eval(key).get('bedrock_region', '')

        # 模型名称的直接赋值
        self.model_name = model_name

        # 使用获取的访问密钥、私有密钥和区域信息创建bedrock-runtime客户端
        self.client = boto3.client(service_name='bedrock-runtime', region_name=self.bedrock_region,
                                   aws_access_key_id=self.bedrock_ak, aws_secret_access_key=self.bedrock_sk)

    def encode(self, texts: list, batch_size=32):
        """
        对给定的文本列表进行编码，生成文本嵌入表示。

        使用预训练的语言模型对每个文本进行处理，得到一个固定长度的嵌入向量。
        不同的模型可能需要不同的输入格式，因此根据模型名称来调整输入数据的格式。

        参数:
        texts: 文本列表，需要进行编码处理的文本。
        batch_size: 批处理大小，默认为32。用于控制每次向模型发送的文本数量。

        返回:
        embeddings: numpy数组，包含所有文本的嵌入向量。
        token_count: 整数，表示所有处理文本中的总词数（或其他令牌数）。
        """
        # 对文本进行截断，确保长度不超过8196个字符
        texts = [truncate(t, 8196) for t in texts]
        embeddings = []
        token_count = 0
        for text in texts:
            # 根据模型名称的不同，准备相应的请求体
            if self.model_name.split('.')[0] == 'amazon':
                body = {"inputText": text}
            elif self.model_name.split('.')[0] == 'cohere':
                body = {"texts": [text], "input_type": 'search_document'}

            # 调用客户端，向预训练模型发送请求，并获取响应
            response = self.client.invoke_model(modelId=self.model_name, body=json.dumps(body))
            model_response = json.loads(response["body"].read())
            # 将模型返回的嵌入向量添加到嵌入列表中
            embeddings.extend([model_response["embedding"]])
            # 累加处理过的文本中的令牌数
            token_count += num_tokens_from_string(text)

        # 将嵌入列表转换为numpy数组，并返回总令牌数
        return np.array(embeddings), token_count

    def encode_queries(self, text):
        """
        对给定的文本进行编码，生成查询的嵌入表示。

        根据模型名称的不同（amazon或cohere），文本将以不同的格式发送给模型进行处理。
        返回模型处理后的嵌入表示以及文本中令牌的数量。

        :param text: 需要编码的文本。
        :return: 文本的嵌入表示（numpy数组）和令牌数量。
        """
        # 初始化一个列表，用于存储文本的嵌入表示。
        embeddings = []
        # 计算文本中的令牌数量。
        token_count = num_tokens_from_string(text)

        # 根据模型名称的不同，准备发送给模型的请求体。
        if self.model_name.split('.')[0] == 'amazon':
            # 对于amazon模型，限制文本长度并构建请求体。
            body = {"inputText": truncate(text, 8196)}
        elif self.model_name.split('.')[0] == 'cohere':
            # 对于cohere模型，同样限制文本长度并以不同的格式构建请求体。
            body = {"texts": [truncate(text, 8196)], "input_type": 'search_query'}

        # 调用客户端的invoke_model方法，发送请求并获取响应。
        response = self.client.invoke_model(modelId=self.model_name, body=json.dumps(body))
        # 解析模型返回的响应，提取嵌入表示。
        model_response = json.loads(response["body"].read())
        # 将提取到的嵌入表示添加到embeddings列表中。
        embeddings.extend([model_response["embedding"]])

        # 返回嵌入表示和令牌数量。
        return np.array(embeddings), token_count

class GeminiEmbed(Base):
    def __init__(self, key, model_name='models/text-embedding-004',
                 **kwargs):
        genai.configure(api_key=key)
        self.model_name = 'models/' + model_name

    def encode(self, texts: list, batch_size=32):
        texts = [truncate(t, 2048) for t in texts]
        token_count = sum(num_tokens_from_string(text) for text in texts)
        result = genai.embed_content(
            model=self.model_name,
            content=texts,
            task_type="retrieval_document",
            title="Embedding of list of strings")
        return np.array(result['embedding']),token_count

    def encode_queries(self, text):
        result = genai.embed_content(
            model=self.model_name,
            content=truncate(text,2048),
            task_type="retrieval_document",
            title="Embedding of single string")
        token_count = num_tokens_from_string(text)
        return np.array(result['embedding']),token_count

class NvidiaEmbed(Base):
    def __init__(
        self, key, model_name, base_url="https://integrate.api.nvidia.com/v1/embeddings"
    ):
        if not base_url:
            base_url = "https://integrate.api.nvidia.com/v1/embeddings"
        self.api_key = key
        self.base_url = base_url
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }
        self.model_name = model_name
        if model_name == "nvidia/embed-qa-4":
            self.base_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/embeddings"
            self.model_name = "NV-Embed-QA"
        if model_name == "snowflake/arctic-embed-l":
            self.base_url = "https://ai.api.nvidia.com/v1/retrieval/snowflake/arctic-embed-l/embeddings"

    def encode(self, texts: list, batch_size=None):
        payload = {
            "input": texts,
            "input_type": "query",
            "model": self.model_name,
            "encoding_format": "float",
            "truncate": "END",
        }
        res = requests.post(self.base_url, headers=self.headers, json=payload).json()
        return (
            np.array([d["embedding"] for d in res["data"]]),
            res["usage"]["total_tokens"],
        )

    def encode_queries(self, text):
        embds, cnt = self.encode([text])
        return np.array(embds[0]), cnt


class LmStudioEmbed(Base):
    def __init__(self, key, model_name, base_url):
        if not base_url:
            raise ValueError("Local llm url cannot be None")
        if base_url.split("/")[-1] != "v1":
            self.base_url = os.path.join(base_url, "v1")
        self.client = OpenAI(api_key="lm-studio", base_url=self.base_url)
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        res = self.client.embeddings.create(input=texts, model=self.model_name)
        return (
            np.array([d.embedding for d in res.data]),
            1024,
        )  # local embedding for LmStudio donot count tokens

    def encode_queries(self, text):
        res = self.client.embeddings.create(text, model=self.model_name)
        return np.array(res.data[0].embedding), 1024
