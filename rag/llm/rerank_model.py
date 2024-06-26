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
#  limitations under the License.
#
import re
import requests
import torch
from FlagEmbedding import FlagReranker
from huggingface_hub import snapshot_download
import os
from abc import ABC
import numpy as np
from api.utils.file_utils import get_home_cache_dir
from rag.utils import num_tokens_from_string, truncate

def sigmoid(x):
    """
    计算sigmoid函数的值。

    Sigmoid函数，也称为S曲线，是一种常用的激活函数，用于将输入值映射到0到1之间的输出值。
    这种函数在神经网络、逻辑回归等模型中广泛应用，因为它能够将连续值转换为概率解释。

    参数:
    x: 输入值，可以是标量、数组或矩阵。

    返回值:
    x的sigmoid函数值，相同维度的标量、数组或矩阵。
    """
    # 使用numpy的指数函数计算sigmoid函数的值
    return 1 / (1 + np.exp(-x))


class Base(ABC):
    def __init__(self, key, model_name):
        pass

    def similarity(self, query: str, texts: list):
        raise NotImplementedError("Please implement encode method!")


class DefaultRerank:
    _model = None

    def __init__(self, key, model_name, **kwargs):
        """
        初始化重排名模型。
        If you have trouble downloading HuggingFace models, -_^ this might help!!

        For Linux:
        export HF_ENDPOINT=https://hf-mirror.com
        如果默认重排名模型未初始化，则尝试从Hugging Face模型库中加载模型。如果加载失败，
        则通过snapshot_download方法下载模型。使用FlagReranker包装模型以支持半精度浮点数
        (如果CUDA可用)。

        参数:
        key: 模型使用的密钥或标识符。
        model_name: 模型的名称，用于从Hugging Face模型库中加载或下载模型。
        **kwargs: 其他关键字参数，用于传递给FlagReranker。
        """
        # 检查是否已初始化默认重排名模型
        if not DefaultRerank._model:
            try:
                # 尝试从缓存目录加载模型，使用半精度浮点数如果CUDA可用
                self._model = FlagReranker(os.path.join(get_home_cache_dir(), re.sub(r"^[a-zA-Z]+/", "", model_name)),
                                           use_fp16=torch.cuda.is_available())
            except Exception as e:
                # 如果加载失败，从Hugging Face仓库下载模型到缓存目录
                self._model = snapshot_download(repo_id=model_name,
                                                local_dir=os.path.join(get_home_cache_dir(),
                                                                       re.sub(r"^[a-zA-Z]+/", "", model_name)),
                                                local_dir_use_symlinks=False)
                # 使用下载的模型初始化FlagReranker
                self._model = FlagReranker(os.path.join(get_home_cache_dir(), model_name),
                                           use_fp16=torch.cuda.is_available())


    def similarity(self, query: str, texts: list):
        """
        计算查询字符串与一系列文本的相似度。

        参数:
        query: str - 查询字符串。
        texts: list - 文本列表。

        返回:
        np.array, int - 一个numpy数组，包含每个文本与查询的相似度分数，以及所有文本的总令牌数。
        """
        # 生成查询字符串与每个文本的配对，并确保文本长度不超过2048个字符
        pairs = [(query, truncate(t, 2048)) for t in texts]

        # 统计所有文本中的令牌（词或词组）总数
        token_count = 0
        for _, t in pairs:
            token_count += num_tokens_from_string(t)

        # 定义批次大小为4096，用于分批处理计算相似度
        batch_size = 4096
        res = []
        # 分批计算相似度分数
        for i in range(0, len(pairs), batch_size):
            scores = self._model.compute_score(pairs[i:i + batch_size], max_length=2048)
            # 如果返回的是单个分数，则添加到结果列表中；如果是多个分数，则扩展结果列表
            if isinstance(scores, float):
                res.append(scores)
            else:
                res.extend(scores)

        # 将所有相似度分数转换为numpy数组，并返回总令牌数
        return np.array(res), token_count


class JinaRanker:
    """
    该类用于初始化Jina排名器，它使用Jina AI的rerank API来对文档进行重新排名。

    参数:
    key: str
        API访问密钥，用于授权访问Jina AI的rerank API。
    model_name: str, 可选
        模型名称，用于指定使用的排名模型，默认为"jina-reranker-v1-base-en"。
    base_url: str, 可选
        API的基础URL，默认为"https://api.jina.ai/v1/rerank"。
    """

    def __init__(self, key, model_name="jina-reranker-v1-base-en", base_url="https://api.jina.ai/v1/rerank"):
        """
        初始化JinaRanker类实例，设置API访问的基础URL、请求头和模型名称。
        """
        # 设置API的基础URL
        self.base_url = base_url
        # 设置请求头，包括内容类型和授权信息
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }
        # 设置使用的模型名称
        self.model_name = model_name


    def similarity(self, query: str, texts: list):
        """
        计算查询字符串与给定文本列表之间的相似度。

        参数:
        query: str - 用于计算相似度的查询字符串。
        texts: list - 需要与查询字符串比较的文本列表。

        返回:
        numpy.array - 包含每个文本与查询字符串的相似度得分的数组。
        int - 被处理的总令牌数。
        """
        # 对输入的文本列表进行截断，确保每个文本的长度不超过8196个字符
        texts = [truncate(t, 8196) for t in texts]

        # 构建请求数据，包含模型名称、查询字符串、处理后的文本列表和需要返回的结果数量
        data = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts)
        }

        # 发送POST请求，计算相似度，并获取响应结果
        res = requests.post(self.base_url, headers=self.headers, json=data).json()

        # 提取响应中的相似度得分和总令牌数，返回相应的数组和整数
        return np.array([d["relevance_score"] for d in res["results"]]), res["usage"]["total_tokens"]


class YoudaoRerank:
    _model = None  # 类级变量，用于存储BCE模型实例

    def __init__(self, key=None, model_name="maidalun1020/bce-reranker-base_v1", **kwargs):
        """
        初始化YoudaoRerank类的实例。

        如果类变量_model尚未初始化，则尝试加载BCE模型。加载过程首先尝试从用户缓存目录加载模型，
        如果失败，则回退到使用预定义的模型名称加载。

        参数:
        key: 用于访问模型的密钥，可选。
        model_name: 模型的名称或路径，指定用于加载的BCE模型，默认为"maidalun1020/bce-reranker-base_v1"。
        **kwargs: 其他传递给RerankerModel构造函数的参数。
        """
        from BCEmbedding import RerankerModel
        if not YoudaoRerank._model:
            try:
                print("LOADING BCE...")
                # 尝试从用户缓存目录加载模型，路径中的model_name经过处理，移除了开头的字母序列
                YoudaoRerank._model = RerankerModel(model_name_or_path=os.path.join(
                    get_home_cache_dir(),
                    re.sub(r"^[a-zA-Z]+/", "", model_name)))
            except Exception as e:
                # 加载失败时，回退到使用不同的model_name尝试加载
                YoudaoRerank._model = RerankerModel(
                    model_name_or_path=model_name.replace(
                        "maidalun1020", "InfiniFlow"))

    
    def similarity(self, query: str, texts: list):
        """
        计算查询字符串与一系列文本的相似度。

        参数:
        query: str - 查询字符串。
        texts: list - 文本列表。

        返回:
        np.array, int - 相似度分数数组和总令牌数。
        """
        # 生成查询字符串和文本的配对列表，并确保文本长度不超过模型的最大长度
        pairs = [(query, truncate(t, self._model.max_length)) for t in texts]

        # 统计所有文本中的令牌总数
        token_count = 0
        for _, t in pairs:
            token_count += num_tokens_from_string(t)

        # 定义批次大小
        batch_size = 32

        # 用于存储所有批次的相似度分数
        res = []
        # 按批次处理配对的字符串，计算相似度分数
        for i in range(0, len(pairs), batch_size):
            # 计算当前批次的相似度分数
            scores = self._model.compute_score(pairs[i:i + batch_size], max_length=self._model.max_length)
            # 应用sigmoid函数平滑分数，并转换为列表格式
            scores = sigmoid(np.array(scores)).tolist()
            # 将批次分数添加到结果列表中
            if isinstance(scores, float): res.append(scores)
            else: res.extend(scores)

        # 将所有批次的相似度分数合并成一个numpy数组，并返回总令牌数
        return np.array(res), token_count
