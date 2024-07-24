# 导入AzureOpenAI类，用于与Azure OpenAI服务进行交互
from openai.lib.azure import AzureOpenAI
# 导入ZhipuAI类，用于访问智谱AI的相关功能
from zhipuai import ZhipuAI
# 导入Generation类，用于调用达观智能的文本生成服务
from dashscope import Generation
# 导入ABC类，作为抽象基类，提供通用的接口或方法
from abc import ABC
# 导入OpenAI类和openai模块，用于与OpenAI平台进行交互
from openai import OpenAI
import openai
# 导入Client类，用于访问Ollama聊天机器人服务
from ollama import Client
# 导入MaasService类，用于访问火山引擎的机器学习即服务（MaaS）
from volcengine.maas.v2 import MaasService
# 导入is_english函数，用于判断文本是否为英文
from rag.nlp import is_english
# 导入num_tokens_from_string函数，用于计算字符串中的令牌数量
from rag.utils import num_tokens_from_string
from groq import Groq
import json
import requests

class Base(ABC):
    def __init__(self, key, model_name, base_url):
        """
        初始化 OpenAI 客户端对象并设置模型名称。

        该构造函数用于创建一个针对特定 OpenAI 模型的客户端实例。
        它允许用户通过提供的 API 密钥和模型名称来访问和使用 OpenAI 服务。

        参数:
        key (str): 用户的 OpenAI API 密钥，用于身份验证和访问服务。
        model_name (str): 指定要使用的 OpenAI 模型的名称，用于调用特定模型的功能。
        base_url (str): OpenAI 服务的基 URL，用于指定 API 调用的端点。

        返回:
        None
        """
        # 初始化 OpenAI 客户端，使用提供的 API 密钥和基础URL
        self.client = OpenAI(api_key=key, base_url=base_url)
        # 设置模型名称，用于后续调用特定模型
        self.model_name = model_name


    def chat(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                **gen_conf)
            ans = response.choices[0].message.content.strip()
            if response.choices[0].finish_reason == "length":
                ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                    [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, response.usage.total_tokens
        except openai.APIError as e:
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        ans = ""
        total_tokens = 0
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                stream=True,
                **gen_conf)
            for resp in response:
                if not resp.choices or not resp.choices[0].delta.content:continue
                ans += resp.choices[0].delta.content
                total_tokens += 1
                if resp.choices[0].finish_reason == "length":
                    ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                        [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
                yield ans

        except openai.APIError as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield total_tokens


class GptTurbo(Base):
    def __init__(self, key, model_name="gpt-3.5-turbo", base_url="https://api.openai.com/v1"):
        if not base_url: base_url="https://api.openai.com/v1"
        super().__init__(key, model_name, base_url)


class MoonshotChat(Base):
    def __init__(self, key, model_name="moonshot-v1-8k", base_url="https://api.moonshot.cn/v1"):
        if not base_url: base_url="https://api.moonshot.cn/v1"
        super().__init__(key, model_name, base_url)


class XinferenceChat(Base):
    def __init__(self, key=None, model_name="", base_url=""):
        """
        初始化函数，用于创建类的实例。

        参数:
        key (Optional[str]): API密钥，用于访问某些在线服务。默认为None。
        model_name (str): 模型名称，用于标识特定的模型。默认为空字符串。
        base_url (str): API的基础URL，用于构建请求的地址。默认为空字符串。

        注意:
        这里将`key`参数重新赋值为"xxx"，是为了隐藏真实的密钥信息，实际使用时应根据具体情况修改。
        """
        # 由于实际开发中密钥信息不应公开，这里使用"xxx"代替真实密钥
        key = "xxx"
        # 调用父类的初始化函数，传入修改后的key以及其他参数
        super().__init__(key, model_name, base_url)


class DeepSeekChat(Base):
    def __init__(self, key, model_name="deepseek-chat", base_url="https://api.deepseek.com/v1"):
        if not base_url: base_url="https://api.deepseek.com/v1"
        super().__init__(key, model_name, base_url)


class AzureChat(Base):
    def __init__(self, key, model_name, **kwargs):
        self.client = AzureOpenAI(api_key=key, azure_endpoint=kwargs["base_url"], api_version="2024-02-01")
        self.model_name = model_name


class BaiChuanChat(Base):
    def __init__(self, key, model_name="Baichuan3-Turbo", base_url="https://api.baichuan-ai.com/v1"):
        if not base_url:
            base_url = "https://api.baichuan-ai.com/v1"
        super().__init__(key, model_name, base_url)

    @staticmethod
    def _format_params(params):
        return {
            "temperature": params.get("temperature", 0.3),
            "max_tokens": params.get("max_tokens", 2048),
            "top_p": params.get("top_p", 0.85),
        }

    def chat(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                extra_body={
                    "tools": [{
                        "type": "web_search",
                        "web_search": {
                            "enable": True,
                            "search_mode": "performance_first"
                        }
                    }]
                },
                **self._format_params(gen_conf))
            ans = response.choices[0].message.content.strip()
            if response.choices[0].finish_reason == "length":
                ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                    [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, response.usage.total_tokens
        except openai.APIError as e:
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        ans = ""
        total_tokens = 0
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                extra_body={
                    "tools": [{
                        "type": "web_search",
                        "web_search": {
                            "enable": True,
                            "search_mode": "performance_first"
                        }
                    }]
                },
                stream=True,
                **self._format_params(gen_conf))
            for resp in response:
                if resp.choices[0].finish_reason == "stop":
                    if not resp.choices[0].delta.content:
                        continue
                    total_tokens = resp.usage.get('total_tokens', 0)
                if not resp.choices[0].delta.content:
                    continue
                ans += resp.choices[0].delta.content
                if resp.choices[0].finish_reason == "length":
                    ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                        [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
                yield ans

        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield total_tokens


class QWenChat(Base):
    def __init__(self, key, model_name=Generation.Models.qwen_turbo, **kwargs):
        import dashscope
        dashscope.api_key = key
        self.model_name = model_name

    def chat(self, system, history, gen_conf):
        from http import HTTPStatus
        if system:
            history.insert(0, {"role": "system", "content": system})
        response = Generation.call(
            self.model_name,
            messages=history,
            result_format='message',
            **gen_conf
        )
        ans = ""
        tk_count = 0
        if response.status_code == HTTPStatus.OK:
            ans += response.output.choices[0]['message']['content']
            tk_count += response.usage.total_tokens
            if response.output.choices[0].get("finish_reason", "") == "length":
                ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                    [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, tk_count

        return "**ERROR**: " + response.message, tk_count

    def chat_streamly(self, system, history, gen_conf):
        from http import HTTPStatus
        if system:
            history.insert(0, {"role": "system", "content": system})
        ans = ""
        tk_count = 0
        try:
            response = Generation.call(
                self.model_name,
                messages=history,
                result_format='message',
                stream=True,
                **gen_conf
            )
            for resp in response:
                if resp.status_code == HTTPStatus.OK:
                    ans = resp.output.choices[0]['message']['content']
                    tk_count = resp.usage.total_tokens
                    if resp.output.choices[0].get("finish_reason", "") == "length":
                        ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                            [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
                    yield ans
                else:
                    yield ans + "\n**ERROR**: " + resp.message if str(resp.message).find("Access")<0 else "Out of credit. Please set the API key in **settings > Model providers.**"
        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield tk_count


class ZhipuChat(Base):
    def __init__(self, key, model_name="glm-3-turbo", **kwargs):
        self.client = ZhipuAI(api_key=key)
        self.model_name = model_name

    def chat(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        try:
            if "presence_penalty" in gen_conf: del gen_conf["presence_penalty"]
            if "frequency_penalty" in gen_conf: del gen_conf["frequency_penalty"]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                **gen_conf
            )
            ans = response.choices[0].message.content.strip()
            if response.choices[0].finish_reason == "length":
                ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                    [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, response.usage.total_tokens
        except Exception as e:
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        if "presence_penalty" in gen_conf: del gen_conf["presence_penalty"]
        if "frequency_penalty" in gen_conf: del gen_conf["frequency_penalty"]
        ans = ""
        tk_count = 0
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                stream=True,
                **gen_conf
            )
            for resp in response:
                if not resp.choices[0].delta.content:continue
                delta = resp.choices[0].delta.content
                ans += delta
                if resp.choices[0].finish_reason == "length":
                    ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                        [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
                    tk_count = resp.usage.total_tokens
                if resp.choices[0].finish_reason == "stop": tk_count = resp.usage.total_tokens
                yield ans
        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield tk_count


class OllamaChat(Base):
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

    def chat(self, system, history, gen_conf):
        """
        与聊天机器人进行对话。

        参数:
        system (str): 系统消息，用于指示或影响机器人的响应。
        history (List[Dict]): 对话历史，包含之前的对话内容和角色。
        gen_conf (Dict): 生成配置，用于定制机器人的响应生成方式。

        返回:
        tuple: 包含机器人的响应和评价次数的元组。
        """
        # 如果有系统消息，将其添加到对话历史的开头
        if system:
            history.insert(0, {"role": "system", "content": system})

        try:
            # 初始化选项字典，用于传递给聊天函数
            options = {}

            # 根据生成配置，动态添加选项到options字典
            if "temperature" in gen_conf: options["temperature"] = gen_conf["temperature"]
            if "max_tokens" in gen_conf: options["num_predict"] = gen_conf["max_tokens"]
            if "top_p" in gen_conf: options["top_k"] = gen_conf["top_p"]
            if "presence_penalty" in gen_conf: options["presence_penalty"] = gen_conf["presence_penalty"]
            if "frequency_penalty" in gen_conf: options["frequency_penalty"] = gen_conf["frequency_penalty"]

            # 调用客户端的chat方法进行对话，并获取响应
            response = self.client.chat(
                model=self.model_name,
                messages=history,
                options=options,
                keep_alive=-1
            )

            # 从响应中提取答案和评价次数
            ans = response["message"]["content"].strip()
            eval_count = response["eval_count"] + response.get("prompt_eval_count", 0)

            return ans, eval_count
        except Exception as e:
            # 如果发生异常，返回错误消息和0的评价次数
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf):
        """
        通过流式调用模型进行对话。

        参数:
        system (str): 系统消息，用于初始化对话。
        history (List[Dict]): 对话历史，包含之前的对话内容。
        gen_conf (Dict): 生成配置，用于定制生成的回答。

        返回:
        Generator: 生成器，产生对话过程中的中间结果和最终结果。
        """
        # 如果有系统消息，插入到对话历史的开头
        if system:
            history.insert(0, {"role": "system", "content": system})

        # 初始化生成选项
        options = {}

        # 根据生成配置，动态设置生成选项
        if "temperature" in gen_conf: options["temperature"] = gen_conf["temperature"]
        if "max_tokens" in gen_conf: options["num_predict"] = gen_conf["max_tokens"]
        if "top_p" in gen_conf: options["top_k"] = gen_conf["top_p"]
        if "presence_penalty" in gen_conf: options["presence_penalty"] = gen_conf["presence_penalty"]
        if "frequency_penalty" in gen_conf: options["frequency_penalty"] = gen_conf["frequency_penalty"]

        # 初始化答案字符串
        ans = ""

        try:
            # 调用客户端的chat方法进行流式对话
            response = self.client.chat(
                model=self.model_name,
                messages=history,
                stream=True,
                options=options,
                keep_alive=-1
            )
            # 遍历对话响应
            for resp in response:
                # 如果当前响应表示对话结束，yield累计的评价指标
                if resp["done"]:
                    yield resp.get("prompt_eval_count", 0) + resp.get("eval_count", 0)
                # 将当前响应的消息内容追加到答案字符串
                ans += resp["message"]["content"]
                # 将当前答案yield出去
                yield ans
        except Exception as e:
            # 如果发生异常，将当前答案和错误信息yield出去
            yield ans + "\n**ERROR**: " + str(e)
        finally:
            # 对话结束，yield 0表示完成
            yield 0


class LocalAIChat(Base):
    def __init__(self, key, model_name, base_url):
        if base_url[-1] == "/":
            base_url = base_url[:-1]
        self.base_url = base_url + "/v1/chat/completions"
        self.model_name = model_name.split("___")[0]

    def chat(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        for k in list(gen_conf.keys()):
            if k not in ["temperature", "top_p", "max_tokens"]:
                del gen_conf[k]
        headers = {
            "Content-Type": "application/json",
        }
        payload = json.dumps(
            {"model": self.model_name, "messages": history, **gen_conf}
        )
        try:
            response = requests.request(
                "POST", url=self.base_url, headers=headers, data=payload
            )
            response = response.json()
            ans = response["choices"][0]["message"]["content"].strip()
            if response["choices"][0]["finish_reason"] == "length":
                ans += (
                    "...\nFor the content length reason, it stopped, continue?"
                    if is_english([ans])
                    else "······\n由于长度的原因，回答被截断了，要继续吗？"
                )
            return ans, response["usage"]["total_tokens"]
        except Exception as e:
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        ans = ""
        total_tokens = 0
        try:
            headers = {
                "Content-Type": "application/json",
            }
            payload = json.dumps(
                {
                    "model": self.model_name,
                    "messages": history,
                    "stream": True,
                    **gen_conf,
                }
            )
            response = requests.request(
                "POST",
                url=self.base_url,
                headers=headers,
                data=payload,
            )
            for resp in response.content.decode("utf-8").split("\n\n"):
                if "choices" not in resp:
                    continue
                resp = json.loads(resp[6:])
                if "delta" in resp["choices"][0]:
                    text = resp["choices"][0]["delta"]["content"]
                else:
                    continue
                ans += text
                total_tokens += 1
            yield ans

        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield total_tokens


class LocalLLM(Base):
    class RPCProxy:
        def __init__(self, host, port):
            self.host = host
            self.port = int(port)
            self.__conn()

        def __conn(self):
            from multiprocessing.connection import Client
            self._connection = Client(
                (self.host, self.port), authkey=b'infiniflow-token4kevinhu')

        def __getattr__(self, name):
            import pickle

            def do_rpc(*args, **kwargs):
                for _ in range(3):
                    try:
                        self._connection.send(
                            pickle.dumps((name, args, kwargs)))
                        return pickle.loads(self._connection.recv())
                    except Exception as e:
                        self.__conn()
                raise Exception("RPC connection lost!")

            return do_rpc

    def __init__(self, key, model_name="glm-3-turbo"):
        self.client = LocalLLM.RPCProxy("127.0.0.1", 7860)

    def chat(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        try:
            ans = self.client.chat(
                history,
                gen_conf
            )
            return ans, num_tokens_from_string(ans)
        except Exception as e:
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        token_count = 0
        answer = ""
        try:
            for ans in self.client.chat_streamly(history, gen_conf):
                answer += ans
                token_count += 1
                yield answer
        except Exception as e:
            yield answer + "\n**ERROR**: " + str(e)

        yield token_count


class VolcEngineChat(Base):
    def __init__(self, key, model_name, base_url):
        """
        Since do not want to modify the original database fields, and the VolcEngine authentication method is quite special,
        Assemble ak, sk, ep_id into api_key, store it as a dictionary type, and parse it for use
        model_name is for display only
        """
        self.client = MaasService('maas-api.ml-platform-cn-beijing.volces.com', 'cn-beijing')
        self.volc_ak = eval(key).get('volc_ak', '')
        self.volc_sk = eval(key).get('volc_sk', '')
        self.client.set_ak(self.volc_ak)
        self.client.set_sk(self.volc_sk)
        self.model_name = eval(key).get('ep_id', '')

    def chat(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        try:
            req = {
                "parameters": {
                    "min_new_tokens": gen_conf.get("min_new_tokens", 1),
                    "top_k": gen_conf.get("top_k", 0),
                    "max_prompt_tokens": gen_conf.get("max_prompt_tokens", 30000),
                    "temperature": gen_conf.get("temperature", 0.1),
                    "max_new_tokens": gen_conf.get("max_tokens", 1000),
                    "top_p": gen_conf.get("top_p", 0.3),
                },
                "messages": history
            }
            response = self.client.chat(self.model_name, req)
            ans = response.choices[0].message.content.strip()
            if response.choices[0].finish_reason == "length":
                ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                    [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, response.usage.total_tokens
        except Exception as e:
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        ans = ""
        tk_count = 0
        try:
            req = {
                "parameters": {
                    "min_new_tokens": gen_conf.get("min_new_tokens", 1),
                    "top_k": gen_conf.get("top_k", 0),
                    "max_prompt_tokens": gen_conf.get("max_prompt_tokens", 30000),
                    "temperature": gen_conf.get("temperature", 0.1),
                    "max_new_tokens": gen_conf.get("max_tokens", 1000),
                    "top_p": gen_conf.get("top_p", 0.3),
                },
                "messages": history
            }
            stream = self.client.stream_chat(self.model_name, req)
            for resp in stream:
                if not resp.choices[0].message.content:
                    continue
                ans += resp.choices[0].message.content
                if resp.choices[0].finish_reason == "stop":
                    tk_count = resp.usage.total_tokens
                yield ans

        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)
        yield tk_count


class MiniMaxChat(Base):
    def __init__(
        self,
        key,
        model_name,
        base_url="https://api.minimax.chat/v1/text/chatcompletion_v2",
    ):
        if not base_url:
            base_url = "https://api.minimax.chat/v1/text/chatcompletion_v2"
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = key

    def chat(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        for k in list(gen_conf.keys()):
            if k not in ["temperature", "top_p", "max_tokens"]:
                del gen_conf[k]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = json.dumps(
            {"model": self.model_name, "messages": history, **gen_conf}
        )
        try:
            response = requests.request(
                "POST", url=self.base_url, headers=headers, data=payload
            )
            response = response.json()
            ans = response["choices"][0]["message"]["content"].strip()
            if response["choices"][0]["finish_reason"] == "length":
                ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                    [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, response["usage"]["total_tokens"]
        except Exception as e:
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        ans = ""
        total_tokens = 0
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = json.dumps(
                {
                    "model": self.model_name,
                    "messages": history,
                    "stream": True,
                    **gen_conf,
                }
            )
            response = requests.request(
                "POST",
                url=self.base_url,
                headers=headers,
                data=payload,
            )
            for resp in response.text.split("\n\n")[:-1]:
                resp = json.loads(resp[6:])
                if "delta" in resp["choices"][0]:
                    text = resp["choices"][0]["delta"]["content"]
                else:
                    continue
                ans += text
                total_tokens += num_tokens_from_string(text)
                yield ans

        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield total_tokens


class MistralChat(Base):

    def __init__(self, key, model_name, base_url=None):
        from mistralai.client import MistralClient
        self.client = MistralClient(api_key=key)
        self.model_name = model_name

    def chat(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        for k in list(gen_conf.keys()):
            if k not in ["temperature", "top_p", "max_tokens"]:
                del gen_conf[k]
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=history,
                **gen_conf)
            ans = response.choices[0].message.content
            if response.choices[0].finish_reason == "length":
                ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                    [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, response.usage.total_tokens
        except openai.APIError as e:
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        for k in list(gen_conf.keys()):
            if k not in ["temperature", "top_p", "max_tokens"]:
                del gen_conf[k]
        ans = ""
        total_tokens = 0
        try:
            response = self.client.chat_stream(
                model=self.model_name,
                messages=history,
                **gen_conf)
            for resp in response:
                if not resp.choices or not resp.choices[0].delta.content:continue
                ans += resp.choices[0].delta.content
                total_tokens += 1
                if resp.choices[0].finish_reason == "length":
                    ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                        [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
                yield ans

        except openai.APIError as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield total_tokens


class BedrockChat(Base):

    def __init__(self, key, model_name, **kwargs):
        import boto3
        self.bedrock_ak = eval(key).get('bedrock_ak', '')
        self.bedrock_sk = eval(key).get('bedrock_sk', '')
        self.bedrock_region = eval(key).get('bedrock_region', '')
        self.model_name = model_name
        self.client = boto3.client(service_name='bedrock-runtime', region_name=self.bedrock_region,
                                   aws_access_key_id=self.bedrock_ak, aws_secret_access_key=self.bedrock_sk)

    def chat(self, system, history, gen_conf):
        from botocore.exceptions import ClientError
        if system:
            history.insert(0, {"role": "system", "content": system})
        for k in list(gen_conf.keys()):
            if k not in ["temperature", "top_p", "max_tokens"]:
                del gen_conf[k]
        if "max_tokens" in gen_conf:
            gen_conf["maxTokens"] = gen_conf["max_tokens"]
            _ = gen_conf.pop("max_tokens")
        if "top_p" in gen_conf:
            gen_conf["topP"] = gen_conf["top_p"]
            _ = gen_conf.pop("top_p")

        try:
            # Send the message to the model, using a basic inference configuration.
            response = self.client.converse(
                modelId=self.model_name,
                messages=history,
                inferenceConfig=gen_conf
            )

            # Extract and print the response text.
            ans = response["output"]["message"]["content"][0]["text"]
            return ans, num_tokens_from_string(ans)

        except (ClientError, Exception) as e:
            return f"ERROR: Can't invoke '{self.model_name}'. Reason: {e}", 0

    def chat_streamly(self, system, history, gen_conf):
        from botocore.exceptions import ClientError
        if system:
            history.insert(0, {"role": "system", "content": system})
        for k in list(gen_conf.keys()):
            if k not in ["temperature", "top_p", "max_tokens"]:
                del gen_conf[k]
        if "max_tokens" in gen_conf:
            gen_conf["maxTokens"] = gen_conf["max_tokens"]
            _ = gen_conf.pop("max_tokens")
        if "top_p" in gen_conf:
            gen_conf["topP"] = gen_conf["top_p"]
            _ = gen_conf.pop("top_p")

        if self.model_name.split('.')[0] == 'ai21':
            try:
                response = self.client.converse(
                    modelId=self.model_name,
                    messages=history,
                    inferenceConfig=gen_conf
                )
                ans = response["output"]["message"]["content"][0]["text"]
                return ans, num_tokens_from_string(ans)

            except (ClientError, Exception) as e:
                return f"ERROR: Can't invoke '{self.model_name}'. Reason: {e}", 0

        ans = ""
        try:
            # Send the message to the model, using a basic inference configuration.
            streaming_response = self.client.converse_stream(
                modelId=self.model_name,
                messages=history,
                inferenceConfig=gen_conf
            )

            # Extract and print the streamed response text in real-time.
            for resp in streaming_response["stream"]:
                if "contentBlockDelta" in resp:
                    ans += resp["contentBlockDelta"]["delta"]["text"]
                    yield ans

        except (ClientError, Exception) as e:
            yield ans + f"ERROR: Can't invoke '{self.model_name}'. Reason: {e}"

        yield num_tokens_from_string(ans)

class GeminiChat(Base):

    def __init__(self, key, model_name,base_url=None):
        from google.generativeai import client,GenerativeModel

        client.configure(api_key=key)
        _client = client.get_default_generative_client()
        self.model_name = 'models/' + model_name
        self.model = GenerativeModel(model_name=self.model_name)
        self.model._client = _client

    def chat(self,system,history,gen_conf):
        if system:
            history.insert(0, {"role": "user", "parts": system})
        if 'max_tokens' in gen_conf:
            gen_conf['max_output_tokens'] = gen_conf['max_tokens']
        for k in list(gen_conf.keys()):
            if k not in ["temperature", "top_p", "max_output_tokens"]:
                del gen_conf[k]
        for item in history:
            if 'role' in item and item['role'] == 'assistant':
                item['role'] = 'model'
            if  'content' in item :
                item['parts'] = item.pop('content')

        try:
            response = self.model.generate_content(
                history,
                generation_config=gen_conf)
            ans = response.text
            return ans, response.usage_metadata.total_token_count
        except Exception as e:
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "user", "parts": system})
        if 'max_tokens' in gen_conf:
            gen_conf['max_output_tokens'] = gen_conf['max_tokens']
        for k in list(gen_conf.keys()):
            if k not in ["temperature", "top_p", "max_output_tokens"]:
                del gen_conf[k]
        for item in history:
            if 'role' in item and item['role'] == 'assistant':
                item['role'] = 'model'
            if  'content' in item :
                item['parts'] = item.pop('content')
        ans = ""
        try:
            response = self.model.generate_content(
                history,
                generation_config=gen_conf,stream=True)
            for resp in response:
                ans += resp.text
                yield ans

        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield  response._chunks[-1].usage_metadata.total_token_count


class GroqChat:
    def __init__(self, key, model_name,base_url=''):
        self.client = Groq(api_key=key)
        self.model_name = model_name

    def chat(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        for k in list(gen_conf.keys()):
            if k not in ["temperature", "top_p", "max_tokens"]:
                del gen_conf[k]
        ans = ""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                **gen_conf
            )
            ans = response.choices[0].message.content
            if response.choices[0].finish_reason == "length":
                ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                    [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, response.usage.total_tokens
        except Exception as e:
            return ans + "\n**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        for k in list(gen_conf.keys()):
            if k not in ["temperature", "top_p", "max_tokens"]:
                del gen_conf[k]
        ans = ""
        total_tokens = 0
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                stream=True,
                **gen_conf
            )
            for resp in response:
                if not resp.choices or not resp.choices[0].delta.content:
                    continue
                ans += resp.choices[0].delta.content
                total_tokens += 1
                if resp.choices[0].finish_reason == "length":
                    ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                        [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
                yield ans

        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield total_tokens


## openrouter
class OpenRouterChat(Base):
    def __init__(self, key, model_name, base_url="https://openrouter.ai/api/v1"):
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = OpenAI(base_url=self.base_url, api_key=key)
        self.model_name = model_name

class StepFunChat(Base):
    def __init__(self, key, model_name, base_url="https://api.stepfun.com/v1"):
        if not base_url:
            base_url = "https://api.stepfun.com/v1"
        super().__init__(key, model_name, base_url)


class NvidiaChat(Base):
    def __init__(
        self,
        key,
        model_name,
        base_url="https://integrate.api.nvidia.com/v1/chat/completions",
    ):
        if not base_url:
            base_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = key
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        for k in list(gen_conf.keys()):
            if k not in ["temperature", "top_p", "max_tokens"]:
                del gen_conf[k]
        payload = {"model": self.model_name, "messages": history, **gen_conf}
        try:
            response = requests.post(
                url=self.base_url, headers=self.headers, json=payload
            )
            response = response.json()
            ans = response["choices"][0]["message"]["content"].strip()
            return ans, response["usage"]["total_tokens"]
        except Exception as e:
            return "**ERROR**: " + str(e), 0

    def chat_streamly(self, system, history, gen_conf):
        if system:
            history.insert(0, {"role": "system", "content": system})
        for k in list(gen_conf.keys()):
            if k not in ["temperature", "top_p", "max_tokens"]:
                del gen_conf[k]
        ans = ""
        total_tokens = 0
        payload = {
            "model": self.model_name,
            "messages": history,
            "stream": True,
            **gen_conf,
        }

        try:
            response = requests.post(
                url=self.base_url,
                headers=self.headers,
                json=payload,
            )
            for resp in response.text.split("\n\n"):
                if "choices" not in resp:
                    continue
                resp = json.loads(resp[6:])
                if "content" in resp["choices"][0]["delta"]:
                    text = resp["choices"][0]["delta"]["content"]
                else:
                    continue
                ans += text
                if "usage" in resp:
                    total_tokens = resp["usage"]["total_tokens"]
                yield ans

        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield total_tokens


class LmStudioChat(Base):
    def __init__(self, key, model_name, base_url):
        from os.path import join

        if not base_url:
            raise ValueError("Local llm url cannot be None")
        if base_url.split("/")[-1] != "v1":
            self.base_url = join(base_url, "v1")
        self.client = OpenAI(api_key="lm-studio", base_url=self.base_url)
        self.model_name = model_name
