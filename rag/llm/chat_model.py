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
from openai.lib.azure import AzureOpenAI
from zhipuai import ZhipuAI
from dashscope import Generation
from abc import ABC
from openai import OpenAI
import openai
from ollama import Client
from volcengine.maas.v2 import MaasService
from rag.nlp import is_english
from rag.utils import num_tokens_from_string


class Base(ABC):
    def __init__(self, key, model_name, base_url):
        self.client = OpenAI(api_key=key, base_url=base_url)
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
        key = "xxx"
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
    def __init__(self, key, model_name="abab6.5s-chat",
                 base_url="https://api.minimax.chat/v1/text/chatcompletion_v2"):
        if not base_url:
            base_url="https://api.minimax.chat/v1/text/chatcompletion_v2"
        super().__init__(key, model_name, base_url)


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
