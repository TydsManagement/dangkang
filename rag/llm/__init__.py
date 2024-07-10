from .embedding_model import *
from .chat_model import *
from .cv_model import *
from .rerank_model import *


EmbeddingModel = {
    "Ollama": OllamaEmbed,
    "OpenAI": OpenAIEmbed,
    "Azure-OpenAI": AzureEmbed,
    "Xinference": XinferenceEmbed,
    "Tongyi-Qianwen": QWenEmbed,
    "ZHIPU-AI": ZhipuEmbed,
    "FastEmbed": FastEmbed,
    "Youdao": YoudaoEmbed,
    "BaiChuan": BaiChuanEmbed,
    "Jina": JinaEmbed,
    "BAAI": DefaultEmbedding,
    "Mistral": MistralEmbed,
    "Bedrock": BedrockEmbed
}


CvModel = {
    "OpenAI": GptV4,
    "Azure-OpenAI": AzureGptV4,
    "Ollama": OllamaCV,
    "Xinference": XinferenceCV,
    "Tongyi-Qianwen": QWenCV,
    "ZHIPU-AI": Zhipu4V,
    "Moonshot": LocalCV
}


ChatModel = {
    "OpenAI": GptTurbo,
    "Azure-OpenAI": AzureChat,
    "ZHIPU-AI": ZhipuChat,
    "Tongyi-Qianwen": QWenChat,
    "Ollama": OllamaChat,
    "Xinference": XinferenceChat,
    "Moonshot": MoonshotChat,
    "DeepSeek": DeepSeekChat,
    "VolcEngine": VolcEngineChat,
    "BaiChuan": BaiChuanChat,
    "MiniMax": MiniMaxChat,
    "Mistral": MistralChat,
    "Bedrock": BedrockChat
}


RerankModel = {
    "BAAI": DefaultRerank,
    "Jina": JinaRerank,
    "Youdao": YoudaoRerank,
}
