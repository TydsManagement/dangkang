from .embedding_model import *
from .chat_model import *
from .cv_model import *
from .rerank_model import *
from .sequence2txt_model import *

EmbeddingModel = {
    "Ollama": OllamaEmbed,
    "LocalAI": LocalAIEmbed,
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
    "Bedrock": BedrockEmbed,
    "Gemini": GeminiEmbed,
    "NVIDIA": NvidiaEmbed,
    "LM-Studio": LmStudioEmbed
}


CvModel = {
    "OpenAI": GptV4,
    "Azure-OpenAI": AzureGptV4,
    "Ollama": OllamaCV,
    "Xinference": XinferenceCV,
    "Tongyi-Qianwen": QWenCV,
    "ZHIPU-AI": Zhipu4V,
    "Moonshot": LocalCV,
    "Gemini": GeminiCV,
    "OpenRouter": OpenRouterCV,
    "LocalAI": LocalAICV,
    "NVIDIA": NvidiaCV,
    "LM-Studio": LmStudioCV
}


ChatModel = {
    "OpenAI": GptTurbo,
    "Azure-OpenAI": AzureChat,
    "ZHIPU-AI": ZhipuChat,
    "Tongyi-Qianwen": QWenChat,
    "Ollama": OllamaChat,
    "LocalAI": LocalAIChat,
    "Xinference": XinferenceChat,
    "Moonshot": MoonshotChat,
    "DeepSeek": DeepSeekChat,
    "VolcEngine": VolcEngineChat,
    "BaiChuan": BaiChuanChat,
    "MiniMax": MiniMaxChat,
    "Minimax": MiniMaxChat,
    "Mistral": MistralChat,
    "Gemini": GeminiChat,
    "Bedrock": BedrockChat,
    "Groq": GroqChat,
    "OpenRouter": OpenRouterChat,
    "StepFun": StepFunChat,
    "NVIDIA": NvidiaChat,
    "LM-Studio": LmStudioChat
}


RerankModel = {
    "BAAI": DefaultRerank,
    "Jina": JinaRerank,
    "Youdao": YoudaoRerank,
    "Xinference": XInferenceRerank,
    "NVIDIA": NvidiaRerank,
    "LM-Studio": LmStudioRerank
}


Seq2txtModel = {
    "OpenAI": GPTSeq2txt,
    "Tongyi-Qianwen": QWenSeq2txt,
    "Ollama": OllamaSeq2txt,
    "Azure-OpenAI": AzureSeq2txt,
    "Xinference": XinferenceSeq2txt
}
