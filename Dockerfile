FROM infiniflow/ragflow-base:v2.0
USER root

# 设置工作目录
WORKDIR /ragflow

# 复制Web前端代码并安装依赖
ADD ./web ./web
RUN cd ./web && npm i --force && npm run build

# 复制其他项目文件
ADD ./api ./api
ADD ./conf ./conf
ADD ./deepdoc ./deepdoc
ADD ./rag ./rag
ADD ./agent ./agent
ADD ./graphrag ./graphrag

# 复制requirements.txt 并安装Python依赖
ADD ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 设置环境变量
ENV PYTHONPATH=/ragflow/
ENV HF_ENDPOINT=https://hf-mirror.com

# 复制Docker相关文件
ADD docker/entrypoint.sh ./entrypoint.sh
ADD docker/.env ./
RUN chmod +x ./entrypoint.sh

# 复制模型文件到指定位置
COPY models/bge-large-zh-v1.5 /root/.ragflow/bge-large-zh-v1.5
COPY models/bge-reranker-v2-m3 /root/.ragflow/bge-reranker-v2-m3
COPY models/nltk_data /root/nltk_data

# 设置默认启动命令
CMD ["./entrypoint.sh"]
