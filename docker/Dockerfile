# syntax=docker/dockerfile:1

FROM ubuntu:22.04
ARG CHAT_MODEL_FILE
ARG EMBEDDING_MODEL_FILE
ARG PROMPT_TEMPLATE
ENV CHAT_MODEL_FILE=${CHAT_MODEL_FILE}
ENV EMBEDDING_MODEL_FILE=${EMBEDDING_MODEL_FILE}
ENV PROMPT_TEMPLATE=${PROMPT_TEMPLATE}
RUN apt-get update && apt-get install -y curl
RUN mkdir /models
COPY $CHAT_MODEL_FILE /models/$CHAT_MODEL_FILE
COPY $EMBEDDING_MODEL_FILE /models/$EMBEDDING_MODEL_FILE
RUN curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash -s -- -v 0.14.1
RUN curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
RUN curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz; tar xzf chatbot-ui.tar.gz; rm chatbot-ui.tar.gz
COPY run.sh .
RUN chmod +x run.sh

ENTRYPOINT ./run.sh $CHAT_MODEL_FILE $EMBEDDING_MODEL_FILE $PROMPT_TEMPLATE "$@"
