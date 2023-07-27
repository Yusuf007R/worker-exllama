FROM runpod/pytorch:3.10-2.0.0-117

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN mkdir data
WORKDIR /data

# Install Python dependencies (Worker Template)
RUN pip install --upgrade pip && \
    pip install safetensors==0.3.1 sentencepiece huggingface_hub git+https://github.com/winglian/runpod-python.git@streaming_job_dev
RUN git clone https://github.com/turboderp/exllama
RUN pip install -r exllama/requirements.txt

ENV PYTHONPATH=/data/exllama
ENV MODEL_REPO="TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
ENV PROMPT_PREFIX=""
ENV PROMPT_SUFFIX=""
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"

COPY download_model.py /data/download_model.py
RUN python /data/download_model.py && \
    rm /data/download_model.py

COPY handler.py /data/handler.py
COPY __init.py__ /data/__init__.py



CMD [ "python", "-m", "handler" ]