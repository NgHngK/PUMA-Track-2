FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TORCH_HOME=/opt/algorithm/models \
    PUMA_ROOT=/opt/algorithm

RUN groupadd -r user && useradd -m --no-log-init -r -g user user \
    && mkdir -p /opt/algorithm /input /output \
    && chown -R user:user /opt/algorithm /input /output \
    && chmod -R 777 /output

WORKDIR /opt/algorithm
COPY pyproject.toml ./
COPY docker/constraints.txt ./constraints.txt
COPY src/ ./src/
RUN pip install --no-cache-dir -c constraints.txt -e . \
    && python -c "import torch, timm; assert torch.__version__.startswith('2.4.1'), torch.__version__; print('torch', torch.__version__, 'timm', timm.__version__)"
COPY docker/deployment_config.json ./deployment_config.json
COPY models/ ./models/
USER user
ENTRYPOINT ["python", "-m", "puma_pipeline.submission.grand_challenge"]
