import os

from huggingface_hub import snapshot_download

snapshot_download(repo_id=os.environ["MODEL_REPO"], local_dir='./model')
