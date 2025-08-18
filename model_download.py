from huggingface_hub import snapshot_download
from pathlab import Path
import os

Local_model_path= Path("./model")
Local_model_path.mkdir(exist_ok=True)

allow_patterns=["*.json", "*.pt", "*.bin", "*.txt", "*.model", "*.safetensors"]

model_download_path = snapshot_download(
        repo_id = model_id,
        cache_dir = Local_model_path,
        allow_patterns=allow_patterns
)