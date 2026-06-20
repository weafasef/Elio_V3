import os
from pathlib import Path
from huggingface_hub import snapshot_download

# 如果在国内遇到网络问题，可以在代码中设置环境变量启用镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model_id = "huihui-ai/Llama-3.2-1B-Instruct-abliterated"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
save_path = str(PROJECT_ROOT / "models" / "Llama-3.2-1B-Instruct-abliterated")

print(f"开始下载 {model_id} 到 {save_path}...")

# snapshot_download 会自动下载整个仓库
snapshot_download(
    repo_id=model_id,
    local_dir=save_path,
    max_workers=4  # 使用多线程加速下载
)

print("下载完成！")