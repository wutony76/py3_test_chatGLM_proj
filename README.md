# py3_test_chatGLM_proj

測試 chatGLM

## 專案簡介

這是一個本機端的 Python 對話模型測試專案。  
目前預設使用 `TinyLlama/TinyLlama-1.1B-Chat-v1.0` 進行命令列聊天，支援多輪對話與即時串流輸出。

## 主要功能

- 自動偵測運算裝置（Apple Silicon `mps`、`cuda`、`cpu`）
- 從 Hugging Face 載入模型與 tokenizer
- 透過終端機互動聊天，輸入 `exit` 或 `quit` 可離開
- 保留對話歷史，讓模型可參考上下文持續回應

## 快速開始

1. 建立並啟用虛擬環境
2. 安裝 PyTorch 與必要套件
3. 執行 `python main_mac.py`

python -m venv selfenv
pip install --upgrade pip

# 安裝支援 macOS MPS 的官方 PyTorch（不要用 CPU-only 版）

source selfenv/bin/activate
pip uninstall -y torch torchvision torchaudio
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install transformers accelerate optimum sentencepiece protobuf

# 驗證 MPS 可用

python -c "import torch; print('mps_available=', torch.backends.mps.is_available(), 'torch', torch.**version**)"

# 模型說明

目前程式預設使用較小且回應更快的模型：TinyLlama/TinyLlama-1.1B-Chat-v1.0。  
程式會優先從專案資料夾 `models/TinyLlama-1.1B-Chat-v1.0` 載入模型，若資料夾不存在才會改用線上模型 ID。
若要改回 ChatGLM3-6B，請將 `main_mac.py` 內的 `MODEL_ID` 改為 `THUDM/chatglm3-6b`，並改回對應的對話介面。

# 下載模型（或用程式自動下載）

git lfs install
mkdir -p models
git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 models/TinyLlama-1.1B-Chat-v1.0

# 執行

python main_mac.py

<!-- pip install torch transformers accelerate optimum
pip install sentencepiece protobuf -->

Close-up portrait in ultrarealistic 4K, digital camera with 50mm lens, framing the face of a 25–35 year-old Korean woman, centered in the composition. Eye-level perspective, as if seated directly across from her. She leans slightly forward, resting her chin on slender fingers with a subtle, alluring smile, sipping a vanilla milkshake with caramel streaks inside the glass. Brown hair loosely tied in a bun, with soft strands framing her face, enhancing her feminine charm. Symmetrical V-shaped face, luminous smooth skin, soft blush on cheeks, tinted lips gently parted. Bright, deep eyes with captivating gaze, long defined lashes, and elegantly arched brows.

She wears a **sexy cut-out dress with stylish design details**, revealing subtle curves with sophisticated elegance. Around her neck, a **dinosaur-shaped diamond necklace**, sparkling with playful luxury. The shiny fast-food counter reflects light, while the blurred background reveals hints of colorful panels and artificial reflections. Overhead soft lighting highlights skin texture, fabric details, and jewelry, with minimal shadows. Color palette blends caramel gold, creamy beige, warm skin tones, and cool metallic reflections for a seductive and mesmerizing atmosphere.
