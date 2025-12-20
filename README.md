## 架設網站
1.下載Web資料夾
2.刪除 node_modules/
3.開CMD
### cd 到backend資料夾
cd backend
npm install
### 啟動網站
node server.js
在frontend資料夾：
4.開啟1.html

Embedding
<img width="1223" height="338" alt="image" src="https://github.com/user-attachments/assets/fb3b5abc-97f1-40e2-a9a5-a8eeb923313f" />

---

## RAG Backend
此部分用向量檢索與問答生成，使用 Python 與 GPU 加速運算。

### 環境
1. 安裝 **Python 3.11** (建議版本，以支援 GPU 加速)。
2. 下載專案後，開 CMD 或 PowerShell 進入專案根目錄。

### 1. 建立並啟動 Python 虛擬環境
```bash
# 建立環境
python -3.11 -m venv venv_gpu

# 啟動環境 (Windows)
.\venv_gpu\Scripts\Activate

```

### 2. 安裝依賴套件 (看人)

由於需要支援 RTX 3070 顯卡加速，請務必**依照順序**執行以下指令：

**A. 安裝 GPU 版 PyTorch(每人不一樣)**

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

```

**B. 安裝其餘 RAG 套件**

```bash
pip install langchain langchain-community langchain-huggingface scikit-learn tqdm sentence-transformers numpy

```

*(內有 requirements.txt 也可以執行 `pip install -r requirements.txt`)*

### 3. 下載核心資料庫 (Google Drive)

由於向量資料庫檔案過大 (1.75 GB)，無法上傳 GitHub，請至雲端下載：

👉 **[https://drive.google.com/file/d/1MB_P0-vx0uXMpf2abzKqLrtzXmmb34RU/view?usp=drive_link]**

**請下載以下 3 個檔案，並放在專案根目錄 (與 `RAG_jack.py` 同層)：**

* `medical_rag_store.pkl` (向量資料庫)
* `健康001_QA_繁體.json` (原始資料)
* `醫學問答001_QA_繁體.json` (原始資料)

### 4. 啟動

確保虛擬環境已啟動 `(venv_gpu)`，執行：

```bash
python RAG_jack.py

```
順利的話
* 系統會自動讀取 `.pkl` 檔 (約 30 秒)，顯示 `✅ 系統就緒！` 後即可開始輸入問題測試。
