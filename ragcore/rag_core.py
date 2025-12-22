import json
import os
import pickle
import torch
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors 
from langchain_huggingface import HuggingFaceEmbeddings

# --- ⚙️ 設定區 ---
FILE_PATHS = ["health.json", "medical.json"] 
VECTOR_STORE_PATH = "medical_rag_store_rtx5070.pkl"
BATCH_SIZE = 512 

# --- 搜尋引擎核心類別 ---
class MedicalSearchEngine:
    def __init__(self, texts, metadatas, embeddings, embedding_func):
        self.texts = texts
        self.metadatas = metadatas
        self.embedding_func = embedding_func
        self.embeddings_np = np.array(embeddings)
        
        # 建立 KNN
        self.knn = NearestNeighbors(n_neighbors=10, metric='cosine', n_jobs=-1)
        self.knn.fit(self.embeddings_np)

    def search(self, query, k=5):
        query_emb = self.embedding_func.embed_query(query)
        query_emb_np = np.array([query_emb])
        dists, indices = self.knn.kneighbors(query_emb_np, n_neighbors=k)
        
        results = []
        for dist, idx in zip(dists[0], indices[0]):
            results.append({
                "doc": self.metadatas[idx], 
                "score": 1 - dist
            })
        return results

def load_and_embed_files(file_paths):
    # 初始化 HuggingFace Embedding (這裡只是為了 embed_documents 用)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    temp_embedding_model = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={'device': device},
        encode_kwargs={'batch_size': BATCH_SIZE, 'normalize_embeddings': False}
    )

    all_texts = []
    all_metadatas = []
    
    print(f"📂 開始讀取檔案...")
    for file_path in file_paths:
        if not os.path.exists(file_path): 
            print(f"⚠️ 找不到 {file_path}，跳過")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"  - {file_path}: 讀入 {len(data)} 筆")
            
            for item in data:
                q = item.get('question', '').strip()
                a = item.get('answer', '').strip()
                if not q: continue
                combined_text = f"問題: {q}\n答案: {a}"
                all_texts.append(combined_text)
                all_metadatas.append({"q": q, "a": a, "source": file_path})

    print(f"⚡ 啟動 Embedding 計算...")
    embeddings = temp_embedding_model.embed_documents(all_texts)
    return all_texts, all_metadatas, embeddings

# ==========================================
# 🔥 關鍵修改：這就是您缺少的函式
# ==========================================
def initialize_rag_system():
    """
    初始化 RAG 系統並回傳 SearchEngine 物件
    """
    print("🔄 [rag_core] 初始化系統中...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 載入模型
    print(f"🔄 [rag_core] 載入 Embedding 模型 ({device})...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={'device': device},
        encode_kwargs={'batch_size': BATCH_SIZE, 'normalize_embeddings': False}
    )

    # 2. 嘗試讀取 Pickle
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"💾 [rag_core] 讀取索引檔: {VECTOR_STORE_PATH}")
        with open(VECTOR_STORE_PATH, "rb") as f:
            saved_data = pickle.load(f)
        
        search_engine = MedicalSearchEngine(
            saved_data['texts'], 
            saved_data['metadatas'], 
            saved_data['embeddings'], 
            embedding_model
        )
        return search_engine
    else:
        print("⚠️ [rag_core] 找不到索引檔，嘗試重新生成...")
        texts, metadatas, embeddings = load_and_embed_files(FILE_PATHS)
        search_engine = MedicalSearchEngine(texts, metadatas, embeddings, embedding_model)
        
        # 補存檔
        with open(VECTOR_STORE_PATH, "wb") as f:
            pickle.dump({
                'texts': texts,
                'metadatas': metadatas,
                'embeddings': embeddings
            }, f)
        return search_engine

# --- 這裡讓 rag_core.py 也可以單獨執行測試 ---
if __name__ == "__main__":
    print("這是核心模組，正在進行自我測試...")
    engine = initialize_rag_system()
    res = engine.search("頭痛", k=1)
    print(f"測試結果: {res}")