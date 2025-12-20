import json
import os
import pickle
import torch
import time
import numpy as np
# 👇 直接使用 sklearn 的核心搜尋套件
from sklearn.neighbors import NearestNeighbors 
from langchain_huggingface import HuggingFaceEmbeddings
# 這裡我們只用它來生成向量，不存取
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.documents import Document

# --- ⚙️ 設定區 ---
FILE_PATHS = ['健康001_QA_繁體.json', '醫學問答001_QA_繁體.json']
VECTOR_STORE_PATH = "medical_rag_store.pkl"

TEST_MODE = False
TEST_LIMIT = 5000 

# --- 1. 硬體加速設定 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--------------------------------------------------")
print(f"🔥 運算裝置: {device.upper()}")
if device == "cuda":
    print(f"🚀 偵測到顯卡: {torch.cuda.get_device_name(0)}")
else:
    print("❌ 警告：目前正在使用 CPU！")
print(f"--------------------------------------------------")

print("正在載入 Embedding 模型...")
embedding_model = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={'device': device},
    encode_kwargs={'batch_size': 64}
)

# --- 定義一個輕量級的搜尋引擎類別 (繞過 LangChain 檢查) ---
class MedicalSearchEngine:
    def __init__(self, texts, metadatas, embeddings, embedding_func):
        self.texts = texts
        self.metadatas = metadatas
        self.embedding_func = embedding_func
        self.embeddings_np = np.array(embeddings)
        
        # 建立搜尋索引
        print("   🔧 啟動高效能搜尋引擎 (KNN)...")
        self.knn = NearestNeighbors(n_neighbors=5, metric='l2')
        self.knn.fit(self.embeddings_np)

    def search(self, query, k=5):
        # 1. 把使用者的問題轉成向量
        query_emb = self.embedding_func.embed_query(query)
        query_emb_np = np.array([query_emb])
        
        # 2. 進行數學計算 (找出最接近的 k 個)
        dists, indices = self.knn.kneighbors(query_emb_np, n_neighbors=k)
        
        # 3. 整理結果
        results = []
        for dist, idx in zip(dists[0], indices[0]):
            results.append({
                "doc": self.metadatas[idx], # 這裡面有 original_question 和 original_answer
                "score": dist
            })
        return results

def load_json_files(file_paths):
    documents = []
    for file_path in file_paths:
        if not os.path.exists(file_path): continue
        print(f"📂 正在讀取 {file_path} ...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if TEST_MODE:
                data = data[:TEST_LIMIT]
                print(f"   ⚡ 測試模式：只取前 {len(data)} 筆")
            else:
                print(f"   📊 全量模式：讀取共 {len(data)} 筆")

            for item in data:
                q = item.get('question', '')
                a = item.get('answer', '')
                doc = Document(
                    page_content=f"問題: {q}\n答案: {a}",
                    metadata={"original_question": q, "original_answer": a}
                )
                documents.append(doc)
                if TEST_MODE and len(documents) >= TEST_LIMIT: break
        except Exception: pass
        if TEST_MODE and len(documents) >= TEST_LIMIT: break
    return documents

def save_db_data(vector_db, path):
    print(f"💾 正在擷取數據並存檔...")
    data = {
        "texts": vector_db._texts,
        "embeddings": vector_db._embeddings,
        "metadatas": vector_db._metadatas,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    size_mb = os.path.getsize(path)/1024/1024
    print(f"✅ 存檔成功！檔案大小: {size_mb:.2f} MB")

def load_engine_from_file(path, embedding_func):
    print(f"📂 正在讀取數據檔 {path} (請耐心等待)...")
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    text_count = len(data.get("texts", []))
    print(f"   👀 檢查: 檔案中包含 {text_count} 筆資料")
    
    # 直接回傳我們自定義的引擎，不再使用 LangChain Store
    return MedicalSearchEngine(
        texts=data["texts"],
        metadatas=data["metadatas"],
        embeddings=data["embeddings"],
        embedding_func=embedding_func
    )

# --- 主程式 ---
if __name__ == "__main__":
    search_engine = None
    
    # 1. 嘗試讀檔
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"⚠️ 發現舊的存檔 {VECTOR_STORE_PATH}")
        user_input = input("❓ 是否要刪除舊檔並重新跑全量運算？(y/n): ")
        
        if user_input.lower() == 'y':
            print("🗑️ 刪除舊檔，準備重新運算...")
            try: os.remove(VECTOR_STORE_PATH)
            except: pass
        else:
            print("📂 嘗試載入舊檔...")
            try:
                search_engine = load_engine_from_file(VECTOR_STORE_PATH, embedding_model)
                print("✅ 舊檔載入成功！")
            except Exception as e:
                print(f"\n❌ 讀檔失敗！錯誤原因: {e}")
                print("💡 請選擇 'y' 重跑一次。")
                exit()

    # 2. 如果沒有引擎 (代表需要新建)
    if search_engine is None:
        docs = load_json_files(FILE_PATHS)
        if not docs: exit()
            
        print(f"\n📊 準備將 {len(docs)} 筆資料送入 RTX 3070 運算...")
        print("⏱️ 開始計時...")
        start_time = time.time()
        
        # 這裡借用 LangChain 來做第一次的向量計算 (因為它有 batching 比較方便)
        temp_db = SKLearnVectorStore.from_documents(
            documents=docs,
            embedding=embedding_model
        )
        
        print(f"🏁 運算完成！耗時: {time.time() - start_time:.2f} 秒")
        
        # 存檔
        save_db_data(temp_db, VECTOR_STORE_PATH)
        
        # 轉換成我們的搜尋引擎
        search_engine = MedicalSearchEngine(
            texts=temp_db._texts,
            metadatas=temp_db._metadatas,
            embeddings=temp_db._embeddings,
            embedding_func=embedding_model
        )

    print("\n✅ 系統就緒！")
    print("--------------------------------------------------")
    while True:
        try:
            query = input("\n請輸入醫學問題 (q 離開): ")
            if query.lower() in ['q', 'exit']: break
            if not query.strip(): continue
            
            # 使用自定義引擎搜尋
            results = search_engine.search(query, k=5)
            
            print(f"\n🔍 搜尋結果 Top 5:")
            for i, res in enumerate(results):
                doc = res["doc"]
                score = res["score"]
                print(f"\n🏆 Top {i+1} (Score: {score:.4f})")
                print(f"❓ Q: {doc['original_question']}")
                # 完整印出答案，沒有切片
                print(f"💡 A: {doc['original_answer']}") 
                print("-" * 30)
        except KeyboardInterrupt:
            break