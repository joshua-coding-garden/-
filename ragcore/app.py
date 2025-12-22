from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from opencc import OpenCC
from rag_core import initialize_rag_system

# 1. 建立 Flask App (這一行就是您缺少的)
app = Flask(__name__)
CORS(app)  # 允許跨域請求

# ================= 設定區 =================
WINDOWS_IP = "172.18.112.1"  # 請確認這是否為您當前的 IP
OLLAMA_API_URL = f"http://{WINDOWS_IP}:11434/api/generate"
MODEL_NAME = "qwen2.5:14b"
cc = OpenCC('s2twp')

# 全域變數存放 RAG 引擎
rag_engine = None

def get_rag_engine():
    global rag_engine
    if rag_engine is None:
        print("🚀 [Server] 正在初始化 RAG 引擎...")
        rag_engine = initialize_rag_system()
    return rag_engine

# 2. 修改後的上下文獲取函式 (包含分數處理)
def get_knowledge_context(engine, user_question):
    results = engine.search(user_question, k=3) 
    context_str = ""
    source_data = [] # 用來存給前端顯示的資料

    if not results:
        return "", []

    for i, res in enumerate(results):
        doc = res['doc']
        score = res['score'] 
        
        # 過濾低分
        if score < 0.35: continue 
        
        # 1. 拼湊給 LLM 看的字串
        context_str += f"【參考文獻 {i+1}】\n內容: {doc['a']}\n\n"
        
        # 2. 準備給前端顯示的詳細資料
        source_data.append({
            "id": i + 1,
            "content": doc['a'],
            "score": round(score * 10, 2) # 將分數轉換為易讀格式
        })
        
    return context_str, source_data

@app.route('/')
def index():
    return "<h1>🚀 AI 醫療伺服器運作中！</h1><p>請打開 index.html 來使用聊天介面。</p>"

# 3. 修改後的 API 接口
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    user_question = data.get('question', '')
    
    if not user_question:
        return jsonify({"answer": "請輸入問題"}), 400

    engine = get_rag_engine()
    
    # 接收兩個回傳值：文獻內容字串、來源詳細資料
    knowledge, sources = get_knowledge_context(engine, user_question)

    prompt = f"""
你是一位經驗豐富且親切的「台灣醫師」。
請閱讀以下的【醫療文獻】，並用「台灣繁體中文」回答患者的問題。

【回答守則】
1. **語氣自然**：像在診間對話一樣，溫暖且專業。
2. **結論先行**：第一句話直接回答重點，接著解釋原因。
3. **消除焦慮**：給予正確觀念並安撫情緒。
4. **禁止機械式用語**：不要說「根據資料」，直接內化成知識。

=== 醫療文獻 ===
{knowledge}
=== 文獻結束 ===

患者問題：{user_question}
醫師回答：
"""
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "num_ctx": 4096
        }
    }

    try:
        print(f"🤖 [Server] 向 Ollama 發送請求: {user_question}")
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        
        result_json = response.json()
        raw_answer = result_json.get("response", "")
        final_answer = cc.convert(raw_answer)
        
        # 回傳答案與來源資料給前端
        return jsonify({
            "answer": final_answer,
            "sources": sources
        })

    except Exception as e:
        print(f"❌ 錯誤: {e}")
        return jsonify({"answer": "目前系統繁忙，請檢查後端連線。", "sources": []}), 500

if __name__ == "__main__":
    get_rag_engine() # 預熱模型
    print("✅ 伺服器已修復並啟動")
    app.run(host='0.0.0.0', port=5000, debug=True)
