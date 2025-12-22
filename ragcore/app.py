from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_core import initialize_rag_system
# 引入新寫的模組
from rag_chat_handler import MultiTurnRAG 

app = Flask(__name__)
CORS(app)

# 全域變數
rag_engine = None
chat_handler = None

def init_system():
    global rag_engine, chat_handler
    if rag_engine is None:
        rag_engine = initialize_rag_system()
        # 初始化多輪對話處理器
        chat_handler = MultiTurnRAG(rag_engine)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    user_question = data.get('question', '')
    # 前端可以傳送一個 user_id (或 session_id) 來區分不同使用者
    # 這裡暫時預設為 "demo_user"
    user_id = data.get('user_id', 'demo_user') 
    
    if not user_question:
        return jsonify({"answer": "請輸入問題"}), 400

    init_system() # 確保系統已初始化
    
    try:
        # 使用新的 chat_handler 處理 (包含重寫、檢索、生成、紀錄歷史)
        answer, sources = chat_handler.process_chat(user_id, user_question)
        
        return jsonify({
            "answer": answer,
            "sources": sources
        })

    except Exception as e:
        print(f"❌ 錯誤: {e}")
        return jsonify({"answer": "系統忙碌中...", "sources": []}), 500

if __name__ == "__main__":
    init_system()
    app.run(host='0.0.0.0', port=5000, debug=True)
