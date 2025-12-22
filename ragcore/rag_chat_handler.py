import requests
import json
from opencc import OpenCC

# 引用您原本的設定
WINDOWS_IP = "172.18.112.1"
OLLAMA_API_URL = f"http://{WINDOWS_IP}:11434/api/generate"
MODEL_NAME = "qwen2.5:14b"
cc = OpenCC('s2twp')

class MultiTurnRAG:
    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        # 簡單的記憶體暫存，實際生產環境通常用 Redis 或資料庫
        # 結構: { "user_id": [ {"role": "user", "content": "..."}, ... ] }
        self.sessions = {} 

    def get_history(self, user_id, limit=6):
        """取得最近 N 輪對話歷史"""
        history = self.sessions.get(user_id, [])
        return history[-limit:] # 限制長度避免 Token 爆炸

    def update_history(self, user_id, role, content):
        if user_id not in self.sessions:
            self.sessions[user_id] = []
        self.sessions[user_id].append({"role": role, "content": content})

    def rewrite_query(self, user_question, history):
        """
        【關鍵步驟】
        利用 LLM 將「多輪對話」中的代詞（它、這個、那個人...）
        還原成具體的名詞，變成一個「獨立可搜尋的問題」。
        """
        if not history:
            return user_question

        # 將歷史轉為字串供 Prompt 使用
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        
        prompt = f"""
你是一個對話分析助手。請根據以下的【對話歷史】，將使用者的【最新問題】改寫為一個「語意完整且獨立的問題」。
只要補全省略的主詞或釐清代名詞即可，不要回答問題，也不要改變原意。

【對話歷史】
{history_str}

【最新問題】
{user_question}

【改寫後的獨立問題】：
"""
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1} # 溫度低一點，保持精準
        }
        
        try:
            print(f"🔄 [Rewriter] 正在重寫問題: {user_question}")
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
            result = response.json().get("response", "").strip()
            print(f"✅ [Rewriter] 重寫結果: {result}")
            return result
        except:
            print("⚠️ 重寫失敗，使用原始問題")
            return user_question

    def process_chat(self, user_id, user_question):
        # 1. 取得歷史
        history = self.get_history(user_id)

        # 2. 【關鍵】重寫問題 (解決 "它" 是誰的問題)
        search_query = self.rewrite_query(user_question, history)

        # 3. 使用重寫後的問題去 RAG 搜尋 (呼叫您原本的 engine)
        results = self.rag_engine.search(search_query, k=3)
        
        # 整理檢索結果
        context_str = ""
        sources = []
        for i, res in enumerate(results):
            if res['score'] < 0.35: continue
            context_str += f"【文獻 {i+1}】{res['doc']['a']}\n"
            sources.append({"id": i+1, "content": res['doc']['a'], "score": round(res['score']*10, 2)})

        # 4. 生成最終回答 (加入 Context + History)
        # 這裡的 Prompt 稍微調整，讓模型知道有歷史對話的存在
        final_prompt = f"""
你是一位台灣醫師。請參考【歷史對話】與【醫療文獻】，回答患者的最新問題。

【歷史對話】
{history}

【醫療文獻】
{context_str}

【患者最新問題】
{user_question}

醫師回答 (繁體中文，親切專業)：
"""
        
        payload = {
            "model": MODEL_NAME,
            "prompt": final_prompt,
            "stream": False,
            "options": {
                "temperature": 0.4,
                "num_ctx": 4096
            }
        }

        print(f"🤖 [Chat] 生成最終回答...")
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        raw_answer = response.json().get("response", "")
        final_answer = cc.convert(raw_answer)

        # 5. 更新歷史
        self.update_history(user_id, "user", user_question)
        self.update_history(user_id, "assistant", final_answer)

        return final_answer, sources
