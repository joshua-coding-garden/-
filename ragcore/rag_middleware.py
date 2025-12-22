import time
import requests
import json
# 1. 引入 OpenCC
from opencc import OpenCC 
from rag_core import initialize_rag_system

# ==========================================
# 🔧 設定區
# ==========================================
WINDOWS_IP = "172.18.112.1"  # 您目前的 Windows IP
OLLAMA_API_URL = f"http://{WINDOWS_IP}:11434/api/generate"
MODEL_NAME = "qwen2.5:14b"

# 初始化轉換器 (s2twp = Simplified to Traditional Taiwan with Phrases)
# 這會連用語都順便修正 (例如：信息 -> 訊息, 質量 -> 品質)
cc = OpenCC('s2twp')

class RAGController:
    def __init__(self):
        print("🚀 [中轉站] 系統啟動中...")
        self.engine = initialize_rag_system()
        print(f"✅ [中轉站] RAG 引擎掛載完成！目標模型: {MODEL_NAME}")

    def get_knowledge_context(self, user_question):
        # 這裡不變，負責撈資料
        results = self.engine.search(user_question, k=3)
        context_str = ""
        if not results:
            return "" # 沒資料就留空，讓 Prompt 決定怎麼回

        for i, res in enumerate(results):
            doc = res['doc']
            score = res['score']
            if score < 0.35: continue 
            
            context_str += f"【參考文獻 {i+1}】\n"
            context_str += f"內容: {doc['a']}\n\n"
        
        return context_str

    def ask_ollama(self, user_question):
        # 1. 檢索資料
        knowledge = self.get_knowledge_context(user_question)
        
        # 2. 準備 Prompt (🔥 關鍵優化處)
        # 指令重點：
        # - 角色：台灣醫師 (語氣親切、專業)
        # - 結構：先講結論 -> 再解釋原因 -> 最後給建議
        # - 禁語：不要說 "根據資料..."
        prompt = f"""
你是一位經驗豐富且親切的「台灣醫師」。
請閱讀以下的【醫療文獻】，並用「台灣繁體中文」回答患者的問題。

【回答守則】
1. **語氣自然**：像在診間對話一樣，溫暖且專業。不要有翻譯腔。
2. **結論先行**：第一句話直接回答是或否，接著再解釋原因。
3. **消除焦慮**：如果患者的問題涉及迷思（如性病、癌症），請給予正確觀念並安撫情緒。
4. **禁止機械式用語**：不要說「根據參考資料顯示」、「文獻提到」，請直接內化成你的知識說出來。

=== 醫療文獻 ===
{knowledge}
=== 文獻結束 ===

患者問題：{user_question}
醫師回答：
"""
        # 3. 設定請求參數 (微調版)
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.4,   # 稍微提高一點點，讓語句更流暢不呆版 (原 0.3)
                "top_p": 0.9,         # 核取樣，讓用詞稍微多樣化
                "repetition_penalty": 1.1, # 避免重複囉嗦
                "num_ctx": 4096
            }
        }

        print(f"🤖 [模型] 正在思考並撰寫建議...")
        
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
            response.raise_for_status()
            
            result_json = response.json()
            raw_answer = result_json.get("response", "")
            
            # 🔥 4. 最後一關：用 OpenCC 強制轉繁體
            final_answer = cc.convert(raw_answer)
            
            return final_answer

        except requests.exceptions.ConnectionError:
            return f"❌ 連線失敗！請確認 Windows IP 是否變更或防火牆設定。"
        except Exception as e:
            return f"❌ 發生錯誤: {e}"

if __name__ == "__main__":
    bot = RAGController()
    while True:
        q = input("\n請輸入醫療問題 (輸入 q 離開): ").strip()
        if q.lower() == 'q': break
        if not q: continue

        answer = bot.ask_ollama(q)
        print("\n" + "="*20 + " 🩺 AI 醫師建議 " + "="*20)
        print(answer) # 這裡印出來的就會是漂亮的繁體中文了
        print("="*55)