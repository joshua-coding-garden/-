import json
import os
from opencc import OpenCC

def convert_health_data_to_qa(input_file, output_file):
    # 1. 初始化繁簡轉換器
    cc = OpenCC('s2t')
    
    if not os.path.exists(input_file):
        print(f"錯誤：找不到檔案 '{input_file}'")
        return

    print(f"正在讀取 {input_file} ...")
    
    qa_list = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # --- 關鍵修改：逐行讀取 (處理 JSONL 格式) ---
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # 跳過空行
                if not line:
                    continue
                
                try:
                    # 解析每一行的 JSON
                    data = json.loads(line)
                    
                    # 定義提取與轉換函數
                    def process_doc_list(doc_list):
                        for item in doc_list:
                            # 提取標題與內容
                            s_title = item.get('title', '')
                            s_content = item.get('content', '')
                            
                            # 繁簡轉換
                            t_question = cc.convert(s_title)
                            t_answer = cc.convert(s_content)
                            
                            # 只有當標題和內容都有值時才加入
                            if t_question and t_answer:
                                qa_list.append({
                                    "question": t_question,
                                    "answer": t_answer
                                })

                    # 2. 檢查並處理 positive_doc
                    if 'positive_doc' in data and isinstance(data['positive_doc'], list):
                        process_doc_list(data['positive_doc'])
                    
                    # 3. 檢查並處理 negative_doc
                    if 'negative_doc' in data and isinstance(data['negative_doc'], list):
                        process_doc_list(data['negative_doc'])
                        
                except json.JSONDecodeError as e:
                    # 如果某一行格式爛掉，只報錯該行，程式繼續執行
                    print(f"警告：第 {line_num} 行無法解析 JSON，已跳過。錯誤原因：{e}")
                    continue

        # 4. 寫入新的 JSON 檔案
        print(f"處理完成！總共提取到 {len(qa_list)} 筆 QA 資料。")
        print(f"正在寫入 {output_file} ...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_list, f, ensure_ascii=False, indent=4)
            
        print("轉換成功！")

    except Exception as e:
        print(f"發生未預期的錯誤：{e}")

# --- 執行設定 ---
if __name__ == "__main__":
    # 請確認檔名是否正確
    input_filename = '健康001-簡中.json'
    output_filename = '健康001_QA_繁體.json'
    
    convert_health_data_to_qa(input_filename, output_filename)