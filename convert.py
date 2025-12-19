import json
import os
from opencc import OpenCC

def convert_to_qa_traditional(input_file, output_file):
    # 1. 初始化繁簡轉換器 (s2t = Simplified to Traditional)
    cc = OpenCC('s2t')
    
    # 檢查檔案是否存在
    if not os.path.exists(input_file):
        print(f"錯誤：找不到檔案 '{input_file}'")
        return

    print(f"正在讀取 {input_file} ...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # 讀取 JSON 數據，處理可能的格式錯誤
            # 如果文件包含多個 JSON 對象（如行分隔的 JSON），需逐行讀取
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                # 嘗試逐行讀取
                f.seek(0)
                lines = f.readlines()
                # 嘗試修復常見的格式問題，例如只有一個對象但包含在文件中
                if len(lines) > 1:
                     print(f"警告：JSON 格式可能不標準，嘗試逐行解析...")
                     # 這邊假設是單個 JSON 對象分佈在多行，或者是多個 JSON 對象
                     # 為了保險起見，我們假設它是一個完整的 JSON 對象
                     # 如果是多個對象（例如每行一個），需要不同的處理
                     # 根據您的錯誤信息 "Extra data: line 2 column 1"，這通常意味著文件包含多個 JSON 對象
                     # 我們將嘗試讀取第一個對象
                     content = "".join(lines)
                     # 嘗試找到第一個完整的 JSON 對象的結束位置
                     # 這是一個簡單的啟發式方法，可能不適用於所有情況
                     brace_count = 0
                     json_str = ""
                     for char in content:
                         json_str += char
                         if char == '{':
                             brace_count += 1
                         elif char == '}':
                             brace_count -= 1
                             if brace_count == 0:
                                 break
                     try:
                        data = json.loads(json_str)
                     except json.JSONDecodeError:
                         print("無法解析 JSON 數據。")
                         return
                else:
                    print(f"JSON 解析錯誤: {e}")
                    return

            
        qa_list = []

        # 2. 定義一個提取函數，處理不同的資料區塊
        def extract_and_convert(doc_list):
            for item in doc_list:
                # 提取 title 作為 question，content 作為 answer
                # 使用 .get() 防止欄位缺失報錯
                s_title = item.get('title', '')
                s_content = item.get('content', '')
                
                # 進行繁體轉換
                t_question = cc.convert(s_title)
                t_answer = cc.convert(s_content)
                
                # 只有當標題和內容都有值時才加入
                if t_question and t_answer:
                    qa_list.append({
                        "question": t_question,
                        "answer": t_answer
                    })

        # 3. 遍歷 JSON 中的 positive_doc 和 negative_doc
        # 根據您之前的檔案結構，這兩個 key 包含主要數據
        if 'positive_doc' in data:
            print(f"正在處理 positive_doc ({len(data['positive_doc'])} 筆)...")
            extract_and_convert(data['positive_doc'])
            
        if 'negative_doc' in data:
            print(f"正在處理 negative_doc ({len(data['negative_doc'])} 筆)...")
            extract_and_convert(data['negative_doc'])

        # 4. 寫入新的 JSON 檔案
        print(f"正在寫入 {output_file}，共 {len(qa_list)} 筆 QA 資料...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 確保輸出的是可讀的中文而非 Unicode 編碼
            # indent=4 讓輸出的 JSON 格式美化，易於閱讀
            json.dump(qa_list, f, ensure_ascii=False, indent=4)
            
        print("轉換成功！")

    except Exception as e:
        print(f"發生錯誤：{e}")

# --- 執行設定 ---
if __name__ == "__main__":
    # 輸入檔名 (請確認您的檔案名稱是否完全一致)
    input_filename = '醫學問答001-簡中.json'
    
    # 輸出檔名
    output_filename = '醫學問答001_QA_繁體.json'
    
    convert_to_qa_traditional(input_filename, output_filename)