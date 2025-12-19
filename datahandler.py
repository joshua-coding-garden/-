import json
import sys
from opencc import OpenCC

# 設定標準輸出為 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def read_and_convert_json(input_file, output_file):
    """
    讀取 JSONL 檔案並轉換成 QA 格式（簡體轉繁體）
    
    Args:
        input_file: 輸入的 JSON/JSONL 檔案路徑
        output_file: 輸出的 JSON 檔案路徑
    """
    # 初始化 OpenCC 簡體轉繁體
    cc = OpenCC('s2t')  # s2t = Simplified to Traditional
    
    # 1. 讀取 JSONL 檔案（每行一個 JSON 物件）
    qa_list = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳過空行
                continue
            
            try:
                data = json.loads(line)
                
                # 2. 轉換格式 - 處理 positive_doc
                if 'positive_doc' in data:
                    for doc in data['positive_doc']:
                        if doc.get('QA') and doc['QA'].get('question'):
                            qa_item = {
                                'question': cc.convert(doc['QA']['question']),  # 簡體轉繁體
                                'answer': cc.convert(doc.get('content', ''))    # 簡體轉繁體
                            }
                            qa_list.append(qa_item)
                
                # 處理 negative_doc（如果需要的話）
                if 'negative_doc' in data:
                    for doc in data['negative_doc']:
                        if doc.get('QA') and doc['QA'].get('question'):
                            qa_item = {
                                'question': cc.convert(doc['QA']['question']),  # 簡體轉繁體
                                'answer': cc.convert(doc.get('content', ''))    # 簡體轉繁體
                            }
                            qa_list.append(qa_item)
                            
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} JSON decode error: {e}")
                continue
    
    # 儲存轉換後的資料
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully converted {len(qa_list)} QA pairs (Simplified to Traditional Chinese)")
    return qa_list

# 使用範例
if __name__ == "__main__":
    input_file = "input.json"  # 你的輸入檔案名稱
    output_file = "qa_output.json"  # 輸出檔案名稱
    
    qa_data = read_and_convert_json(input_file, output_file)
    
    # 印出統計資訊
    print(f"\nTotal: {len(qa_data)} data entries")
    
    # 印出第一筆資料作為預覽
    if qa_data:
        print("\nFirst entry preview:")
        print(f"Question length: {len(qa_data[0]['question'])} chars")
        print(f"Answer length: {len(qa_data[0]['answer'])} chars")