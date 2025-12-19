import json
import os
from opencc import OpenCC

# ================= 設定區 =================
INPUT_FILENAME = 'source.json'
OUTPUT_FILENAME = 'converted_qa_dataset.json'
# =========================================

def mine_and_convert():
    if not os.path.exists(INPUT_FILENAME):
        print(f"❌ 錯誤：找不到檔案 '{INPUT_FILENAME}'")
        return

    print(f"⛏️  啟動 '資料挖掘機' 模式...")
    print("正在無視檔案結構，直接從亂碼中搶救資料...")

    cc = OpenCC('s2twp')
    qa_pairs = []
    
    # 讀取整個檔案內容
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ 讀取檔案失敗: {e}")
        return

    decoder = json.JSONDecoder()
    pos = 0
    total_len = len(content)
    success_count = 0

    while pos < total_len:
        # 1. 尋找下一個 '{' 的位置
        # 我們跳過空白，尋找物件的開始
        try:
            # 略過非 '{' 的字元 (這能幫助我們跳過逗號、中括號等結構字元)
            while pos < total_len and content[pos] != '{':
                pos += 1
            
            if pos >= total_len:
                break

            # 2. 嘗試從這個位置解析一個 JSON 物件
            obj, end_pos = decoder.raw_decode(content, idx=pos)
            
            # 3. 如果解析成功，檢查這是不是我們要的資料
            # 情況 A: 原始資料 (title / content)
            if isinstance(obj, dict) and 'title' in obj and 'content' in obj:
                title = obj.get('title', '')
                text = obj.get('content', '')
                if title and text:
                    qa_pairs.append({
                        "question": cc.convert(title),
                        "answer": cc.convert(text)
                    })
                    success_count += 1

            # 情況 B: 已經是 QA 格式 (question / answer) - 針對你結尾看到的那些資料
            elif isinstance(obj, dict) and 'question' in obj and 'answer' in obj:
                q = obj.get('question', '')
                a = obj.get('answer', '')
                if q and a:
                    qa_pairs.append({
                        "question": q, # 假設已經是繁體，若不是可加 cc.convert
                        "answer": a
                    })
                    success_count += 1
            
            # 情況 C: 如果這是最外層的大包裝 (positive_doc)，我們不想解析它，
            # 因為它可能包含了裡面所有的資料，會導致重複或記憶體爆炸。
            # 但 raw_decode 通常是貪婪的。如果它解析了整個大物件，我們就拿不到裡面的小物件了。
            # 為了避免這個問題，如果解析出的物件包含 'positive_doc' 這個 key，我們視為無效，
            # 強制指標只前進 1 格，讓迴圈繼續往裡面找小物件。
            if isinstance(obj, dict) and ('positive_doc' in obj or 'negative_doc' in obj):
                pos += 1 # 放棄這個大物件，鑽進去裡面找
                continue

            # 解析成功且處理完畢，將指標移到這個物件的結束位置
            pos = end_pos
            
            if success_count % 100 == 0:
                print(f"已挖掘出 {success_count} 筆資料...", end='\r')

        except json.JSONDecodeError:
            # 如果在這裡解析失敗，代表這不是一個完整的 JSON 物件
            # 我們就往前移動一格，繼續嘗試下一個 '{'
            pos += 1
        except Exception:
            pos += 1

    print(f"\n\n🎉 挖掘結束！總共搶救出 {len(qa_pairs)} 筆資料 (目標: 1615)。")
    
    if len(qa_pairs) > 0:
        print(f"💾 正在存檔至 '{OUTPUT_FILENAME}'...")
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=4)
        print("✨ 成功！")
    else:
        print("⚠️ 警告：沒有挖到任何資料。")

if __name__ == "__main__":
    mine_and_convert()
