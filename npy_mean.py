import numpy as np
import os
import glob

# --- 設定 ---
# 属性
ATTR = "relaxing"
# .npyファイルが格納されているディレクトリ
INPUT_DIR = f'./mood_npys/{ATTR}' 
# 計算結果の平均を保存するファイルパス
OUTPUT_FILE = f'./attrib/{ATTR}.npy'
# 期待される配列のshape
EXPECTED_SHAPE = (512,)
# --- 設定終わり ---


def calculate_average():
    """設定に基づいて.npyファイルの平均を計算し保存する"""
    search_path = os.path.join(INPUT_DIR, '*.npy')
    npy_files = glob.glob(search_path)

    if not npy_files:
        print(f"エラー: ディレクトリ '{INPUT_DIR}' に.npyファイルが見つかりませんでした。")
        return

    print(f"{len(npy_files)} 個の.npyファイルが見つかりました。")

    total_sum = np.zeros(EXPECTED_SHAPE, dtype=np.float64)
    valid_file_count = 0

    for file_path in npy_files:
        try:
            array = np.load(file_path)
            if array.shape != EXPECTED_SHAPE:
                print(f"警告: '{file_path}'のshapeが期待値{EXPECTED_SHAPE}と異なります。スキップします。 Shape: {array.shape}")
                continue
            
            total_sum += array
            valid_file_count += 1
        except Exception as e:
            print(f"警告: '{file_path}'の読み込み中にエラーが発生しました。スキップします。詳細: {e}")

    if valid_file_count == 0:
        print("エラー: 平均を計算できる有効なファイルがありませんでした。")
        return

    # 平均を計算
    average_array = total_sum / valid_file_count

    # 結果を保存
    np.save(OUTPUT_FILE, average_array)
    print("\n処理が完了しました。")
    print(f"平均化された配列が '{OUTPUT_FILE}' に保存されました。")
    print(f"Shape: {average_array.shape}, Dtype: {average_array.dtype}")

if __name__ == '__main__':
    calculate_average()