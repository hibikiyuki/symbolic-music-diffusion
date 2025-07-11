import json
import os
import shutil
from tqdm import tqdm

# --- 設定項目 ---

# 1. MidiCapsのデータファイル（JSONL形式を想定）
DATASET_FILE = 'train.json' # ご自身のファイル名に変更してください

# 2. 元のMIDIファイルが展開されているルートディレクトリ
# 例: 'lmd_full/' フォルダが '/path/to/midi/lmd_full' にある場合
MIDI_ROOT_DIR = '' # 必ずご自身の環境に合わせて変更してください

# 3. コピー先のディレクトリ
OUTPUT_DIR = './mood_midi_collection_2000'

# 4. 分析したいmoodのリスト
TARGET_MOODS = [
    'dark', 'dream', 'emotional', 'energetic', 'happy',
    'inspiring', 'love', 'meditative', 'motivational', 'positive', 'relaxing'
]

# 5. 各moodから抽出する曲数
NUM_SAMPLES_PER_MOOD = 2000 # 必要に応じて変更してください

# --- スクリプト本体 ---

def process_midicaps_dataset():
    # 出力ディレクトリを作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"出力先ディレクトリ: {OUTPUT_DIR}")

    # 1. まず全データをメモリに読み込む
    print(f"データセットファイル '{DATASET_FILE}' を読み込み中...")
    all_data = []
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                all_data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"合計 {len(all_data)} 件のデータを読み込みました。")


    # 2. 各moodごとに処理
    for mood_name in TARGET_MOODS:
        print(f"\n--- ムード '{mood_name}' の処理を開始 ---")
        
        # 3. 対象moodを含むデータをフィルタリングし、確率を紐付ける
        mood_specific_data = []
        for entry in all_data:
            if 'mood' in entry and mood_name in entry['mood']:
                try:
                    # moodリスト内のインデックスを取得
                    idx = entry['mood'].index(mood_name)
                    # 対応するmood_probを取得
                    prob = entry['mood_prob'][idx]
                    # 元のデータに確率を一時的に追加してリストに保存
                    entry_with_prob = entry.copy()
                    entry_with_prob['target_mood_prob'] = prob
                    mood_specific_data.append(entry_with_prob)
                except (ValueError, IndexError):
                    # moodとmood_probの対応が取れない場合はスキップ
                    continue
        
        if not mood_specific_data:
            print(f"ムード '{mood_name}' が含まれるデータが見つかりませんでした。")
            continue
            
        print(f"'{mood_name}' を含む曲が {len(mood_specific_data)} 件見つかりました。")

        # 4. 確率が高い順にソート
        mood_specific_data.sort(key=lambda x: x['target_mood_prob'], reverse=True)
        
        # 5. 上位から指定した数を抽出
        top_samples = mood_specific_data[:NUM_SAMPLES_PER_MOOD]
        print(f"上位 {len(top_samples)} 件を抽出します。")
        
        # 6. MIDIファイルをコピー
        mood_output_dir = os.path.join(OUTPUT_DIR, mood_name)
        os.makedirs(mood_output_dir, exist_ok=True)
        
        print(f"MIDIファイルを '{mood_output_dir}' にコピーしています...")
        copy_count = 0
        for sample in tqdm(top_samples, desc=f"Copying for {mood_name}"):
            # 元のMIDIファイルのフルパスを構築
            # locationは 'lmd_full/...' の形式なので、MIDI_ROOT_DIRと結合
            source_path = os.path.join(MIDI_ROOT_DIR, sample['location'])
            
            # コピー先のパス
            file_name = os.path.basename(sample['location'])
            dest_path = os.path.join(mood_output_dir, file_name)
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                copy_count += 1
            else:
                print(f"[警告] ファイルが見つかりません: {source_path}")
                
        print(f"'{mood_name}' のために {copy_count} 個のファイルをコピーしました。")

    print("\nすべての処理が完了しました。")


if __name__ == '__main__':
    # MIDI_ROOT_DIR が初期値のままの場合、ユーザーに確認を促す
    if MIDI_ROOT_DIR == '/path/to/your/midi_files/':
        print("!!! 警告 !!!")
        print("スクリプト内の 'MIDI_ROOT_DIR' を、ご自身のMIDIファイルが保存されているディレクトリパスに必ず変更してください。")
        print("例: MIDIファイルが '/home/user/downloads/lmd_full' にある場合、")
        print("   MIDI_ROOT_DIR = '/home/user/downloads/' と設定します。")
    else:
        process_midicaps_dataset()