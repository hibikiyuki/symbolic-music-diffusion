# 多分使わなくていい

import os
import note_seq
from magenta.models.music_vae import configs
# from magenta.common import search_for_files
import glob

# --- 設定項目 ---
INPUT_MIDI_DIR = 'data/lakh/test'  # 元のMIDIファイルが入っているディレクトリ
OUTPUT_MIDI_DIR = 'data/lakh/test_preprocessed' # 分割後のMIDIを保存するディレクトリ
CONFIG_NAME = 'cat-mel_2bar_big' # 使用するモデルのConfig名
# ----------------

def preprocess_midi_files(input_dir, output_dir, config_name):
  """MIDIファイルを指定されたconfigに基づいてサブシーケンスに分割する"""
  
  # 出力ディレクトリを作成
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # モデルのconfigをロード
  config = configs.CONFIG_MAP[config_name]
  
  # MIDIファイルを取得
  # input_files = search_for_files(input_dir, ['mid', 'midi'])
  input_files = glob.glob(f'{INPUT_MIDI_DIR}/*.mid')

  print(f"Found {len(input_files)} MIDI files in {input_dir}")

  for midi_path in input_files:
    print(f"Processing: {midi_path}")
    try:
      ns = note_seq.midi_file_to_note_sequence(midi_path)
      
      # configのdata_converterを使ってサブシーケンスを抽出
      # to_tensorsは内部で NoteSequence をモデルが扱える単位に分割する
      tensors = config.data_converter.to_tensors(ns).outputs
      
      if not tensors:
        print(f"  -> No valid subsequences found. Skipping.")
        continue
      
      # テンソルをNoteSequenceに戻す
      subsequences = config.data_converter.from_tensors(tensors)
      
      base_filename = os.path.splitext(os.path.basename(midi_path))[0]
      
      for i, sub_ns in enumerate(subsequences):
        output_filename = os.path.join(output_dir, f"{base_filename}_{i}.mid")
        note_seq.sequence_proto_to_midi_file(sub_ns, output_filename)
      
      print(f"  -> Saved {len(subsequences)} subsequences.")
      
    except Exception as e:
      print(f"  -> Error processing file: {e}")

if __name__ == '__main__':
  preprocess_midi_files(INPUT_MIDI_DIR, OUTPUT_MIDI_DIR, CONFIG_NAME)