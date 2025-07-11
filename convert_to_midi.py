import os
import glob
import sys
import argparse # コマンドライン引数を扱うため

# sample_audio.py が依存する utils.data_utils をインポートするためにパスを追加
# このパスは、お使いのプロジェクトのディレクトリ構造に合わせて調整してください。
# 例: このスクリプトが project_root/tools/convert_to_midi.py にあり、
#     utils が project_root/utils にある場合。
try:
    # sample_audio.py と同じ階層か、utilsディレクトリがPYTHONPATHにある場合
    import utils.data_utils as data_utils
    import note_seq
except ImportError:
    # パスを明示的に追加する例
    # 現在のスクリプトのディレクトリを取得
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root が utils ディレクトリの親であると仮定
    project_root = os.path.abspath(os.path.join(current_script_dir, "..")) # 必要に応じて調整
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    try:
        import utils.data_utils as data_utils
        import note_seq
    except ImportError:
        print("エラー: utils.data_utils または note_seq が見つかりません。")
        print("       Magenta (note-seq)がインストールされているか、")
        print("       また、utils.data_utils.py へのパスが正しいか確認してください。")
        print(f"       (sys.path: {sys.path})")
        sys.exit(1)

def convert_pkl_to_midi(pkl_input_dir, midi_output_dir):
    """
    指定されたディレクトリ内の .pkl (NoteSequence) ファイルを MIDI に変換する。

    Args:
        pkl_input_dir (str): NoteSequence の .pkl ファイルが格納されているディレクトリ。
                             例: './audio/gen/ns'
        midi_output_dir (str): 生成された MIDI ファイルを保存するディレクトリ。
    """
    if not os.path.isdir(pkl_input_dir):
        print(f"エラー: 入力ディレクトリ '{pkl_input_dir}' が存在しません。")
        return

    if not os.path.exists(midi_output_dir):
        os.makedirs(midi_output_dir)
        print(f"作成しました: '{midi_output_dir}'")

    pkl_files = glob.glob(os.path.join(pkl_input_dir, "*.pkl"))
    
    if not pkl_files:
        print(f"情報: '{pkl_input_dir}' に .pkl ファイルが見つかりません。")
        return

    print(f"'{pkl_input_dir}' から .pkl ファイルを処理中...")
    converted_count = 0
    for pkl_file_path in pkl_files:
        try:
            # utils.data_utils.load を使って NoteSequence オブジェクトをロード
            # sample_audio.py で data_utils.save を使って保存されているため、
            # data_utils.load で読み込むのが適切です。
            note_sequence_obj = data_utils.load(pkl_file_path)

            # ロードされたオブジェクトが本当に NoteSequence か確認 (念のため)
            if not isinstance(note_sequence_obj, note_seq.NoteSequence):
                print(f"警告: {pkl_file_path} は有効な NoteSequence オブジェクトではありません。スキップします。")
                continue

            base_name = os.path.basename(pkl_file_path)
            midi_file_name = os.path.splitext(base_name)[0] + ".mid"
            midi_file_path = os.path.join(midi_output_dir, midi_file_name)

            note_seq.note_sequence_to_midi_file(note_sequence_obj, midi_file_path)
            # print(f"MIDI を保存しました: {midi_file_path}")
            converted_count += 1

        except Exception as e:
            print(f"エラー: {pkl_file_path} の処理中にエラーが発生しました: {e}")
    
    print(f"処理完了。{converted_count} 個のファイルを MIDI に変換しました。出力先: '{midi_output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NoteSequence .pkl ファイルを MIDI に変換します。")
    parser.add_argument("input_dir", type=str,
                        help="NoteSequence の .pkl ファイルが格納されている入力ディレクトリ (例: ./audio/gen/ns)。")
    parser.add_argument("output_dir", type=str,
                        help="生成された MIDI ファイルを保存する出力ディレクトリ (例: ./audio/gen/midi)。")
    
    args = parser.parse_args()
    
    convert_pkl_to_midi(args.input_dir, args.output_dir)

    # 使用例 (コマンドラインから実行する場合):
    # python convert_script_name.py ./audio/gen/ns ./audio/gen/midi
    # (convert_script_name.py はこのスクリプトを保存したファイル名)