import tensorflow as tf
import os
import glob
import random
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', None,
                    'Directory containing the input TFRecord files.')
flags.DEFINE_string('output_dir', None,
                    'Directory to save the split TFRecord files.')
flags.DEFINE_float('train_split_ratio', 0.8,
                   'Fraction of data to use for training (0.0 to 1.0).')
# flags.DEFINE_string('train_filename_prefix', 'train', # 削除
#                     'Prefix for training TFRecord files.')
# flags.DEFINE_string('eval_filename_prefix', 'eval',   # 削除
#                     'Prefix for evaluation TFRecord files.')
flags.DEFINE_integer('shard_size', 10000,
                     'Approximate number of records per output shard file.')
flags.DEFINE_integer('random_seed', 42,
                     'Random seed for shuffling the file list (optional).')

def main(argv):
    del argv  # Unused.

    if not FLAGS.input_dir or not FLAGS.output_dir:
        logging.error('Input directory and output directory must be specified.')
        return

    if not 0.0 < FLAGS.train_split_ratio < 1.0:
        logging.error('train_split_ratio must be between 0.0 and 1.0 (exclusive).')
        return

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # 1. 入力TFRecordファイルのリストを取得
    input_files = sorted(glob.glob(os.path.join(FLAGS.input_dir, '*.tfrecord*')))
    if not input_files:
        logging.error(f"No TFRecord files found in {FLAGS.input_dir}")
        return
    logging.info(f"Found {len(input_files)} input TFRecord files.")

    if FLAGS.random_seed is not None:
        random.seed(FLAGS.random_seed)
        random.shuffle(input_files)
        logging.info(f"Shuffled input file list with seed {FLAGS.random_seed}.")

    # 2. 全レコードをイテレートするためのデータセットを作成
    raw_dataset = tf.data.TFRecordDataset(input_files)

    logging.info("Counting total records... This may take a while for large datasets.")
    total_records = 0
    for _ in raw_dataset:
        total_records += 1
    logging.info(f"Total records found: {total_records}")

    if total_records == 0:
        logging.error("No records found in the input TFRecord files.")
        return

    # 3. 分割ポイントを計算
    num_train_records = int(total_records * FLAGS.train_split_ratio)
    num_eval_records = total_records - num_train_records

    logging.info(f"Splitting into {num_train_records} training records and {num_eval_records} evaluation records.")

    # 4. データを分割して新しいTFRecordファイルに書き出す
    train_writer = None
    eval_writer = None
    train_record_count = 0
    eval_record_count = 0
    train_shard_index = 0 # 0-indexed shard number
    eval_shard_index = 0  # 0-indexed shard number

    # 総シャード数を計算 (ファイル名で使用)
    # num_train_shards = ((num_train_records - 1) // FLAGS.shard_size) + 1 if num_train_records > 0 else 0
    # num_eval_shards = ((num_eval_records - 1) // FLAGS.shard_size) + 1 if num_eval_records > 0 else 0
    # より安全な計算 (レコード数が0の場合を考慮)
    num_train_shards = (num_train_records + FLAGS.shard_size - 1) // FLAGS.shard_size if num_train_records > 0 else 0
    num_eval_shards = (num_eval_records + FLAGS.shard_size - 1) // FLAGS.shard_size if num_eval_records > 0 else 0


    raw_dataset_for_writing = tf.data.TFRecordDataset(input_files)
    dataset_iterator = iter(raw_dataset_for_writing)

    # Training データの書き込み
    if num_train_records > 0: # trainデータがある場合のみ処理
        logging.info("Writing training data...")
        for i in range(num_train_records):
            if train_record_count % FLAGS.shard_size == 0:
                if train_writer:
                    train_writer.close()
                # 出力ファイル名を指定の形式に
                output_path = os.path.join(FLAGS.output_dir, f"training_seqs.tfrecord-{train_shard_index:05d}-of-{num_train_shards:05d}")
                train_writer = tf.io.TFRecordWriter(output_path)
                train_shard_index += 1
                logging.info(f"Writing to training shard: {output_path}")

            try:
                record = next(dataset_iterator)
                train_writer.write(record.numpy())
                train_record_count += 1
            except StopIteration:
                logging.warning("Ran out of records before writing all expected training records.")
                break
            if (i + 1) % (FLAGS.shard_size // 10 or 1) == 0 :
                 logging.info(f"  Wrote {i+1}/{num_train_records} training records...")

        if train_writer:
            train_writer.close()
        logging.info(f"Finished writing {train_record_count} training records.")
    else:
        logging.info("No training records to write based on the split ratio.")


    # Evaluation データの書き込み
    if num_eval_records > 0: # evalデータがある場合のみ処理
        logging.info("Writing evaluation data...")
        for i in range(num_eval_records):
            if eval_record_count % FLAGS.shard_size == 0:
                if eval_writer:
                    eval_writer.close()
                # 出力ファイル名を指定の形式に
                output_path = os.path.join(FLAGS.output_dir, f"eval_seqs.tfrecord-{eval_shard_index:05d}-of-{num_eval_shards:05d}")
                eval_writer = tf.io.TFRecordWriter(output_path)
                eval_shard_index += 1
                logging.info(f"Writing to evaluation shard: {output_path}")

            try:
                record = next(dataset_iterator)
                eval_writer.write(record.numpy())
                eval_record_count += 1
            except StopIteration:
                logging.info("Reached end of dataset while writing evaluation records.")
                break
            if (i + 1) % (FLAGS.shard_size // 10 or 1) == 0 :
                 logging.info(f"  Wrote {i+1}/{num_eval_records} evaluation records...")

        if eval_writer:
            eval_writer.close()
        logging.info(f"Finished writing {eval_record_count} evaluation records.")
    else:
        logging.info("No evaluation records to write based on the split ratio.")

    logging.info(f"Splitting complete. Output files are in {FLAGS.output_dir}")

if __name__ == '__main__':
    flags.mark_flag_as_required('input_dir')
    flags.mark_flag_as_required('output_dir')
    app.run(main)