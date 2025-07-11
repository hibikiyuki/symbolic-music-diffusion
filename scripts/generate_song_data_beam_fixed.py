# Copyright 2021 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
r"""Dataset generation."""

import pickle

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam.metrics import Metrics
# Make sure magenta and note_seq are importable
try:
    from magenta.models.music_vae import TrainedModel
    import note_seq
except ImportError:
    # Handle cases where magenta might not be directly on the python path
    # depending on installation method. You might need to adjust sys.path
    # or ensure magenta is properly installed in the environment.
    logging.error("Could not import magenta or note_seq. "
                  "Ensure Magenta is installed and accessible.")
    import sys
    sys.exit(1)

# Assuming config and utils are in the parent directory or accessible
# Use absolute import if running with -m, or adjust path if necessary
try:
  import config
  from utils import song_utils
except ImportError:
    # Fallback for direct script execution (might cause issues with relative imports)
    logging.warning("Could not perform relative import. "
                    "Trying direct import (may fail if not run with -m).")
    import config
    from utils import song_utils


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'pipeline_options', '--runner=DirectRunner',
    'Command line flags to use in constructing the Beam pipeline options.')

# Model - Default values might need adjustment based on your setup
# Consider removing defaults or making them more generic if loaded from config
flags.DEFINE_string('model', 'cat-mel_2bar_big', 'MusicVAE model configuration key (e.g., from config.py).')
flags.DEFINE_string('checkpoint', None, 'Path to the MusicVAE model checkpoint.')

# Data transformation
flags.DEFINE_enum('mode', 'melody', ['melody', 'multitrack'],
                  'Data generation mode.')
flags.DEFINE_string('input', None, 'Path or pattern to input TFRecord files (containing NoteSequences).')
flags.DEFINE_string('output', None, 'Output path for TFRecord files (containing pickled encoding matrices).')

# Add required flags check
flags.mark_flag_as_required('input')
flags.mark_flag_as_required('output')
flags.mark_flag_as_required('checkpoint')


class EncodeSong(beam.DoFn):
  """Encode song (NoteSequence protos) into MusicVAE embeddings (pickled numpy arrays)."""

  def setup(self):
    logging.info('Loading pre-trained model config: %s', FLAGS.model)
    # Ensure the model key exists in the config
    if FLAGS.model not in config.MUSIC_VAE_CONFIG:
        raise ValueError(f"Model configuration '{FLAGS.model}' not found in config.MUSIC_VAE_CONFIG.")
    self.model_config = config.MUSIC_VAE_CONFIG[FLAGS.model]

    logging.info('Loading model checkpoint from: %s', FLAGS.checkpoint)
    self.model = TrainedModel(self.model_config,
                              batch_size=1, # Process one NoteSequence at a time
                              checkpoint_dir_or_path=FLAGS.checkpoint)
    # Initialize counters here if needed, or rely on process method
    # self.skipped_long_counter = Metrics.counter('EncodeSong', 'skipped_long_song')
    # self.no_melodies_counter = Metrics.counter('EncodeSong', 'extracted_no_melodies')
    # ... etc

  def process(self, ns_proto):
    # Decode the NoteSequence proto
    try:
      ns = note_seq.NoteSequence.FromString(ns_proto)
    except Exception as e: # Catch potential decoding errors
      logging.error("Failed to decode NoteSequence proto: %s", e)
      Metrics.counter('EncodeSong', 'failed_decode').inc()
      return

    logging.debug('Processing %s::%s (%f)', ns.id, ns.filename, ns.total_time)
    if ns.total_time > 60 * 60:
      logging.info('Skipping notesequence with >1 hour duration: %s', ns.filename)
      Metrics.counter('EncodeSong', 'skipped_long_song').inc()
      return

    Metrics.counter('EncodeSong', 'attempted_encoding').inc()

    try:
      if FLAGS.mode == 'melody':
        # Default chunk length might be defined in config or constants
        chunk_length = self.model_config.hparams.max_seq_len // 16 # Example: typically 16 steps per bar
        melodies = song_utils.extract_melodies(ns)
        if not melodies:
          Metrics.counter('EncodeSong', 'extracted_no_melodies').inc()
          return
        Metrics.counter('EncodeSong', 'extracted_melody').inc(len(melodies))
        # Ensure data_converter is correctly accessed
        songs = [
            song_utils.Song(melody, self.model_config.data_converter,
                            chunk_length) for melody in melodies
        ]
        encoding_matrices = song_utils.encode_songs(self.model, songs)
      elif FLAGS.mode == 'multitrack':
        # Default chunk length might be defined in config or constants
        chunk_length = self.model_config.hparams.max_seq_len // 16 # Example
        song = song_utils.Song(ns,
                               self.model_config.data_converter,
                               chunk_length,
                               multitrack=True)
        encoding_matrices = song_utils.encode_songs(self.model, [song])
      else:
        # This case should ideally not be reached due to flags.DEFINE_enum
        raise ValueError(f'Unsupported mode: {FLAGS.mode}')

      # Check encoding results and yield pickled data
      for matrix in encoding_matrices:
        # Add more robust shape check if needed, based on model output
        if matrix is None or matrix.size == 0: # Check if encoding failed or result is empty
             Metrics.counter('EncodeSong', 'skipped_empty_matrix').inc()
             continue

        # Example shape check (adapt based on your model's expected output dimensions)
        expected_embedding_dim = self.model_config.hparams.z_size
        if len(matrix.shape) < 2 or matrix.shape[-1] != expected_embedding_dim:
            logging.warning(f"Unexpected matrix shape: {matrix.shape}, expected embedding dim: {expected_embedding_dim}")
            Metrics.counter('EncodeSong', 'skipped_invalid_shape').inc()
            continue

        if matrix.shape[1] == 0: # Check if sequence length is zero
          Metrics.counter('EncodeSong', 'skipped_zero_length_matrix').inc()
          continue

        Metrics.counter('EncodeSong', 'encoded_matrix_success').inc()
        yield pickle.dumps(matrix)

    except Exception as e:
        logging.error(f"Error encoding song {ns.filename}: {e}", exc_info=True)
        Metrics.counter('EncodeSong', 'encoding_error').inc()


def main(argv):
  del argv  # unused

  # Log the flags for debugging
  logging.info("Starting pipeline with flags:")
  logging.info("Pipeline Options: %s", FLAGS.pipeline_options)
  logging.info("Model: %s", FLAGS.model)
  logging.info("Checkpoint: %s", FLAGS.checkpoint)
  logging.info("Mode: %s", FLAGS.mode)
  logging.info("Input: %s", FLAGS.input)
  logging.info("Output: %s", FLAGS.output)


  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      FLAGS.pipeline_options.split(','))

  with beam.Pipeline(options=pipeline_options) as p:
    # Read NoteSequence protos directly from the input TFRecord(s)
    # FLAGS.input can be a single file path or a pattern like /path/to/data/*.tfrecord
    raw_notesequences = p | 'ReadInputTFRecord' >> beam.io.ReadFromTFRecord(
                                                      FLAGS.input)
                                                      # No need for coder here, reads raw bytes

    # Process the raw bytes (NoteSequence protos)
    encoded_data = (
        raw_notesequences
        # Decode NoteSequence proto (might be needed if EncodeSong expects objects,
        # but EncodeSong currently decodes internally. Keep as raw bytes.)
        # | 'DecodeProto' >> beam.Map(note_seq.NoteSequence.FromString) # Keep commented if EncodeSong handles bytes
        | 'shuffle_input' >> beam.Reshuffle()
        | 'encode_song' >> beam.ParDo(EncodeSong()) # Input is raw proto bytes, output is pickled bytes
        | 'shuffle_output' >> beam.Reshuffle()
    )

    # Write the pickled numpy arrays (bytes) to the output TFRecord
    _ = encoded_data | 'WriteOutputTFRecord' >> beam.io.WriteToTFRecord(
                                                    FLAGS.output)
                                                    # No coder needed, writing raw bytes

  logging.info("Pipeline finished.")


if __name__ == '__main__':
  # Ensure flags are parsed before app.run
  # flags.FLAGS(sys.argv) # This is usually handled by app.run
  app.run(main)