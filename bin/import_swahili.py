#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import sys
import os
import pandas
sys.path.insert(1, os.path.join(sys.path[0], '..'))

def _download_and_preprocess_data(data_dir):
  overview_of_recordings = pandas.read_csv("data/swahili/overview-of-recordings.csv")
  overview_of_recordings.set_index("file_name", inplace=True)
  print("OVERVIEW HEAD")
  print(overview_of_recordings.head())

  print("Training data preparation")
  swahili_training = pandas.DataFrame()

  for filename in os.listdir(data_dir + "/recordings/train/"):
    if filename.endswith(".wav"):
      print(filename)
      recording_row = overview_of_recordings.loc[filename]
      recording_row[['_unit_id']] = os.path.getsize(data_dir + "/recordings/train/" + filename)
      swahili_training = swahili_training.append(recording_row[['_unit_id', 'sw_transcription']])
  swahili_training.reset_index(inplace=True)
  print("SWAHILI HEAD")
  print(swahili_training.head())
  swahili_training = swahili_training.rename(columns={"index": "wav_filename", "_unit_id": "wav_filesize", "sw_transcription": "transcript"}, errors="raise")
  swahili_training['wav_filename'] = swahili_training['wav_filename'].apply(lambda x: "recordings/train/" + x)
  swahili_training.to_csv(os.path.join(data_dir, "swahili_training.csv"), index=False)

  print("test data preparation")

if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1])







