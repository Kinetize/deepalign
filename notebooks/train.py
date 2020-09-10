import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH "] = "true"

import arrow
import tensorflow as tf
from tqdm.notebook import tqdm

from deepalign import Dataset
from deepalign import fs
from deepalign.alignments import ConfNet

already_done = [] # ['gigantic-0.0-1', 'gigantic-0.0-2', 'gigantic-0.0-3', 'gigantic-0.0-4', 'gigantic-0.1-1', 'gigantic-0.1-2']

datasets = sorted([f.name for f in fs.get_event_log_files() if f.name not in already_done])

import datetime

for dataset_name in datasets:
    print(f'Starting dataset {dataset_name} at {datetime.datetime.now().time()}')

    for ea, ca in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        start_time = arrow.now()
        dataset = Dataset(dataset_name, use_case_attributes=ca, use_event_attributes=ea)
        if ca and dataset.num_case_attributes == 0:
            continue
        confnet = ConfNet(dataset, use_case_attributes=ca, use_event_attributes=ea)
        confnet.fit(dataset, batch_size=100, epochs=50, validation_split=0.1,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])
        confnet.save(
            str(fs.MODEL_DIR / f'{dataset_name}_{confnet.identifier}_{start_time.format(fs.DATE_FORMAT)}'))

from deepalign.alignments.processmining import OptimalCostAligner
from deepalign.alignments.processmining import HeuristicsMinerAligner
from deepalign.alignments.processmining import InductiveMinerAligner

datasets = sorted([f.name for f in fs.get_event_log_files()])

aligners = [OptimalCostAligner, HeuristicsMinerAligner, InductiveMinerAligner]

for aligner_class in tqdm(aligners):
    for dataset_name in tqdm(datasets):
        dataset = Dataset(dataset_name)
        aligner = aligner_class()
        aligner.fit(dataset)
        file_name = f'{dataset_name}_{aligner.abbreviation}'
        aligner.save(file_name)
