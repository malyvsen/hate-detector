from types import SimpleNamespace
import numpy as np
from . import config
from .tokenizer import tokenizer
from .model import model
from . import client


def classify(texts, server=None, progress=lambda x: x):
    num_batches = int(np.ceil(len(texts) / config.batch_size))
    result = []
    for batch in progress(np.array_split(texts, num_batches)):
        result += classify_batch(batch, server=server)
    return result


def classify_batch(batch, server):
    batch_ids = tokenize_batch(batch)
    if server is None:
        scores = model(batch_ids).numpy()
    else:
        scores = client.score_batch(batch_ids, server)
    return [score[0] < score[1] for score in scores]


def tokenize_batch(batch):
    encoded = [
        tokenizer.encode_plus(
            t,
            max_length=config.max_sequence_length,
            pad_to_max_length=True,
            truncation=True
        )
        for t in batch
    ]
    input_ids = [t['input_ids'] for t in encoded]
    return np.array(input_ids)