from types import SimpleNamespace
import numpy as np
from hate import config
from .tokenizer import tokenizer
from .model import model


def classify(texts):
    num_batches = int(np.ceil(len(texts) / config.batch_size))
    result = []
    for batch in np.array_split(texts, num_batches):
        result += classify_batch(batch)
    return result


def classify_batch(batch):
    model_input = tokenize_batch(batch)
    scores = model([model_input.ids, model_input.attentions]).numpy()
    return [score[0] < score[1] for score in scores]


def tokenize_batch(batch):
    encoded = [tokenizer.encode_plus(t, max_length=config.max_sequence_length, pad_to_max_length=True) for t in batch]
    input_ids = [t['input_ids'] for t in encoded]
    attention_masks = [t['attention_mask'] for t in encoded]
    return SimpleNamespace(ids=np.array(input_ids), attentions=np.array(attention_masks))