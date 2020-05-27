import json
import requests
from . import config


def score_batch(batch_ids, server_url='http://localhost:8501'):
    batch_json = json.dumps({
        'signature_name': 'serving_default',
        'instances': batch_ids.tolist()
    })
    response = requests.post(
        url=server_url + '/v1/models/hate:predict',
        data=batch_json
    )
    response.raise_for_status()
    return response.json()['predictions']