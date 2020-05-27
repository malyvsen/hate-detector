import os
import transformers


model_config_path = os.path.dirname(__file__) + '/model/config.pkl'
model_weights_path = os.path.dirname(__file__) + '/model/weights.h5'

tokenizer_type = transformers.DistilBertTokenizer
nlp_model_type = transformers.TFDistilBertForSequenceClassification
pretrained_name = 'distilbert-base-uncased'

input_layer_name = 'input'
max_sequence_length = 256
batch_size = 64