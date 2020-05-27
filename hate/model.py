import pickle
import tensorflow as tf
from hate import config

try:
    model_config = pickle.load(open(config.model_config_path, 'rb'))
    nlp_model = config.nlp_model_type(model_config)
    input_layer = tf.keras.layers.Input(shape=(config.max_sequence_length,), dtype=tf.int32, name=config.input_layer_name)
    output_placeholder = nlp_model(input_layer)[0]
    model = tf.keras.Model(inputs=[input_layer], outputs=[output_placeholder])
    model.load_weights(config.model_weights_path)
except OSError:
    model = None # client mode!