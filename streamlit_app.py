# app.py

import tensorflow as tf
import numpy as np
import os
import streamlit as st
from tqdm import tqdm
# from scipy.io.wavfile import write
import mitdeeplearning as mdl

# Load data and build the model
songs = mdl.lab1.load_training_data()
songs_joined = "\n\n".join(songs)
vocab = sorted(set(songs_joined))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def vectorize_string(string):
    vectorized_output = np.array([char2idx[char] for char in string])
    return vectorized_output

vectorized_songs = vectorize_string(songs_joined)

# Define the model and load weights
def LSTM(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True,
    )

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        LSTM(rnn_units),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

vocab_size = len(vocab)
params = {
    "embedding_dim": 512,
    "rnn_units": 1024,
    "batch_size": 1
}
checkpoint_dir = './training_checkpoints'
model = build_model(vocab_size, params["embedding_dim"], params["rnn_units"], batch_size=params["batch_size"])
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

# Generate text function
def generate_text(model, start_string, generation_length=1000):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()
    tqdm._instances.clear()

    for i in tqdm(range(generation_length)):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

# Streamlit app
st.title('Music Generation App')

length = st.number_input('Enter length of generated text:', min_value=100, max_value=2000, step=100)

generated_text = generate_text(model, start_string="X", generation_length=length)

st.write('Generated Text:')
st.write(generated_text)

st.write('Generated Audio:')

generated_songs = mdl.lab1.extract_song_snippet(generated_text)
for i, song in enumerate(generated_songs):
    waveform = mdl.lab1.play_song(song)
    if waveform:
        st.audio(waveform.data, format='audio/wav', sample_rate=waveform.fs)
