import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from google.cloud import storage


# connect to TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')  
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

# get data from storage bucket
client = storage.Client()
bucket = client.get_bucket('gideon116')

src_test0 = bucket.get_blob('MIT_mixed_augm/src-test.txt')
tgt_test0 = bucket.get_blob('MIT_mixed_augm/tgt-test.txt')
src_valid0 = bucket.get_blob('MIT_mixed_augm/src-val.txt')
tgt_valid0 = bucket.get_blob('MIT_mixed_augm/tgt-val.txt')
src_train0 = bucket.get_blob('MIT_mixed_augm/src-train.txt')
tgt_train0 = bucket.get_blob('MIT_mixed_augm/tgt-train.txt')

# read text files. SRC files are reactants and TGT files are products
with src_test0.open("rt") as f:
    src_test = f.read()

with tgt_test0.open("rt") as f:
    tgt_test = f.read()

with src_valid0.open("rt") as f:
    src_valid = f.read()

with tgt_valid0.open("rt") as f:
    tgt_valid = f.read()

with src_train0.open("rt") as f:
    src_train = f.read()

with tgt_train0.open("rt") as f:
    tgt_train = f.read()

# make arrays with one reaction as one element
src_test2 = np.array(src_test.split("\n"))
tgt_test2 = np.array(tgt_test.split("\n"))
src_valid2 = np.array(src_valid.split("\n"))
tgt_valid2 = np.array(tgt_valid.split("\n"))
src_train2 = np.array(src_train.split("\n"))
tgt_train2 = np.array(tgt_train.split("\n"))

# add START and END tokens to the products
for index, i in enumerate(tgt_test2):
    i = "[start] " + i + " [end]"
    tgt_test2[index] = i

for index, i in enumerate(tgt_valid2):
    i = "[start] " + i + " [end]"
    tgt_valid2[index] = i

for index, i in enumerate(tgt_train2):
    i = "[start] " + i + " [end]"
    tgt_train2[index] = i

# vocab size is the number of different characters
# for example: (, ), C, [N], O
vocab_size = 285

# max len is the length of the longest reaction
max_len = 260

# convert characters to tokens and pad with 0s to get equal lengths
trial = np.concatenate((src_test2, src_valid2, src_train2, tgt_test2, tgt_valid2, tgt_train2), axis=0)
tokenizer = Tokenizer(num_words=vocab_size, filters='')
tokenizer.fit_on_texts(trial)
sequences = tokenizer.texts_to_sequences(trial)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

encoder_input_data = padded_sequences[len(src_test2)+len(src_valid2):len(src_test2)+len(src_valid2)+len(src_train2)]
decoder_input_data = padded_sequences[len(src_test2)+len(src_valid2)+len(src_train2)+len(tgt_test2)+len(tgt_valid2):]

# decoder_target_data is one step ahead of decoder_input_data. So it does not have the START token
decoder_target_data = np.copy(decoder_input_data)
decoder_target_data = np.roll(decoder_target_data, -1, axis=1)
decoder_target_data[:, -1] = 0

# CLEAR RAM
src_test2 = 0
tgt_test2 = 0
src_valid2 = 0
tgt_valid2 = 0
src_train2 = 0
tgt_train2 = 0
trial = 0
sequences = 0
padded_sequences = 0


# variables for use during inference
testing_data = encoder_input_data[:1]
testing_val_data = decoder_target_data[:1]
start_token = decoder_input_data[0][0]
end_token = decoder_input_data[0][-1]


# to train on TPU define model under stratagy scope
with strategy.scope():
    latent_dim = 1024

    num_encoder_tokens = 285 
    num_decoder_tokens = 285
    max_decoder_seq_length = 260

    # encoder inputs the data and outputs states
    encoder_inputs = keras.Input(shape=(None,))

    # embed layer for one-hot encodding and there are multiple LSTM layers
    encoder_embedding = keras.layers.Embedding(num_encoder_tokens, num_encoder_tokens,
                                               input_length=max_decoder_seq_length)(encoder_inputs)
    encoder_LSTM1 = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    encoder_outputs1, state_h1, state_c1 = encoder_LSTM1(encoder_embedding)
    encoder_LSTM2 = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    encoder_outputs2, state_h2, state_c2 = encoder_LSTM2(encoder_outputs1)
    encoder_LSTM3 = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    encoder_outputs3, state_h, state_c = encoder_LSTM3(encoder_outputs2)
    encoder_states = [state_h, state_c]

    # decoder
    decoder_inputs = keras.Input(shape=(None,))
    decoder_embedding = keras.layers.Embedding(num_encoder_tokens, num_decoder_tokens,
                                               input_length=max_decoder_seq_length)(decoder_inputs)
    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(decoder_embedding,
                                                                          initial_state=encoder_states)
    
    # output
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # define or load model
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # model = keras.models.load_model("EMBD_Seq2Seq2_checkpoint.h5")

# add checkpoints to save and replace models after every epoch
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath="EMBD_Seq2Seq2_checkpoint.h5",
                                                               save_freq="epoch")

model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=1024,
    epochs=100,
    validation_split=0.2,
    callbacks=[model_checkpoint_callback])

# save final model
model.save("EMBD_Seq2Seq2.h5")

# inference model

# inference encoder has the same input as model encoder
encoder_inputs = model.input[0] 

# model layer 6 is encoder_LSTM3 from model
encoder_outputs, state_h_enc, state_c_enc = model.layers[6].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

# inference decoder has the same input as model decoder
decoder_inputs = model.input[1]

# model layer 5 is decoder_embedding from model
decoder_lstm_input = model.layers[5](decoder_inputs)
decoder_state_input_h = keras.Input(shape=(latent_dim,))
decoder_state_input_c = keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# model layer 7 is decoder_LSTM from model
decoder_lstm = model.layers[7]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_lstm_input, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]

# model layer 8 is output_dense from model
output_dense = model.layers[8]
decoder_outputs = output_dense(decoder_outputs)
inference_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


def decode_sequence(input_reaction):
  
    states_value = encoder_model.predict(input_reaction)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = 1.0

    stop_condition = False
    result_prediction = []
    while not stop_condition:
        output_tokens, h, c = inference_model.predict([target_seq] + states_value)

        prediction = np.argmax(output_tokens[0, -1, :])
        result_prediction.append(prediction)

        if prediction == end_token or len(result_prediction) > max_decoder_seq_length:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = 1.0

        states_value = [h, c]
        
    return result_prediction

print(np.array(decode_sequence(encoder_input_data[:1])))
print(decoder_target_data[:1])
