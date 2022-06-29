import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from google.cloud import storage

print('############################# IMPORT COMPLETE')

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

print('############################# DATA OBTAINED')

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

print('############################# START END ADDED')

# group reactants and products
reactants = np.concatenate((src_train2, src_test2, src_valid2,), axis=0)
products = np.concatenate((tgt_train2, tgt_test2, tgt_valid2), axis=0)
total_reaction = np.concatenate((src_train2, src_test2, src_valid2, tgt_train2, tgt_test2, tgt_valid2), axis=0)

number_of_samples = len(src_test2) + len(src_valid2) + len(src_train2)
number_of_training_samples = len(src_train2) + len(src_test2)

# CLEAR RAM
src_test2 = 0
tgt_test2 = 0
src_valid2 = 0
tgt_valid2 = 0
src_train2 = 0
tgt_train2 = 0

# make train and val reactant-product sets
cat_train = []
cat_valid = []

for i in range(number_of_samples):
    if i <= number_of_training_samples:
        cat_train.append((reactants[i], products[i]))
    elif i > number_of_training_samples:
        cat_valid.append((reactants[i], products[i]))

print('############################# PAIRS CREATED')

# CLEAR RAM
reactants = 0
products = 0

# vocab size is the number of different characters
# for example: (, ), C, [N], O
vocab_size = 285

# sequence_length is the length of the longest reaction
sequence_length = 260
batch_size = 1024

cat_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length)

cat_vectorization.adapt(total_reaction)

print('############################# TOKENIZED')

# CLEAR RAM
total_reaction = 0

def format_dataset(reactant, product):
    encoder_input_data = cat_vectorization(reactant)
    decoder_input_data = cat_vectorization(product)
    
    # decoder_target_data is one step ahead of decoder_input_data. So it does not have the START token
    temp = tf.roll(decoder_input_data, -1, axis=1)
    decoder_target_data = temp - 1 * tf.roll(cat_vectorization("[start]"), -1, axis=0)

    return ({"encoder_inputs": encoder_input_data, "decoder_inputs": decoder_input_data}, decoder_target_data)


def make_dataset(reactant_product_pairs):
    reactant, product = zip(*reactant_product_pairs)
    reactant = list(reactant)
    product = list(product)

    dataset_pair = tf.data.Dataset.from_tensor_slices((reactant, product))
    dataset_pair = dataset_pair.batch(batch_size)
    dataset_pair = dataset_pair.map(format_dataset)
    return dataset_pair

# make tf datasets
cat_train_ds = make_dataset(cat_train)
cat_val_ds = make_dataset(cat_valid)

print('############################# DATASETS CREATED')

# define transformer encoder (from keras.org)
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config


# define positional embedder (from keras.org)
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config


# define transformer decoder (from keras.org)
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(latent_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "latent_dim": self.latent_dim,
            "num_heads": self.num_heads,
        })
        return config


# to train on TPU define model under stratagy scope
with strategy.scope():
    embed_dim = 285
    latent_dim = 2048
    num_heads = 16
    sequence_length = 260
    vocab_size = 285
    
    # encoder
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    encoder_embedding = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(encoder_embedding)
    encoder = keras.Model(encoder_inputs, encoder_outputs)

    # decoder
    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")

    decoder_embedding = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
    decoder_transformer = TransformerDecoder(embed_dim, latent_dim, num_heads)(decoder_embedding, encoded_seq_inputs)
    decoder_transformer = layers.Dropout(0.5)(decoder_transformer)
    decoder_outputs = layers.Dense(vocab_size, activation="softmax")(decoder_transformer)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)
    decoder_outputs = decoder([decoder_inputs, encoder_outputs])

    # define or load model
    transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    transformer.compile("rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # model = keras.models.load_model("transformer_checkpoint.h5")

print('############################# MODEL DEFINED')

# add checkpoints to save and replace models after every epoch
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="transformer_checkpoint.h5", save_freq="epoch")

transformer.fit(cat_train_ds, epochs=100, validation_data=cat_val_ds, callbacks=[model_checkpoint_callback])

# save final model
transformer.save('transformer.h5')

# inference
cat_vocab = cat_vectorization.get_vocabulary()
cat_index_lookup = dict(zip(range(len(cat_vocab)), cat_vocab))
max_product_length = 260

def decode_sequence(input_reaction):
    tokenized_input_reaction = cat_vectorization([input_reaction])
    result_prediction = "[start]"

    for i in range(max_product_length):
        tokenized_target_sentence = cat_vectorization([result_prediction])
        predictions = transformer([tokenized_input_reaction, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = cat_index_lookup[sampled_token_index]
        result_prediction += " " + sampled_token

        if sampled_token == "[end]":
            break
        
    return result_prediction

test_reactants = [pair[0] for pair in cat_train[:10]]
real_products = [pair[1] for pair in cat_train[:10]]

print("model output = ", decode_sequence(test_reactants[0]))
print("real value = ", real_products[0])

    
