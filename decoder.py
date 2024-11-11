import tensorflow as tf

try: from transformer import TransformerBlock, PositionalEncoding
except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

########################################################################################

@tf.keras.saving.register_keras_serializable(package="MyLayers")
class RNNDecoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO:
        # Now we will define image and word embedding, decoder, and classification layers

        # Define feed forward layer to embed image features into a vector 
        # with the models hidden size
        self.dense = tf.keras.layers.Dense(self.hidden_size, activation = "tanh")
        # Define decoder layer:   
        self.decoder = tf.keras.layers.LSTM(self.hidden_size, return_sequences=True, return_state=True)
        self.wordembedding = tf.keras.layers.Embedding(input_dim = vocab_size + 1, output_dim = hidden_size, input_length = window_size)
        # Define classification layer:
        self.classifier = tf.keras.layers.Dense(self.vocab_size)
        
    def call(self, encoded_images, captions):
        """
        :param encoded_images: tensor of shape [BATCH_SIZE x 2048]
        :param captions: tensor of shape [BATCH_SIZE x WINDOW_SIZE]
        :return: batch logits of shape [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]
        """

        # TODO:
        # 1) Embed the encoded images into a vector of the correct dimension
        # 2) Pass your english sentence embeddings, and the image embeddings, to your decoder 
        # 3) Apply dense layer(s) to the decoder out to generate logits
        # img_embedding = self.dense(encoded_images)
        img_embedding = self.dense(encoded_images)
        # img_embedding = tf.expand_dims(img_embedding, 1)
        # _, h_img, c_img = self.image_lstm(img_embedding_expanded) 
        word_embedding = self.wordembedding(captions)
        # initial_state = [h_img, c_img]
        decoder_output, *decoder_state = self.decoder(word_embedding, initial_state=[img_embedding, tf.zeros_like(img_embedding)])
        logits = self.classifier(decoder_output)
        return logits

        # raise NotImplementedError("RNNDecoder Not Implemented Yet")

    def get_config(self):
        base_config = super().get_config()
        config = {k:getattr(self, k) for k in ["vocab_size", "hidden_size", "window_size"]}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

########################################################################################

@tf.keras.saving.register_keras_serializable(package="MyLayers")
class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO:
        # Now we will define image and word embedding, positional encoding, tramnsformer decoder, and classification layers

        # Define feed forward layer to embed image features into a vector 
        # with the models hidden size
        self.img_embedding = tf.keras.layers.Dense(hidden_size)
        self.word_embedding = tf.keras.layers.Embedding(input_dim = vocab_size + 1, output_dim = hidden_size, input_length = window_size)
        # Define positional encoding layer for language:
        self.positional_encoding = PositionalEncoding(vocab_size, hidden_size, window_size)

        # Define transformer decoder layer:
        self.transformer_block = TransformerBlock(hidden_size)

        # Define classification layer
        self.classifier = tf.keras.layers.Dense(vocab_size)

    def call(self, encoded_images, captions):
        """
        :param encoded_images: tensor of shape [BATCH_SIZE x 1 x 2048]
        :param captions: tensor of shape [BATCH_SIZE x WINDOW_SIZE]
        :return: batch logits of shape [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]
        """
        # TODO:
        # 1) Embed the encoded images into a vector of the correct dimension
        # 2) Pass the captions through your word embedding layer
        # 3) Add positional embeddings to the word embeddings
        # 4) Pass the english embeddings and the image sequences, to the decoder
        # 5) Apply dense layer(s) to the decoder out to generate **logits**
        img_embedding = self.img_embedding(encoded_images)
        img_embedding = tf.expand_dims(img_embedding, axis = 1)
        word_embedding = self.word_embedding(captions)
        positional_encoding = self.positional_encoding(captions)
        word_embedding += positional_encoding
        logits = self.transformer_block(inputs = word_embedding, context_sequence = img_embedding)
        output = self.classifier(logits)
        return output
        # raise NotImplementedError("TransformerDecoder Not Implemented Yet")

    def get_config(self):
        base_config = super().get_config()
        config = {k:getattr(self, k) for k in ["vocab_size", "hidden_size", "window_size"]}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)    
