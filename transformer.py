import math
import numpy as np
import tensorflow as tf

@tf.keras.saving.register_keras_serializable(package="transformer_layers")
class AttentionMatrix(tf.keras.layers.Layer):

    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask

    def call(self, inputs):
        """
        STUDENT MUST WRITE:

        This functions runs a single attention head.

        :param K: is [batch_size x window_size_keys x embedding_size]
        :param Q: is [batch_size x window_size_queries x embedding_size]
        :return: attention matrix
        """
        K, Q = inputs
        window_size_queries = Q.get_shape()[1]  # window size of queries
        window_size_keys    = K.get_shape()[1]  # window size of keys
        embedding_size_keys = K.get_shape()[2]

        mask = tf.convert_to_tensor(
            value=np.transpose(np.tril(np.ones((window_size_queries, window_size_keys)) * np.NINF, -1), (1, 0)),
            dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])

        # TODO:
        # 1) compute attention weights using queries and key matrices (if use_mask==True, then make sure to add the attention mask before softmax)
        scores = Q @ tf.transpose(K, perm = [0, 2, 1])
        # dk = Q.shape[-1]
        scores = scores / math.sqrt(embedding_size_keys)
        if self.use_mask:
            scores += atten_mask
        # 2) return the attention matrix
        attention_matrix = tf.nn.softmax(scores, axis=-1)
        return attention_matrix

        # Check lecture slides for how to compute self-attention
        # Remember:
        # - Q is [batch_size x window_size_queries x embedding_size]
        # - K is [batch_size x window_size_keys x embedding_size]
        # - Mask is [batch_size x window_size_queries x window_size_keys]

        # Here, queries are matmuled with the transpose of keys to produce for every query vector, weights per key vector.
        # This can be thought of as: for every query word, how much should I pay attention to the other words in this window?
        # Those weights are then used to create linear combinations of the corresponding values for each query.
        # Those queries will become the new embeddings.
        

        # raise NotImplementedError("AttentionMatrix Not Implemented Yet")

@tf.keras.saving.register_keras_serializable(package="transformer_layers")
class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention

        # TODO:
        # Initialize the weight matrices for K, V, and Q.
        # They should be able to produce a (batch_size, output_size) tensor
        # Hint: use self.add_weight(...) - refer to the handout for more information!
        self.W_k = self.add_weight(shape=(input_size, output_size), initializer='zeros', trainable=True, name="W_k")
        self.W_q = self.add_weight(shape=(input_size, output_size), initializer='zeros', trainable=True, name="W_q")
        self.W_v = self.add_weight(shape=(input_size, output_size), initializer='zeros', trainable=True, name="W_v")
        # Initialize the attention matrix.
        self.attention_matrix = AttentionMatrix(use_mask=self.use_mask)

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        STUDENT MUST WRITE:

        This functions runs a single attention head.

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """

        # TODO:
        # - Apply 3 matrices to turn inputs into keys, values, and queries. You will need to use tf.tensordot for this.
        # - Call AttentionMatrix with the keys and queries.
        # - Apply the attention matrix to the values.
        K = tf.matmul(inputs_for_keys, self.W_k)
        Q = tf.matmul(inputs_for_queries, self.W_q)
        V = tf.matmul(inputs_for_values, self.W_v)
        attention_weights = self.attention_matrix((K, Q))
        attention_output = tf.matmul(attention_weights, V)
        return attention_output

        # raise NotImplementedError("AttentionHead Not Implemented Yet") 

@tf.keras.saving.register_keras_serializable(package="transformer_layers")
class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, emb_sz, use_mask, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)
        self.emb_sz = emb_sz
        self.use_mask = use_mask
        # Initialize Attention Heads Here
        self.head = [
            AttentionHead(input_size = emb_sz, output_size = emb_sz // 3, is_self_attention=use_mask) for _ in range(3)
        ]
        self.dense = tf.keras.layers.Dense(self.emb_sz)

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        This functions runs a multiheaded attention layer.

        Requirements:
            - Create three different attention heads
            - Concatenate the outputs of these heads together
            - Apply a linear layer

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """
        key_head = self.head[0](inputs_for_keys, inputs_for_values, inputs_for_queries)
        value_head = self.head[1](inputs_for_keys, inputs_for_values, inputs_for_queries)
        query_head = self.head[2](inputs_for_keys, inputs_for_values, inputs_for_queries)
        all_head = tf.concat([key_head, value_head, query_head], axis = -1)
        output = self.dense(all_head)
        return output
        # raise NotImplementedError("MultiHeadedAttention Not Implemented Yet")

@tf.keras.saving.register_keras_serializable(package="transformer_layers")
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz, multiheaded=False, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        # TODO:
        # 1) Define the Feed Forward, self-attention, encoder-decoder-attention, and layer normalization layers
        # 2) Use multiheaded attention
        self.emb_sz = emb_sz
        self.multiheaded = multiheaded
        # self.encoder_self_attention = (
        #     MultiHeadedAttention(emb_sz=emb_sz, use_mask=False) if multiheaded else AttentionHead(input_size=emb_sz, output_size=emb_sz, is_self_attention=False)
        # )
        self.layer = 6
        self.self_attention = [
            MultiHeadedAttention(emb_sz=emb_sz, use_mask=True) \
            if multiheaded else AttentionHead(input_size=emb_sz, output_size=emb_sz, is_self_attention=True) \
            for _ in range(self.layer)
        ]
        self.cross_attention = [
            MultiHeadedAttention(emb_sz=emb_sz, use_mask=False) \
            if multiheaded else AttentionHead(input_size=emb_sz, output_size=emb_sz, is_self_attention=False)\
            for _ in range(self.layer)
        ]
        self.feed_forward = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(4 * emb_sz, activation='relu'),  # Expansion
                tf.keras.layers.Dense(emb_sz)  # Compression
            ]) for _ in range(self.layer)
        ]
        self.dropout1 = [tf.keras.layers.Dropout(0.2) for _ in range(self.layer)]
        self.dropout2 = [tf.keras.layers.Dropout(0.2) for _ in range(self.layer)]
        self.dropout3 = [tf.keras.layers.Dropout(0.2) for _ in range(self.layer)]
        self.layer_norm1 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(self.layer)]
        self.layer_norm2 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(self.layer)]
        self.layer_norm3 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(self.layer)]
        self.Linear = tf.keras.layers.Dense(emb_sz, activation = "relu")
        # self.softmax = tf.keras.

    @tf.function
    def call(self, inputs, context_sequence):
        """
        This functions calls a transformer block.

        :param inputs: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :return: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        """

        # TODO:
        # 1) compute MASKED attention on the inputs
        # 2) residual connection and layer normalization
        # 3) computed UNMASKED attention using context
        # 4) residual connection and layer normalization
        # 5) feed forward layer
        # 6) residual layer and layer normalization
        # 7) call relu and return tensor
        x = inputs
        for i in range(self.layer):
            curr_input1 = x
            x = self.self_attention[i](x, x, x)
            x = self.dropout1[i](x)
            x = self.layer_norm1[i](curr_input1 + x)
            curr_input2 = x
            x = self.cross_attention[i](context_sequence, context_sequence, x)
            x = self.dropout2[i](x)
            x = self.layer_norm2[i](curr_input2 + x)
            curr_input3 = x
            x = self.feed_forward[i](x)
            x = self.dropout3[i](x)
            x = self.layer_norm3[i](curr_input3 + x)
        x = self.Linear(x)
        return x


        # raise NotImplementedError("TransformerBlock Not Implemented Yet")


@tf.keras.saving.register_keras_serializable(package="transformer_layers", name="positional_encoding")
def positional_encoding(length, depth):
    ## TODO:
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates 
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis = -1)
    return tf.cast(pos_encoding, dtype=tf.float32)

    # raise NotImplementedError("positional_encoding Not Implemented Yet")


@tf.keras.saving.register_keras_serializable(package="transformer_layers")
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)

        ## Sinosoidal positional encoding: offset by varying sinosoidal frequencies.
        ## HINT: https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
        self.pos_encoding = positional_encoding(length=window_size, depth=embed_size)[tf.newaxis, :window_size, :]

    def call(self, x):
        ## TODO: Get embeddings and and scale them by sqrt of embedding size, and add positional encoding.
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        x += self.pos_encoding
        return x
        # raise NotImplementedError("PositionalEncoding Not Implemented Yet")
