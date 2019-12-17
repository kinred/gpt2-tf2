import tensorflow as tf
from layers.feed_forward import Conv1d


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, att_dropout=0.1, residual_dropout=0.1, scale=True):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.att_dropout = att_dropout
        self.residual_dropout = residual_dropout
        self.scale = scale

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.c_attn = Conv1d(self.d_model, self.d_model * 3)
        self.c_proj = Conv1d(self.d_model, self.d_model)

    def attention_mask(nd, ns, *, dtype):
        """1's in the lower triangle, counting from the lower right corner.
        Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd)[:,None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    def shape_list(x):
        """Deal with dynamic shape in tensorflow cleanly."""
        static = x.shape.as_list()
        dynamic = tf.shape(x)
        return [dynamic[i] if s is None else s for i, s in enumerate(static)]

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = MultiHeadAttention.shape_list(w)
        b = MultiHeadAttention.attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attention(self, q, k, v, training, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        if self.scale:
            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            matmul_qk = matmul_qk / tf.math.sqrt(dk)

        #        if mask is not None:
        #            matmul_qk += (mask * -1e9)

        matmul_qk = MultiHeadAttention.mask_attn_weights(matmul_qk)

        attention_weights = tf.nn.softmax(matmul_qk, axis=-1)  # (..., seq_len_q, seq_len_k)

        if training:
            attention_weights = tf.nn.dropout(attention_weights, rate=self.att_dropout, name="attn_dropout")
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def merge_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, num_heads, depth)

        merged = tf.reshape(x, (batch_size, -1, self.d_model))
        # (batch_size, seq_len_q, d_model)
        return merged

    def call(self, x, mask=None, past_layer=None, training=True):
        x = self.c_attn(x)
        query, key, value = tf.split(x, 3, axis=2)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        present = tf.stack([key, value], axis=1)

        if past_layer is not None:
            past_key, past_value = tf.unstack(past_layer, axis=1)
            key = tf.concat([past_key, key], axis=-2)
            value = tf.concat([past_value, value], axis=-2)

        scaled_attention, attention_weights = self.multihead_attention(query, key, value, training, mask)

        concat_attention = self.merge_heads(scaled_attention)

        output = self.c_proj(concat_attention)  # (batch_size, seq_len_q, d_model)
        if training:
            output = tf.nn.dropout(output, rate=self.residual_dropout, name="resid_dropout")

        return output, present
