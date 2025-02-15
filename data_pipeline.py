import collections
import tensorflow as tf

PAD_ID = 0
UNKNOWN_ID = 1
START_ID = 3
END_ID = 4


def load_vocab(vocab_path):
    vocab = collections.OrderedDict()
    index = 0
    for line in open(vocab_path, 'r').read().splitlines():
        vocab[line.split()[0]] = index
        index += 1
    inv_vocab = {v: k for k, v in vocab.items()}
    return vocab, inv_vocab


def convert_by_vocab(vocab, items):
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def parse_example(serialized_example):
    data_fields = {
        "inputs": tf.io.VarLenFeature(tf.int64),
        "targets": tf.io.VarLenFeature(tf.int64)
    }
    parsed = tf.io.parse_single_example(serialized_example, data_fields)
    inputs = tf.sparse.to_dense(parsed["inputs"])
    targets = tf.sparse.to_dense(parsed["targets"])

    inputs = tf.cast(inputs, tf.int32)
    targets = tf.cast(targets, tf.int32)

    return inputs, targets


def input_fn(tf_records, batch_size=32, padded_shapes=([-1], [-1]), epoch=1, buffer_size=65536):
    if type(tf_records) is str:
        tf_records = [tf_records]
    dataset = tf.data.TFRecordDataset(tf_records, buffer_size=buffer_size)
#    dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.map(parse_example)
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    dataset = dataset.repeat(epoch)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
