import tensorflow as tf
import numpy as np
from ftfy import fix_text
import sentencepiece as spm
from collections import Counter
import os
import datetime
import click
import tqdm
import glob
import csv

_ROOT = os.path.abspath(os.path.dirname(__file__))
PROCESS_DATA_FILE_NAME = "processed.txt"
BPE_TSV_FILE_NAME = "bpe_spm.tsv"
BPE_MODEL_FILE_NAME = "bpe_model"
TF_RECORDS_DIR = "tf_records/"
BOS_ID = 3
EOS_ID = 4


def process_text(text_files, data_dir):
    print("Pre-processing the text data.....")
    process_data_file = data_dir + "/" + PROCESS_DATA_FILE_NAME
    file_writer = open(process_data_file, "w")
    for file_name in tqdm.tqdm(text_files):
        fr = open(file_name, 'r')
        file_writer.writelines([fix_text(line, normalization='NFKC') for line in fr.readlines()])
        fr.close
    file_writer.close()


def train_byte_pair_encoding(data_dir, vocab_size):
    print("Training BytePair encoding......")
    process_data_file = data_dir + "/" + PROCESS_DATA_FILE_NAME    
    bpe_tsv_file = data_dir + "/" + BPE_TSV_FILE_NAME
    bpe_model_file = data_dir + "/" + BPE_MODEL_FILE_NAME
    token_dict = Counter()
    with open(process_data_file, 'r') as fr:
        for line in tqdm.tqdm(fr):
            token_dict.update(line.split())

    with open(bpe_tsv_file, 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for word in token_dict:
            tsv_output.writerow([word, token_dict[word]])

    spmcmd = '--input={spm_input} --model_prefix={spm_model} --input_format=tsv --vocab_size={vocab_size} --user_defined_symbols=[SEP],[BOS],[EOS] --character_coverage=1.0 --hard_vocab_limit=true --model_type=bpe --pad_id=0 --unk_id=1 --bos_id=-1 --eos_id=-1 --pad_piece=[PAD] --unk_piece=[UNK]'.format(
        spm_input=bpe_tsv_file, spm_model=bpe_model_file, vocab_size=vocab_size)

    spm.SentencePieceTrainer.train(spmcmd)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(inputs, targets):
    feature = {
        'inputs': _int64_feature(inputs),
        'targets': _int64_feature(targets)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def create_tf_records(data_dir, min_seq_len, max_seq_len, per_file_limit=50000):
    print("Creating TF Records...............")
    s = spm.SentencePieceProcessor()
    process_data_file = data_dir + "/" + PROCESS_DATA_FILE_NAME    
    bpe_tsv_file = data_dir + "/" + BPE_TSV_FILE_NAME
    bpe_model_file = data_dir + "/" + BPE_MODEL_FILE_NAME + ".model"
    tf_records_dir = data_dir + "/" + TF_RECORDS_DIR
    s.Load(bpe_model_file)
    if not os.path.exists(tf_records_dir):
        os.makedirs(tf_records_dir)
    filename = tf_records_dir + str(datetime.datetime.now().timestamp()) + ".tfrecord"
    tf_writer = tf.io.TFRecordWriter(filename)
    doc_counts = 0
    line_to_long_count = 0
    with open(process_data_file, 'r') as f:
        para = ""
        for line in tqdm.tqdm(f):
            if len(s.encode_as_ids(line)) >= (max_seq_len - 4):
                print("line to long!")
                line_to_long_count += 1
            else:
                if len(para + line) > max_seq_len:
                    encoded_ids = s.encode_as_ids(para)                    
                    if max_seq_len > len(encoded_ids) > min_seq_len:
                        inputs = np.array([BOS_ID] + encoded_ids)
                        targets = np.array(encoded_ids + [EOS_ID])
                        example = serialize_example(inputs, targets)
                        tf_writer.write(example)
                        doc_counts += 1
                        para = ""
                    if doc_counts >= per_file_limit:
                        tf_writer.write(example)
                        doc_counts = 0
                        tf_writer.close()
                        filename = tf_records_dir + str(datetime.datetime.now().timestamp()) + ".tfrecord"
                        tf_writer = tf.io.TFRecordWriter(filename)
                else:
                    para += "\n" + line
    print("Lines to long: " + str(line_to_long_count))                

@click.command()
@click.option('--text-dir', type=str, default="/data/scraped", show_default=True, help="path to raw text files")
@click.option('--data-dir', type=str, default="/data", show_default=True, help="directory where training data will be stored")
@click.option('--vocab-size', type=int, default=50000, show_default=True, help="byte pair vocab size")
@click.option('--min-seq-len', type=int, default=15, show_default=True, help="minimum sequence length")
@click.option('--max-seq-len', type=int, default=512, show_default=True, help="minimum sequence length")
def pre_process(text_dir, data_dir, vocab_size, min_seq_len, max_seq_len):
    text_dir = os.path.abspath(text_dir)
    data_dir = os.path.abspath(data_dir)
    text_files = glob.glob((text_dir + "/*.txt"))
    process_text(text_files, data_dir)
    train_byte_pair_encoding(data_dir, vocab_size)
    create_tf_records(data_dir, min_seq_len, max_seq_len)


if __name__ == "__main__":
    pre_process()
