import click
import glob
from gpt2_model import *
import tensorflow as tf
import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = _ROOT + "/log"


@click.command()
@click.option('--model-dir', type=str, default="./model", show_default=True, help="Directory to store created model")
@click.option('--num-layers', type=int, default=8, show_default=True, help="No. of decoder layers")
@click.option('--embedding-size', type=int, default=768, show_default=True, help="Embedding size")
@click.option('--num-heads', type=int, default=8, show_default=True, help="Number of heads")
@click.option('--dff', type=int, default=3072, show_default=True, help="Filter Size")
@click.option('--max-seq-len', type=int, default=515, show_default=True, help="Seq length")
@click.option('--vocab-size', type=int, default=50000, show_default=True, help="Vocab size")
@click.option('--optimizer', type=str, default="adam", show_default=True, help="optimizer type")
def create_model(model_dir, num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
          optimizer="adam"):

    if not os.path.exists(model_dir):
        print("model dir does not exist, creating it.")
        os.mkdir(model_dir)

    model = Gpt2(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
                 optimizer=optimizer)
    model.create_optimizer()
    model.save_params(model_dir)
    print("\nCreated GPT2 Model, with following params:\n")
    model.print_params()

if __name__ == "__main__":
    create_model()
