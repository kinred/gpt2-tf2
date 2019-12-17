import click
import glob
from gpt2_model import *
from data_pipeline import input_fn
import tensorflow as tf
import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = _ROOT + "/log"


@click.command()
@click.option('--model-dir', type=str, default="./model", show_default=True, help="Directory to load model")
@click.option('--data-dir', type=str, default="./data", show_default=True, help="training data directory")
@click.option('--batch-size', type=int, default=16, show_default=True, help="batch size")
@click.option('--learning-rate', type=float, default=0.001, show_default=True, help="learning rate")
@click.option('--distributed', type=bool, default=False, show_default=True, help="distributed training")
@click.option('--mxp', type=bool, default=False, show_default=True, help="enable mixed precission training")
def train(model_dir, data_dir, batch_size=16, learning_rate=0.001, distributed=False, mxp=False):
    data_dir = os.path.abspath(data_dir)
    model_dir = os.path.abspath(model_dir)
    tf_records = glob.glob(data_dir + "/tf_records/*.tfrecord")
    dataset = None
    if distributed:
        dataset = input_fn(tf_records, batch_size=batch_size)
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
        dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
        with mirrored_strategy.scope():
            model = Gpt2.create_from_params(model_dir)
            model.creat_optimizer(learning_rate=learning_rate, mixed_precission=mxp)
            model.create_checkpoint_manager(model_dir)
            model.create_summary_writer(LOG_DIR)

        model.mirrored_strategy = mirrored_strategy
    else:
        dataset = input_fn(tf_records, batch_size=batch_size)
        model = Gpt2.create_from_params(model_dir)
        model.create_optimizer(learning_rate=learning_rate, mixed_precission=mxp)
        model.create_checkpoint_manager(model_dir)
        model.create_summary_writer(LOG_DIR)
    print("Trainign Model...............")
    model.print_params()
    model.fit(dataset)
    print("Training Done................")


if __name__ == "__main__":
    train()
