import time
import os
import pickle

from src.data.bengali_data import build_data_loader
from dotenv import find_dotenv, load_dotenv
from src.config.config import cfg

load_dotenv(find_dotenv())

PATH_DATA_INTERIM = os.getenv("PATH_DATA_INTERIM")

def benchmark_data_loading(train: bool = False, iterations: int = 100):
    """
    Benchmark how long it takes to load some validation data.
    :param train:  whether use training data or validation data set.
    :param iterations: how many iteration of loading test to do.
    :return:
    """

    time_start = time.time()

    # Load pickle data?
    if train:
        train_data = pickle.load(open(os.path.join(PATH_DATA_INTERIM, 'train_data.p'), 'rb'))
    else:
        train_data = pickle.load(open(os.path.join(PATH_DATA_INTERIM, 'val_data.p'), 'rb'))
    time_end = time.time()
    print(f"Total time it took to load the train/val data and labels: {time_end - time_start}")

    # Build the Pytorch data loader based on the YAML configuraiton.
    data_loader = build_data_loader(train_data, cfg.DATASET, True)

    iterator = iter(data_loader)

    # Number of frames to load for the benchmark.
    num_frames = iterations

    time_start = time.time()
    for i in range(num_frames):
        frames, labels = next(iterator)
        print(frames)
        print(labels)
    time_end = time.time()

    print(f"Total time it took to load {num_frames} frames and labels: {time_end - time_start}")

if __name__=="__main__":
    benchmark_data_loading(train=True, iterations=5)