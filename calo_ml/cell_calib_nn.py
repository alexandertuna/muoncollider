import argparse
import glob
import logging
import numpy as np
from tqdm import tqdm
from typing import List

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages

SEED = 42 # 1337
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

E_MAX = 500
CMAP = "hot"
LAYER_CHUNKS = 10

def main() -> None:
    logging.basicConfig(
        filename="log.log",
        format="%(asctime)s %(message)s",
        filemode="w",
        level=logging.DEBUG,
    )
    ops = options()
    files = get_files(ops.i, int(ops.n))
    layer_calib = Trainer(files, int(ops.b), int(ops.e))
    layer_calib.train()
    #layer_calib.plot()
    #layer_calib.plot_vs_time()


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", help="Input file(s) with features and labels. Wildcards allowed", default="/work/tuna/data/2024_05_31/*npz")
    parser.add_argument("-n", help="Maximum number of input files to consider", default=0)
    parser.add_argument("-b", help="Batch size", default=32)
    parser.add_argument("-e", help="Epochs for training", default=32)
    parser.add_argument("--ones", help="Initialize weights to one", action="store_true")
    return parser.parse_args()


def get_files(input: str, n: int) -> List[str]:
    files = glob.glob(input)
    if len(files) == 0:
        raise Exception(f"No files found: {input}")
    if n > 0:
        return files[:n]
    return files


def load_key(fnames: List[str], key: str) -> np.ndarray:
    return np.concatenate([load_key_one_file(fname, key) for fname in fnames])


def load_key_one_file(fname: str, key: str) -> np.ndarray:
    with np.load(fname) as fi:
        if key == "features":
            print(f"Loading {fname} ...")
            n_ev, n_layer, n_pix0, n_pix1 = fi["features"].shape
            if n_layer % LAYER_CHUNKS != 0:
                raise Exception(f"Layer count ({n_layer}) not divisible by {LAYER_CHUNKS}")
            return fi["features"].reshape(
                n_ev, n_layer // LAYER_CHUNKS, LAYER_CHUNKS, n_pix0, n_pix1
            ).sum(axis=2)
        elif key == "labels":
            return fi[key].flatten()
        raise Exception(f"Unknown key: {key}")


class Trainer:
    def __init__(self, input: str, batch_size: int, epochs: int) -> None:
        self.model = LayerCalibration(input, batch_size)
        self.model.to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.005, weight_decay=0.001)
        self.n_epochs = epochs
        self.n_frame_skip = 10
        self.n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"N(parameters): {self.n_params:_}")

    def train(self) -> None:
        for i_epoch in tqdm(range(self.n_epochs)):
            self.model.train()
            for i_batch, (x, y) in enumerate(self.model.data_train):
                x = x.to(device)
                y = y.to(device)
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
            self.model.eval()
            train_loss = self.evaluate_loss("train")
            dev_loss = self.evaluate_loss("dev")
            logger.info(f"Train loss = {train_loss:.1f}, dev loss = {dev_loss:.1f}")

    def evaluate_loss(self, name: str) -> None:
        assert name in ["train", "dev"]
        with torch.no_grad():
            loss = 0
            data = getattr(self.model, f"data_{name}")
            for x, y in data:
                x = x.to(device)
                y = y.to(device)
                y_pred = self.model(x)
                loss += self.criterion(y_pred, y)
            loss /= len(data)
            return loss


class LayerCalibration(nn.Module):

    def __init__(self, inputs: str, batch_size: int) -> None:
        super().__init__()
        self.inputs = inputs
        self.batch_size = batch_size
        self.load_data()
        # layers = self.n_layers()
        # pixels = self.n_pixels()
        # logger.info(f"Layers: {layers}, pixels: {pixels}")
        features = self.n_features()
        dropout = 0.01
        logger.info(f"Features: {features}")
        self.net = nn.Sequential(
            nn.Linear(features, features // 10),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features // 10, features // 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features // 100, 1),
        )
        logger.info("Net:")
        logger.info(self.net)

    def n_features(self) -> int:
        features, label = self.dataset_train[0]
        return features.shape[-1]

    #def n_pixels(self) -> int:
    #    features, label = self.dataset_train[0]
    #    return features.shape[-1]

    def load_data(self) -> None:
        logger.info("Loading data ...")
        features = torch.tensor(load_key(self.inputs, "features")).float().flatten(start_dim=1)
        labels = torch.tensor(load_key(self.inputs, "labels")).reshape(-1, 1).float()
        logger.info(f"Features: {features.shape}")
        logger.info(f"Labels: {labels.shape}")
        self.x_train, x_devtest = train_test_split(features, train_size=0.8, random_state=SEED)
        self.y_train, y_devtest = train_test_split(labels, train_size=0.8, random_state=SEED)
        self.x_dev, self.x_test = train_test_split(x_devtest, train_size=0.5, random_state=SEED)
        self.y_dev, self.y_test = train_test_split(y_devtest, train_size=0.5, random_state=SEED)
        logger.info(f"x (train): {self.x_train.shape}")
        logger.info(f"y (train): {self.y_train.shape}")
        logger.info(f"x (dev): {self.x_dev.shape}")
        logger.info(f"y (dev): {self.y_dev.shape}")
        logger.info(f"x (test): {self.x_test.shape}")
        logger.info(f"y (test): {self.y_test.shape}")
        self.dataset_train = ParticleDataset(self.x_train, self.y_train)
        self.dataset_dev = ParticleDataset(self.x_dev, self.y_dev)
        self.dataset_test = ParticleDataset(self.x_test, self.y_test)
        self.data_train = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)
        self.data_dev = DataLoader(self.dataset_dev, batch_size=self.batch_size, shuffle=True)
        self.data_test = DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=True)
        logger.info("Loaded data!")

    def forward(self, x):
        return self.net(x)


class ParticleDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        if len(self.features) != len(labels):
            raise Exception(f"Features and labels dont match: {self.features.shape} vs {self.labels.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ParticleDatasetIterable(IterableDataset):
    def __init__(self, data_location: str, shuffle: bool = True):
        self.data_location = data_location
        self.shuffle = shuffle

    def __iter__(self):
        paths = glob.glob(self.data_location)
        for path in paths:
            with np.load(path) as fi:
                features, labels = fi["features"], fi["labels"]
                its = range(len(features))
                if self.shuffle:
                    its = np.random.permutation(its)
                for it in its:
                    yield features[it], labels[it]


if __name__ == "__main__":
    main()

