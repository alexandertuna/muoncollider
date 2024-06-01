import argparse
import logging
import numpy as np

SEED = 1337
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def main():
    logging.basicConfig(
        filename="log.log",
        format="%(asctime)s %(message)s",
        filemode="w",
        level=logging.DEBUG,
    )
    ops = options()
    trainer = Trainer(ops.f, ops.l)
    trainer.train()


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-f", help="Input filename for features", default="/work/tuna/muoncollider/data.features.npy")
    parser.add_argument("-l", help="Input filename for labels", default="/work/tuna/muoncollider/data.labels.npy")
    return parser.parse_args()

class Trainer:
    def __init__(self, features: str, labels: str) -> None:
        self.model = LayerCalibration(features, labels)
        self.model.to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        self.n_epochs = 50

    def train(self) -> None:
        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch}")
            self.model.train()
            for i, (x, y) in enumerate(self.model.data_train):
                if i % 10 == 0:
                    print(f"Batch {i}")
                x = x.to(device)
                y = y.to(device)
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
            self.model.eval()
            self.evaluate_loss("train", self.model.data_train)
            self.evaluate_loss("dev", self.model.data_dev)

    def evaluate_loss(self, name: str, data: DataLoader) -> None:
        with torch.no_grad():
            loss = 0
            for x, y in data:
                x = x.to(device)
                y = y.to(device)
                y_pred = self.model(x)
                loss += self.criterion(y_pred, y)
            loss /= len(data)
            print(f"{name} loss: {loss}")

class LayerCalibration(nn.Module):

    def __init__(self, features: str, labels: str) -> None:
        super().__init__()
        self.filename_features = features
        self.filename_labels = labels
        self.load_data()
        layers = self.n_layers()
        self.net = nn.Sequential(
            nn.Linear(layers, layers),
            nn.ReLU(),
            nn.Linear(layers, 1),
            # nn.Dropout(dropout),
        )
        logger.info("Net:")
        logger.info(self.net)

    def n_layers(self) -> int:
        features, label = self.dataset_train[0]
        return features.shape[0]

    def load_data(self) -> None:
        logger.info("Loading data ...")
        features = torch.tensor(np.load(self.filename_features)).float()
        labels = torch.tensor(np.load(self.filename_labels)).reshape(-1, 1).float()
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
        self.data_train = DataLoader(self.dataset_train, batch_size=32, shuffle=True)
        self.data_dev = DataLoader(self.dataset_dev, batch_size=32, shuffle=True)
        self.data_test = DataLoader(self.dataset_test, batch_size=32, shuffle=True)

    def forward(self, x):
        #print("x.shape", x.shape)
        #print("x.sum(axis=2, 3).shape", x.sum(axis=(2, 3)).shape)
        #import time
        #time.sleep(1)
        return self.net(x.sum(axis=(2, 3)))

    def try_summing(self) -> None:
        diff = self.x_train.sum(axis=(1, 2, 3)) - self.y_train
        print(diff.shape)
        print(diff)
        # print(self.x_train[:events].sum(axis=(1, 2, 3)))
        # print(self.y_train[:events])


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


if __name__ == "__main__":
    main()

