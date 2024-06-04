import argparse
import logging
import numpy as np
from tqdm import tqdm

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


def main() -> None:
    logging.basicConfig(
        filename="log.log",
        format="%(asctime)s %(message)s",
        filemode="w",
        level=logging.DEBUG,
    )
    ops = options()
    trainer = Trainer(ops.i, int(ops.b))
    trainer.train()
    # print the weights after training
    print("w", trainer.model.net[0].weight)
    print("b", trainer.model.net[0].bias)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", help="Input filename with features and labels", default="data.npz")
    parser.add_argument("-b", help="Batch size", default=32)
    return parser.parse_args()

class Trainer:
    def __init__(self, input: str, batch_size: int) -> None:
        self.model = LayerCalibration(input, batch_size)
        self.model.to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=0.01)
        self.n_epochs = 10
        self.wandb = "weights_and_bias.npz"
        print(self.model.try_summing())
        print(f"N(parameters): {sum(p.numel() for p in self.model.parameters())}")

    def train(self) -> None:
        iter_per_epoch = self.model.n_iter_per_epoch()
        weights_vs_time = torch.zeros(self.n_epochs * iter_per_epoch, self.model.n_layers(), device=device)
        bias_vs_time = torch.zeros(self.n_epochs * iter_per_epoch, device=device)
        for i_epoch in tqdm(range(self.n_epochs)):
            self.model.train()
            for i_batch, (x, y) in enumerate(self.model.data_train):
                with torch.no_grad():
                    weights_vs_time[i_epoch * iter_per_epoch + i_batch] = self.model.net[0].weight[0].flatten()
                    bias_vs_time[i_epoch * iter_per_epoch + i_batch] = self.model.net[0].bias.flatten()
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
            print(f"Train loss = {train_loss:.1f}, dev loss = {dev_loss:.1f}")
        print(f"Writing weights and bias to npz file")
        np.savez(self.wandb, weights_vs_time=weights_vs_time.cpu().numpy(), bias_vs_time=bias_vs_time.cpu().numpy())
        print(weights_vs_time)
        print(bias_vs_time)

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

    def __init__(self, input: str, batch_size: int) -> None:
        super().__init__()
        self.input = input
        self.batch_size = batch_size
        self.load_data()
        dropout = 0.1
        layers = self.n_layers()
        explode = 4
        # how can I initialize weights to 1?
        self.net = nn.Sequential(
            # nn.Linear(layers, layers * explode),
            # nn.ReLU(),
            # nn.Linear(layers * explode, layers * explode),
            # nn.ReLU(),
            # nn.Linear(layers * explode, 1),
            nn.Linear(layers, 1),
        )
        logger.info("Net:")
        logger.info(self.net)

    def n_iter_per_epoch(self) -> int:
        return len(self.data_train)

    def n_layers(self) -> int:
        features, label = self.dataset_train[0]
        return features.shape[0]

    def load_data(self) -> None:
        logger.info("Loading data ...")
        inp = np.load(self.input)
        features = torch.tensor(inp["features"]).float()
        labels = torch.tensor(inp["labels"]).reshape(-1, 1).float()
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

    def forward(self, x):
        return self.net(x.sum(axis=(2, 3)))

    def try_summing(self) -> float:
        with torch.no_grad():
            # get the difference between the sum of the x_train and the y_train
            mse = nn.MSELoss()
            return mse(self.x_train.sum(axis=(1, 2, 3)).flatten(), self.y_train.flatten())


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

