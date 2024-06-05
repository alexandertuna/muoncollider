import argparse
import logging
import numpy as np
from tqdm import tqdm

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
    no_calib = NoCalibration(ops.i)
    # no_calib.plot()
    global_calib = GlobalCalibration(ops.i)
    trainer = Trainer(ops.i, int(ops.b), int(ops.e))
    trainer.train()
    trainer.plot_vs_time()


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", help="Input filename with features and labels", default="data.npz")
    parser.add_argument("-b", help="Batch size", default=32)
    parser.add_argument("-e", help="Epochs for training", default=16)
    return parser.parse_args()


class GlobalCalibration:
    def __init__(self, input: str) -> None:
        self.input = input
        self.load_data()
        self.train()

    def load_data(self) -> None:
        logger.info("Loading data ...")
        inp = np.load(self.input)
        self.features = inp["features"].sum(axis=(1, 2, 3)).flatten()
        self.labels = inp["labels"].flatten()
        logger.info(f"Features: {self.features.shape}")
        logger.info(f"Labels: {self.labels.shape}")

    def train(self) -> None:
        alpha = np.linspace(0.5, 1.5, 201)
        beta = np.linspace(-20, 20, 201)
        features = self.features[..., np.newaxis, np.newaxis]
        labels = self.labels[..., np.newaxis, np.newaxis]
        alpha2d, beta2d = np.meshgrid(alpha, beta)
        features2d = features * alpha2d + beta2d
        mse = ((features2d - labels) ** 2).mean(axis=0)

        the_min = np.unravel_index(mse.argmin(), mse.shape)
        mse_min = mse[the_min]
        alpha_min = alpha2d[the_min]
        beta_min = beta2d[the_min]

        fig, ax = plt.subplots()
        hist, xedges, yedges, im = ax.hist2d(
            x=alpha2d.flatten(),
            y=beta2d.flatten(),
            weights=mse.flatten(),
            bins=[alpha, beta],
            cmap="viridis",
            norm="log",
        )
        ax.scatter([alpha_min], [beta_min], s=100, marker="o", facecolors='none', edgecolors='r')
        cbar = fig.colorbar(im, ax=ax)
        ax.set_xlabel("alpha")
        ax.set_ylabel("beta")
        cbar.set_label("Mean squared error [GeV^2]")
        plt.savefig("global_calibration.pdf")

    def plot(self) -> None:
        pl = Plotter("global_calibration.pdf", self.labels, self.features)
        pl.plot()


class NoCalibration:
    def __init__(self, input: str) -> None:
        self.input = input
        self.load_data()
        self.train()

    def load_data(self) -> None:
        logger.info("Loading data ...")
        inp = np.load(self.input)
        self.features = inp["features"].sum(axis=(1, 2, 3)).flatten()
        self.labels = inp["labels"].flatten()
        logger.info(f"Features: {self.features.shape}")
        logger.info(f"Labels: {self.labels.shape}")

    def train(self) -> None:
        diff = self.features - self.labels
        mse = (diff ** 2).mean()
        logger.info(f"Mean squared error: {mse}")

    def plot(self) -> None:
        pl = Plotter("no_calibration.pdf", self.labels, self.features)
        pl.plot()




class Plotter:
    def __init__(self, output: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self.output = output
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_diff = y_true - y_pred

    def plot(self) -> None:
        with PdfPages(self.output) as pdf:
            self.plot_truth_vs_reco(pdf)
            self.plot_truth_vs_squared_error(pdf)

    def plot_truth_vs_reco(self, pdf: PdfPages) -> None:
        fig, ax = plt.subplots()
        bins = np.linspace(0, 550, 100)
        hist, xedges, yedges, im = ax.hist2d(self.y_true, self.y_pred, bins=[bins, bins], cmin=0.5, cmap="viridis")
        cbar = fig.colorbar(im, ax=ax)
        ax.plot([0, 550], [0, 550], color="red", linestyle="--", linewidth=1)
        ax.grid()
        ax.set_axisbelow(True)
        ax.set_xlabel("Truth energy")
        ax.set_ylabel("Reconstructed energy")
        pdf.savefig()
        plt.close()

    def plot_truth_vs_squared_error(self, pdf: PdfPages) -> None:
        fig, ax = plt.subplots()
        bins = np.linspace(0, 550, 50), np.linspace(0, 200, 50)
        hist, xedges, yedges, im = ax.hist2d(self.y_true, self.y_diff ** 2, bins=bins, cmin=-0.1, cmap="viridis")
        cbar = fig.colorbar(im, ax=ax)
        ax.plot(bins[0][:-1], np.quantile(hist, 0.9, axis=1), color="red", marker="_", linestyle="None")
        ax.grid()
        ax.set_axisbelow(True)
        ax.set_xlabel("Truth energy")
        ax.set_ylabel("(Truth - Reconstructed energy) ^ 2")
        pdf.savefig()
        plt.close()


class Trainer:
    def __init__(self, input: str, batch_size: int, epochs: int) -> None:
        self.model = LayerCalibration(input, batch_size)
        self.model.to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=0.001)
        self.n_epochs = epochs
        self.wandb = "weights_and_bias.npz"
        logger.info(f"N(parameters): {sum(p.numel() for p in self.model.parameters())}")

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
            logger.info(f"Train loss = {train_loss:.1f}, dev loss = {dev_loss:.1f}")
        logger.info(f"Writing weights and bias to npz file")
        np.savez(self.wandb, weights_vs_time=weights_vs_time.cpu().numpy(), bias_vs_time=bias_vs_time.cpu().numpy())
        logger.info(weights_vs_time)
        logger.info(bias_vs_time)

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

    def plot_vs_time(self) -> None:
        import matplotlib.pyplot as plt
        data = np.load(self.wandb)
        weights_vs_time = data["weights_vs_time"]
        bias_vs_time = data["bias_vs_time"]
        comb_vs_time = np.concatenate([weights_vs_time, bias_vs_time.reshape(-1, 1)], axis=1)
        if False:
            fig, ax = plt.subplots()
            vals = comb_vs_time[-1]
            ax.scatter(x=np.arange(len(vals)), y=vals, color='green', marker='o', linestyle="")
            ax.set_xlabel(f"Weight index (bias is the last one)")
            ax.set_ylabel(f"Value")
            plt.savefig("weights_and_bias_vs_time.png")
        else:
            # draw animation
            fig, ax = plt.subplots()
            (line, ) = ax.plot([], [], color="green", marker="o", linestyle="")
            ax.grid()
            ax.set_axisbelow(True)
            ax.set_xlabel(f"Weight index (bias is the last one)")
            ax.set_ylabel(f"Value")
            ax.set_xlim(0, comb_vs_time.shape[1])
            ax.set_ylim(-0.2, 1.3)
            text = ax.text(0.0, 1.0, "Optimizer step 0", transform=ax.transAxes)
            speedup = 10
            def run(iteration):
                if iteration % 10 == 0:
                    logger.info(f"iteration = {iteration}")
                line.set_data(np.arange(len(comb_vs_time[iteration * speedup])), comb_vs_time[iteration * speedup])
                text.set_text(f"Optimizer step {iteration * speedup}")
                return (line, )
            ani = animation.FuncAnimation(fig, run, frames=len(comb_vs_time)//speedup, blit=True)
            ani.save("weights_and_bias_vs_time.gif", writer=animation.PillowWriter(fps=30))



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

