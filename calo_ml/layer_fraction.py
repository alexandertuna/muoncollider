"""
A script to plot the fraction of energy deposited
  at each layer of the ecal.
"""

import argparse
import glob
import logging
import numpy as np
from tqdm import tqdm
from typing import List

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

E_MIN = 50
E_MAX = 500
CMAP = "hot"

def main() -> None:
    logging.basicConfig(
        filename="log.log",
        format="%(asctime)s %(message)s",
        filemode="w",
        level=logging.DEBUG,
    )
    ops = options()
    files = get_files(ops.i, int(ops.n))
    plotter = Plotter(files, "layer_fraction.pdf")
    plotter.plot()


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", help="Input file(s) with features and labels. Wildcards allowed", default="/work/tuna/data/2024_05_31/*npz")
    parser.add_argument("-n", help="Maximum number of input files to consider", default=0)
    return parser.parse_args()


class Plotter:
    def __init__(self, input: str, output: str) -> None:
        self.input = input
        self.output = output
        logger.info(f"Loading data ...")
        self.features = load_key(self.input, "features")
        self.labels = load_key(self.input, "labels")
        self.n_layers = self.features.shape[-1]
        logger.info(f"Features: {self.features.shape}")

    def plot(self) -> None:
        with PdfPages(self.output) as pdf:
            self.plot_1d_average(pdf)
            self.plot_2d(pdf)

    def plot_1d_average(self, pdf: PdfPages) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        print(self.features.shape)
        sums = self.features.sum(axis=0)
        hist, edges, patches = ax.hist(
            x=np.arange(len(sums)),
            weights=sums,
            bins=np.arange(len(sums)),
            density=True,
        )
        ax.tick_params(top=True, right=True)
        ax.set_xlabel("Layer number")
        ax.set_ylabel("Fraction of energy")
        ax.grid()
        ax.set_axisbelow(True)
        msg = f"Photon gun, {E_MIN}-{E_MAX} GeV, (theta, phi) = (20, 0)"
        ax.text(0.02, 1.02, msg, transform=ax.transAxes)
        fig.subplots_adjust(bottom=0.12, left=0.15, right=0.95, top=0.95)
        pdf.savefig()
        plt.close()

    def plot_2d(self, pdf: PdfPages) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))

        # 2d plot of fraction of energy in each layer vs. photon energy
        import time
        data = np.zeros((E_MAX, self.n_layers))
        t_0 = time.time()
        for i in range(len(self.features)):
            energy = int(self.labels[i])
            for j in range(self.n_layers):
                data[energy, j] += self.features[i, j] / energy
        t_1 = time.time()
        print(f"t(old) = {t_1 - t_0}")

        data = np.zeros((E_MAX, self.n_layers))
        t_0 = time.time()
        energy_indices = self.labels.astype(int)
        divided_features = self.features / energy_indices[:, np.newaxis]
        np.add.at(data, (energy_indices[:, np.newaxis], np.arange(self.n_layers)), divided_features)
        t_1 = time.time()
        print(f"t(new) = {t_1 - t_0}")

        data = data[E_MIN:E_MAX]
        

        # merge every 10 steps of energy, and normalize rows
        bin_width = 5
        data = data.reshape(-1, bin_width, self.n_layers).sum(axis=1)
        data /= data.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(data, origin="lower", aspect="auto", extent=[0, self.n_layers, E_MIN, E_MAX])
        ax.axvline(x=15, color="gray", linestyle="--")
        ax.tick_params(top=True, right=True)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Fraction of energy in this layer")
        ax.set_xlabel("Layer number")
        ax.set_ylabel("Photon energy [GeV]")
        msg = f"Photon gun, {E_MIN}-{E_MAX} GeV, (theta, phi) = (20, 0)"
        ax.text(0.02, 1.02, msg, transform=ax.transAxes)
        fig.subplots_adjust(bottom=0.12, left=0.15, right=0.92, top=0.95)
        pdf.savefig()
        plt.close()


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
            return fi["features"].sum(axis=(2, 3))
        elif key == "labels":
            return fi[key].flatten()
        raise Exception(f"Unknown key: {key}")



if __name__ == "__main__":
    main()

