"""
A script to preprocess particle gun data from a flat Pandas Dataframe
to a image-like numpy array.
"""

import argparse
import logging
import os
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.sparse import coo_matrix
from tqdm import tqdm

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

THETA = math.radians(20.0)


def main() -> None:
    ops = options()
    logging.basicConfig(filename='log.log', format='%(asctime)s %(message)s', filemode='w', level=logging.DEBUG)
    processor = ProcessFlatToImage(ops.i, ops.o)
    processor.load_data()
    processor.make_new_columns()
    processor.make_label_array()
    processor.make_image_array()
    processor.write_data()
    processor.make_image_plots(events=2)
    processor.make_diagnostic_plots(events=2)


class ProcessFlatToImage:
    
    def __init__(self, input: str, output: str) -> None:
        self.input = input
        self.output = output
        if not os.path.isfile(self.input):
            raise Exception(f"{self.input} does not exist")
        self.df = pd.DataFrame()
        self.features = np.array([])
        self.labels = np.array([])
        self.shape = (images.pixels, images.pixels)

    def load_data(self) -> None:
        logger.info("Loading dataframe ... ")
        self.df = pd.read_parquet(self.input)

    def make_new_columns(self) -> None:
        logger.info("Making new columns of data ... ")
        def cell_size(system: int) -> float:
            return cell_sizes.ecal if system in systems.ecal else cell_sizes.hcal
        cell_size = np.vectorize(cell_size)(self.df.hit_system)
        subtract_x = self.df.hit_z * np.tan(THETA)
        subtract_y = self.df.hit_z * 0.0
        self.df["hit_xp"] = np.rint(self.df.hit_x / cell_size)
        self.df["hit_yp"] = np.rint(self.df.hit_y / cell_size)
        self.df["hit_xpp"] = np.rint((self.df.hit_x - subtract_x) / cell_size)
        self.df["hit_ypp"] = np.rint((self.df.hit_y - subtract_y) / cell_size)
        self.df["hit_xppp"] = np.rint((self.df.hit_x - subtract_x) / cell_size + images.pixels // 2)
        self.df["hit_yppp"] = np.rint((self.df.hit_y - subtract_y) / cell_size + images.pixels // 2)

    def make_label_array(self) -> None:
        """ Group the rows by event, and pluck the truth energy of that event """
        logger.info("Making label array ... ")
        self.labels = self.df.groupby("event").first().truth_e

    def make_image_array(self) -> None:
        """ NB: this is only implemented for ecal currently """
        logger.info("Making image array ... ")
        events = len(np.unique(self.df.event))
        df = self.df[
            ((self.df.hit_system == systems.ecal_barrel) |
             (self.df.hit_system == systems.ecal_endcap)) &
            (self.df.hit_xppp >= 0) &
            (self.df.hit_yppp >= 0) &
            (self.df.hit_xppp < images.pixels) &
            (self.df.hit_yppp < images.pixels)
        ]
        def process_group(group):
            coo = coo_matrix((group["hit_e"], (group["hit_yppp"], group["hit_xppp"])), shape=self.shape)
            return coo.toarray()
        self.features = np.zeros(shape=(events, layers.ecal, images.pixels, images.pixels))
        for (event, layer), group in tqdm(df.groupby(["event", "hit_layer"])):
            self.features[event, layer] = process_group(group)

    def write_data(self) -> None:
        logger.info("Writing npz file ... ")
        np.savez_compressed(self.output + ".npz", features=self.features, labels=self.labels)

        logger.info("Writing npy files ... ")
        np.save(self.output + ".features.npy", self.features)
        np.save(self.output + ".labels.npy", self.labels)

    def make_diagnostic_plots(self, events: int) -> None:
        logger.info("Making diagnostic plots ... ")
        pdfname = self.output + ".diagnostic.pdf"
        df = self.df[
            ((self.df.hit_system == systems.ecal_barrel) |
             (self.df.hit_system == systems.ecal_endcap))
        ]
        with PdfPages(pdfname) as pdf:
            for i_event, event in df.groupby("event"):
                if i_event >= events:
                    break
                logger.info(f"Plotting event {i_event} ... ")
                self.make_event_plot(event, pdf)

    def make_event_plot(self, df: pd.DataFrame, pdf: PdfPages) -> None:
        for i_layer, layer in df.groupby("hit_layer"):
            if i_layer % 10 == 0:
                logger.info(f"Plotting layer {i_layer}... ")
            self.make_layer_plot(i_layer, layer, pdf)

    def make_layer_plot(self, layer: int, df: pd.DataFrame, pdf: PdfPages) -> None:
        energy = df.truth_e.min()
        nplots = len(plotting.cols)
        scat = [None]*nplots
        cbar = [None]*nplots
        fig, ax = plt.subplots(nrows=1, ncols=nplots, figsize=(5*nplots, 4))
        for i_ax, (x, y) in enumerate(plotting.cols):
            scat[i_ax] = ax[i_ax].scatter(
                df[x],
                df[y],
                c=np.log10(df.hit_e * plotting.GeV_to_MeV),
                vmin=plotting.vmin,
                vmax=plotting.vmax,
                cmap=plotting.cmap,
            )
            self.annotate_axis(ax[i_ax], i_ax, layer, energy)
            cbar[i_ax] = fig.colorbar(scat[i_ax], ax=ax[i_ax])
            cbar[i_ax].set_label("Energy [log$_{10}$(MeV)]")
            fig.subplots_adjust(bottom=0.14, left=0.05, right=0.96, top=0.94)
            fig.subplots_adjust(wspace=0.25)

        pdf.savefig()
        plt.close()

    def annotate_axis(self, ax, i_ax: int, layer: int, energy: float) -> None:
        ax.tick_params(right=True, top=True)
        ax.axis(plotting.axis[i_ax])
        ax.grid(True)
        ax.set_xlabel(f"x [{plotting.unit[i_ax]}]")
        ax.set_ylabel(f"y [{plotting.unit[i_ax]}]")
        ax.set_axisbelow(True)
        ax.text(0.06, 1.02, f"Layer {layer}", transform=ax.transAxes)
        ax.text(0.51, 1.02, f"E = {int(energy)} GeV", transform=ax.transAxes)

    def make_image_plots(self, events: int) -> None:
        logger.info("Making image plots ... ")
        pdfname = self.output + ".image.pdf"
        with PdfPages(pdfname) as pdf:
            for i_event in range(len(self.features)):
                if i_event >= events:
                    break
                logger.info(f"Plotting image {i_event} ... ")
                self.make_image_plot(i_event, pdf)

    def make_image_plot(self, i_event: int, pdf: PdfPages) -> None:
        for i_layer in range(len(self.features[i_event])):
            if i_layer % 10 == 0:
                logger.info(f"Plotting image layer {i_layer}... ")
            self.make_image_layer_plot(i_event, i_layer, pdf)

    def make_image_layer_plot(self, i_event: int, i_layer: int, pdf: PdfPages) -> None:
        energy = self.labels[i_event]
        arr = self.features[i_event, i_layer]
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(
            np.log10(np.where(arr > 0, arr * plotting.GeV_to_MeV, plotting.eps)),
            cmap=plotting.cmap,
            vmin=plotting.vmin,
            vmax=plotting.vmax,
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Energy [log$_{10}$(MeV)]")
        self.annotate_axis(ax, -1, i_layer, energy)
        fig.subplots_adjust(bottom=0.14, left=0.05, right=0.96, top=0.94)
        fig.subplots_adjust(wspace=0.25)
        pdf.savefig()
        plt.close()


# Command-line options
def options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Input parquet file", required=True)
    parser.add_argument("-o", help="Basename for output files", default="data")
    return parser.parse_args()

# Constants: image parameters
@dataclass(frozen=True)
class images:
    pixels = 40

# Constants: calo cell sizes (mm)
@dataclass(frozen=True)
class cell_sizes:
    ecal = 5.1
    hcal = 30.0

# Constants: calo layers
@dataclass(frozen=True)
class layers:
    ecal = 50
    hcal = 75

# Constants: system ids
@dataclass(frozen=True)
class systems:
    ecal_barrel = 20
    ecal_endcap = 29
    hcal_barrel = 10
    hcal_endcap = 11
    yoke_barrel = 13
    yoke_endcap = 14
    ecal = [ecal_barrel, ecal_endcap]
    hcal = [hcal_barrel, hcal_endcap]

# Constants: plotting
@dataclass(frozen=True)
class plotting:
    x_center, y_center = 181.0, 0.0 # mm
    cell = cell_sizes.ecal
    offset = images.pixels / 2
    axis = [
        [cell*(x_center - offset), cell*(x_center + offset), cell*(y_center - offset), cell*(y_center + offset)],
        [x_center - offset, x_center + offset, y_center - offset, y_center + offset],
        [-offset, offset, -offset, offset],
        [0, 2*offset, 0, 2*offset],
    ]
    unit = [
        "mm",
        "cell",
        "cell, centered",
        "pixel",
    ]
    cols = [
        ("hit_x", "hit_y"),
        ("hit_xp", "hit_yp"),
        ("hit_xpp", "hit_ypp"),
        ("hit_xppp", "hit_yppp"),
    ]
    vmin, vmax = 0, 3
    cmap = "gist_heat_r"
    GeV_to_MeV = 1000.0
    eps = 1e-3


if __name__ == "__main__":
    main()
