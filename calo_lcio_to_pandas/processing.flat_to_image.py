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
from typing import List

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


class ProcessFlatToImage:
    
    def __init__(self, input: str, output: str) -> None:
        self.input = input
        self.output = output
        self.df = pd.DataFrame()
        self.features = np.array([])
        self.labels = np.array([])
        self.particle = {22: "photon", 2112: "neutron"}
        if not os.path.isfile(self.input):
            raise Exception(f"File ")

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
        logger.info("Making label array ... ")
        events, indexs = np.unique(self.df.event, return_index=True)
        self.labels = self.df.truth_e[indexs]
        # todo: add pdgid?
        # events = np.unique(self.df.event)
        # self.labels = np.zeros(events)
        # for event in tqdm(events):
        #     this_ev = self.df[self.df.event == event]
        #     self.labels[event] = this_ev.truth_e.min()

    def make_image_array(self) -> None:
        # NB: this is only implemented for ecal currently
        logger.info("Making image array ... ")
        events, indexs = np.unique(self.df.event, return_index=True)
        

# Command-line options
def options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Input parquet file", required=True)
    parser.add_argument("-o", help="Basename for output files", default="data")
    return parser.parse_args()

# Constants: image parameters
@dataclass(frozen=True)
class images:
    pixels = 20

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

def layers(system: int) -> range:
    # return range(5)
    return range(50) if system == ECAL_ENDCAP else range(75)

def subtraction_x(system: int) -> float:
    return 0.0 # ECal_inner_radius if system == ECAL_ENDCAP else 0.0

def subtraction_y(system: int) -> float:
    return 0.0

def cell_size(system: int) -> float:
    return ECal_cell_size if system == ECAL_ENDCAP else HCal_cell_size

def main2():
    print(f"Loading {FNAME} ...")
    df = pd.read_parquet(FNAME)
    print(f"Loaded!")

    OFFSET = 20

    print(f"Deriving new columns ...")
    df["hit_xp"] = np.rint(df.hit_x / ECal_cell_size)
    df["hit_yp"] = np.rint(df.hit_y / ECal_cell_size)
    df["hit_xpp"] = np.rint((df.hit_x - (df.hit_z * np.tan(THETA))) / np.vectorize(cell_size)(df.hit_system))
    df["hit_ypp"] = np.rint((df.hit_y)                              / np.vectorize(cell_size)(df.hit_system))
    df["hit_xppp"] = np.rint((df.hit_x - (df.hit_z * np.tan(THETA))) / np.vectorize(cell_size)(df.hit_system) + OFFSET)
    df["hit_yppp"] = np.rint((df.hit_y)                              / np.vectorize(cell_size)(df.hit_system) + OFFSET)
    print(f"Derived!")
    print(df)

    CENTER = 181
    axes = {
        ECAL_ENDCAP: [
            [ECal_cell_size*(CENTER - OFFSET), ECal_cell_size*(CENTER + OFFSET), -ECal_cell_size*OFFSET, ECal_cell_size*OFFSET],
            [CENTER - OFFSET, CENTER + OFFSET, -OFFSET, OFFSET],
            [-OFFSET, OFFSET, -OFFSET, OFFSET],
            [0, 2*OFFSET, 0, 2*OFFSET],
            [0, 2*OFFSET, 0, 2*OFFSET],
        ],
        HCAL_ENDCAP: [
            [872, 974, -51, 51],
            [171, 191, -10, 10],
            [-OFFSET, OFFSET, -OFFSET, OFFSET],
            [0, 2*OFFSET, 0, 2*OFFSET],
        ],
    }
    size = {
        ECAL_ENDCAP: 10,
        HCAL_ENDCAP: 50,
    }
    unit = [
        "mm",
        "cell",
        "cell, offset (centered)",
        "cell, offset",
        "pixel",
    ]
    xys = [
        ("hit_x", "hit_y"),
        ("hit_xp", "hit_yp"),
        ("hit_xpp", "hit_ypp"),
        ("hit_xppp", "hit_yppp"),
        (None, None),
    ]

    vmin, vmax = 0, 3
    cmap = "gist_heat_r"
    GeV_to_MeV = 1000.0
    MeV_to_GeV = 1 / GeV_to_MeV
    shape = (2*OFFSET, 2*OFFSET)
    draw = False
    n_events = 10

    arr = np.zeros(shape=(n_events, len(layers(ECAL_ENDCAP)), *shape))
    truth = np.zeros(n_events)

    with PdfPages(PDF) as pdf:

        for event in tqdm(range(n_events)):

            this_ev = df[df.event == event]
            energy = this_ev.truth_e.min()
            particle = PARTICLE_TYPE[this_ev.truth_pdgid.min()]
            truth[event] = energy

            # scatter plots
            # for system in [ECAL_ENDCAP, HCAL_ENDCAP]:
            for system in [ECAL_ENDCAP]:

                this_sys = this_ev[this_ev.hit_system == system]

                for layer in layers(system):

                    this = this_sys[this_sys.hit_layer == layer]

                    coo_df = this[
                        (this.hit_xppp >= 0) &
                        (this.hit_yppp >= 0) &
                        (this.hit_xppp < 2*OFFSET) &
                        (this.hit_yppp < 2*OFFSET)
                    ]
                    coo = coo_matrix((coo_df.hit_e, (coo_df.hit_yppp, coo_df.hit_xppp)), shape=shape)
                    arr[event][layer] = coo.toarray()
                    # print(layer, len(this))

                    if event >= 0:
                        continue

                    this_energy = this.hit_e.sum()
                    window_energy = coo_df.hit_e.sum()
                    window_frac = 0 if this_energy==0 else window_energy / this_energy * 100

                    scat = [None]*len(xys)
                    cbar = [None]*len(xys)
                    fig, ax = plt.subplots(nrows=1, ncols=len(xys), figsize=(24, 4))
                    for i_ax, (x, y) in enumerate(xys):
                        if i_ax == len(xys) - 1 and x==None and y==None:
                            im = ax[i_ax].imshow(
                                np.log10(np.where(coo.toarray() > 0, GeV_to_MeV * coo.toarray(), 1e-3)),
                                cmap=cmap,
                                vmin=vmin,
                                vmax=vmax,
                            )
                            cbar[i_ax] = fig.colorbar(im, ax=ax[i_ax])
                            ax[i_ax].text(0.06, 0.90, f"Window E = {window_energy:.1f} ({window_frac:.1f}%)", transform=ax[i_ax].transAxes)
                        else:
                            scat[i_ax] = ax[i_ax].scatter(
                                this[x],
                                this[y],
                                s=size[system],
                                c=np.log10(GeV_to_MeV * this.hit_e),
                                vmin=vmin,
                                vmax=vmax,
                                cmap=cmap,
                            )
                            cbar[i_ax] = fig.colorbar(scat[i_ax], ax=ax[i_ax])
                        ax[i_ax].tick_params(right=True, top=True)
                        ax[i_ax].axis(axes[system][i_ax])
                        ax[i_ax].grid(True)
                        ax[i_ax].set_xlabel(f"x [{unit[i_ax]}]")
                        ax[i_ax].set_ylabel(f"y [{unit[i_ax]}]")
                        ax[i_ax].set_axisbelow(True)
                        ax[i_ax].text(0.06, 1.02, f"Layer {layer}", transform=ax[i_ax].transAxes)
                        ax[i_ax].text(0.51, 1.02, f"{particle} E = {int(energy)} GeV", transform=ax[i_ax].transAxes)
                        cbar[i_ax].set_label("Energy [log$_{10}$(MeV)]")
                        fig.subplots_adjust(bottom=0.14, left=0.05, right=0.96, top=0.94)
                        fig.subplots_adjust(wspace=0.25)

                    pdf.savefig()
                    plt.close()

                # animation
                if event < 0:
                    print(f"Making images to animate ...")
                    fig, ax = plt.subplots()
                    images = []
                    first = True
                    for layer in layers(system):
                        im = ax.imshow(
                            arr[event][layer],
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax,
                            animated=True,
                        )
                        if first:
                            ax.set_xlabel(f"x [pixel]")
                            ax.set_ylabel(f"y [pixel]")
                            ax.set_axisbelow(True)
                            cbar = fig.colorbar(im, ax=ax)
                            cbar.set_label("Energy [log$_{10}$(MeV)]")
                            first = False
                        images.append([im])
                    print(f"Making animation ...")
                    ani = animation.ArtistAnimation(
                        fig,
                        images,
                        interval=50,
                        blit=True,
                        repeat_delay=1000,
                    )
                    print(f"Saving animation ...")
                    ani.save("movie.gif")
                    print(f"Done animating!")

            # histograms
            for system in [ECAL_ENDCAP, HCAL_ENDCAP]:
                continue
                print(f"Histograms for {system} ... ")
                for layer in layers(system):
                    this = df[(df.hit_system == system) & (df.hit_layer == layer)]
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.hist2d(this.hit_xp, this.hit_yp, weights=this.hit_e, cmin=1e-6, bins=(50, 50))
                    ax.tick_params(right=True, top=True)
                    axes = [
                    -ECal_outer_radius,
                    ECal_outer_radius,
                    -ECal_outer_radius,
                    ECal_outer_radius,
                    ] if system == ECAL_ENDCAP else [
                        -HCal_outer_radius,
                        HCal_outer_radius,
                        -HCal_outer_radius,
                        HCal_outer_radius,
                    ]
                    ax.axis(axes)
                    ax.grid(True)
                    pdf.savefig()
                    plt.close()


    # histogram of energy in window
    fig, ax = plt.subplots()
    ax.hist(arr.sum(axis=(1, 2, 3)) / truth)
    plt.savefig("fraction.pdf")

    # save
    np.save(FNAME + ".features.npy", arr)
    np.save(FNAME + ".labels.npy", truth)


if __name__ == "__main__":
    main()
