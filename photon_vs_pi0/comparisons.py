import argparse
import glob
import numpy as np
import scipy as sp
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import List

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import logging

logging.basicConfig(level=logging.INFO)

def main():
    ops = options()
    comp = ComparisonPlots(ops.photon, ops.pizero, ops.output)
    comp.plot()


def options():
    parser = argparse.ArgumentParser(description="Event Displays")
    parser.add_argument("--photon", help="Input file (photons)", required=True)
    parser.add_argument("--pizero", help="Input file (pizeros)", required=True)
    parser.add_argument("-o", "--output", help="Output file", required=True)
    return parser.parse_args()


class ComparisonPlots:
    def __init__(self, photon_file: str, pizero_file, output_file: str) -> None:
        self.photon_file = photon_file
        self.pizero_file = pizero_file
        self.output_file = output_file
        self.photon_df = self.get_df(photon_file)
        self.pizero_df = self.get_df(pizero_file)
        for df in [
            self.photon_df,
            self.pizero_df,
        ]:
            df["hit_r"] = np.sqrt(df["hit_x"] ** 2 + df["hit_y"] ** 2)
            df["hit_ex"] = df["hit_e"] * df["hit_x"]
            df["hit_ey"] = df["hit_e"] * df["hit_y"]
            df["hit_er"] = df["hit_e"] * df["hit_r"]

    def get_df(self, filename: str) -> pd.DataFrame:
        print(f"Reading {filename} ... ")
        def expand(input: str) -> List[str]:
            return [f for path in input.split(",") for f in glob.glob(path)]
        fnames = expand(filename)
        if len(fnames) == 1:
            return pd.read_parquet(fnames[0])
        return pd.concat([pd.read_parquet(f) for f in fnames])

    def plot(self) -> None:
        with PdfPages(self.output_file) as pdf:
            self.plot_radial_std(pdf)

    def plot_radial_std(self, pdf: PdfPages) -> None:
        for coord in ["x", "y", "r", "ex", "ey", "er"]:
            fig, ax = plt.subplots()
            bins = np.linspace(-20, 60, 100)
            ax.hist(self.photon_df.groupby("event")[f"hit_{coord}"].std(), histtype="step", label="Photon", bins=bins, color="blue")
            ax.hist(self.pizero_df.groupby("event")[f"hit_{coord}"].std(), histtype="step", label="Pizero", bins=bins, color="green")
            ax.set_xlabel(f"{coord} [mm]")
            ax.set_ylabel("Counts")
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)

    def plot_event(self, event: int, group: pd.DataFrame, pdf: PdfPages) -> None:
        nrows, ncols = NLAYERS // NCOLS_PER_ROW, NCOLS_PER_ROW
        fig, ax = plt.subplots(
            figsize=(4.5 * ncols, 4 * nrows), nrows=nrows, ncols=ncols
        )
        for i, (layer, layer_group) in enumerate(group.groupby("hit_layer")):
            energy = np.log10(layer_group["hit_e"] + 1e-6)
            i_ax = i // ncols, i % ncols
            scat = ax[i_ax].scatter(
                x=layer_group["hit_x"],
                y=layer_group["hit_y"],
                c=energy,
                vmin=plotting.vmin,
                vmax=plotting.vmax,
                cmap=plotting.cmap,
            )
            self.stylize(ax[i_ax], event, layer)
            fig.colorbar(scat, ax=ax[i_ax])
            fig.subplots_adjust(bottom=0.04, left=0.04, right=0.98, top=0.98)
            fig.subplots_adjust(wspace=0.25, hspace=0.25)

        pdf.savefig(fig)
        plt.close(fig)

    def stylize(self, ax: mpl.axes.Axes, event: int, layer: int) -> None:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_xlim(800, 950)
        ax.set_ylim(-75, 75)
        ax.grid(linewidth=0.1)
        ax.set_axisbelow(True)
        ax.tick_params(right=True, top=True)
        ax.text(
            0.100,
            1.02,
            f"{particle.name[self.pdgid]}, Event {event}, Layer {layer}",
            transform=ax.transAxes,
        )
        ax.text(0.956, 1.02, f"E [log(GeV)]", transform=ax.transAxes)


@dataclass(frozen=True)
class particle:
    name = {
        22: "Photon",
        111: "pizero",
    }


@dataclass(frozen=True)
class plotting:
    vmin, vmax = -3, -0.01
    cmap = "gist_heat_r"


if __name__ == "__main__":
    main()
