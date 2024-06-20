import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import logging
logging.basicConfig(level=logging.INFO)

NLAYERS = 50
NCOLS_PER_ROW = 10


def main():
    ops = options()
    ev = EventDisplays(ops.input, ops.output, ops.nevents)
    ev.plot()


def options():
    parser = argparse.ArgumentParser(description='Event Displays')
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-o', '--output', help='Output file', required=True)
    parser.add_argument('-n', '--nevents', help='Number of events to display', type=int, default=10)
    return parser.parse_args()


class EventDisplays:
    def __init__(self, input_file: str, output_file: str, n_events: int) -> None:
        self.input_file = input_file
        self.output_file = output_file
        self.n_events = n_events
        self.df = pd.read_parquet(input_file)
        self.pdgid = self.df["truth_pdgid"].median()
        print(self.df.describe())


    def plot(self) -> None:
        with PdfPages(self.output_file) as pdf:
            for i, (event, group) in tqdm(enumerate(self.df.groupby("event"))):
                if i >= self.n_events:
                    break
                self.plot_event(event, group, pdf)


    def plot_event(self, event: int, group: pd.DataFrame, pdf: PdfPages) -> None:
        nrows, ncols = NLAYERS // NCOLS_PER_ROW, NCOLS_PER_ROW
        fig, ax = plt.subplots(figsize=(4.5*ncols, 4*nrows), nrows=nrows, ncols=ncols)
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
        ax.text(0.100, 1.02, f"{particle.name[self.pdgid]}, Event {event}, Layer {layer}", transform=ax.transAxes)
        ax.text(0.956, 1.02, f"E [log(GeV)]", transform=ax.transAxes)


@dataclass(frozen=True)
class particle:
    name = {
        22: "Photon",
        111: "Pi0",
    }


@dataclass(frozen=True)
class plotting:
    vmin, vmax = -3, -0.01
    cmap = "gist_heat_r"


if __name__ == '__main__':
    main()
