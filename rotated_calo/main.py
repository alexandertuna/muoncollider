import pyLCIO # type: ignore

import argparse
from dataclasses import dataclass
import glob
import numpy as np
import pandas as pd
import logging
import os
from typing import List
from tqdm import tqdm

import matplotlib as mpl # type: ignore

mpl.use("Agg")
import matplotlib.pyplot as plt # type: ignore
from matplotlib.backends.backend_pdf import PdfPages # type: ignore

COL_NAME = "ECalEndcapCollection"
INNER_RADIUS = 310.0
OUTER_RADIUS = 2200
PARTICLE_RADIUS = 100.0
CORNER_MIN = 0
CORNER_MAX = 600
MIN_HITS = 50
SIDE_A = 1

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
    plotter = LayerPlotter(expand(ops.i), ops.o, int(ops.l))
    plotter.plot()


def expand(input: str) -> List[str]:
    return [f for path in input.split(",") for f in glob.glob(path)]


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", help="Input filename", required=True)
    parser.add_argument("-o", help="Output filename", required=True)
    parser.add_argument("-l", help="Layer to plot", required=True)
    return parser.parse_args()


class LayerPlotter:
    def __init__(self, filenames: List[str], pdfname: str, layer: int) -> None:
        for filename in filenames:
            logger.info(f"Found {filename}")
        self.filenames = filenames
        self.layer = layer
        self.pdfname = pdfname

    def read(self) -> pyLCIO.EVENT.LCEvent:
        for filename in tqdm(self.filenames):
            reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
            reader.setReadCollectionNames([COL_NAME])
            reader.open(filename)
            for event in reader:
                yield event
            reader.close()

    def plot(self) -> None:
        logger.info("Plotting ... ")
        with PdfPages(self.pdfname) as pdf:
            for event in self.read():
                df = EventDecoder(event).data
                # self.plot_layer(df, pdf, one_event=True)
                break
            self.plot_all_events(pdf)

    def plot_all_events(self, pdf: PdfPages) -> None:
        logger.info("Plotting all events overlaid ... ")
        df = pd.concat([EventDecoder(event).data for event in self.read()])
        # self.plot_layer(df, pdf, one_event=False)
        self.plot_inner_radius(df, pdf)

    def plot_layer(self, df: pd.DataFrame, pdf: PdfPages, one_event: bool) -> None:
        logger.info(f"Plotting df with {len(df)} hits ... ")
        layer = df[(df["hit_layer"] == self.layer) & (df["hit_system"] == systems.ecal_endcap) & (df["hit_side"] == 1)]
        fig, ax = plt.subplots(figsize=(4, 4))
        s = 20 if one_event else 0.05
        ax.scatter(layer["hit_x"], layer["hit_y"], s=s, linewidth=0)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.grid(linewidth=0.1)
        ax.set_axisbelow(True)
        ax.tick_params(top=True, right=True)
        if one_event:
            ax.set_title(f"Event {int(event)}, layer {self.layer}")
            ax.set_xlim([-75, 75])
            ax.set_ylim([800, 1000])
        else:
            fname = ",".join([os.path.basename(f) for f in self.filenames])
            ax.set_title(f"{fname}, layer {self.layer}", fontsize=8)
            ax.set_xlim([-OUTER_RADIUS, OUTER_RADIUS])
            ax.set_ylim([-OUTER_RADIUS, OUTER_RADIUS])
            # add circles
            circle1 = plt.Circle((0, 0), INNER_RADIUS, color="black", fill=False)
            circle2 = plt.Circle((0, 0), OUTER_RADIUS, color="black", fill=False)
            ax.add_artist(circle1)
            ax.add_artist(circle2)
            for (_, group) in layer.groupby("event"):
                if len(group) < MIN_HITS:
                    continue
                circle = plt.Circle((group["hit_x"].median(), group["hit_y"].median()), PARTICLE_RADIUS, color="red", fill=False)
                ax.add_artist(circle)
        fig.subplots_adjust(bottom=0.12, left=(0.17 if one_event else 0.21), right=0.95, top=0.93)
        pdf.savefig(fig)
        plt.close(fig)


    def plot_inner_radius(self, df: pd.DataFrame, pdf: PdfPages) -> None:
        logger.info("Plotting corner ... ")
        subset = df[(df["hit_system"] == systems.ecal_endcap) & (df["hit_side"] == SIDE_A) & (df["hit_x"] < CORNER_MAX) & (df["hit_y"] < CORNER_MAX) & (df["hit_x"] >= CORNER_MIN) & (df["hit_y"] >= CORNER_MIN)]
        subset = subset[["hit_x", "hit_y"]].drop_duplicates()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(subset["hit_x"], subset["hit_y"], s=2.0, linewidth=0)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_xlim([0, CORNER_MAX])
        ax.set_ylim([0, CORNER_MAX])
        ax.grid(linewidth=0.1)
        ax.set_axisbelow(True)
        fig.subplots_adjust(bottom=0.12, left=0.21, right=0.95, top=0.93)
        pdf.savefig(fig)
        plt.close(fig)


class EventDecoder:
    def __init__(self, event: pyLCIO.EVENT.LCEvent) -> None:
        d = self.default_dict()
        event_number = event.getEventNumber()
        col = event.getCollection(COL_NAME)
        for hit in col:
            id0 = hit.getCellID0()
            position = hit.getPosition()
            d["event"].append(event_number)
            d["hit_x"].append(position[0])
            d["hit_y"].append(position[1])
            d["hit_z"].append(position[2])
            d["hit_e"].append(hit.getEnergy())
            d["hit_system"].append(id0 & self.mask(5))
            d["hit_side"].append((id0 >> 5) & self.mask(2))
            d["hit_layer"].append((id0 >> 19) & self.mask(9))
        self.data = pd.DataFrame(d)
        # logger.info(f"Event {event_number} has {len(self.data)} hits")

    def default_dict(self) -> dict:
        return {
            "event": [],
            "hit_x": [],
            "hit_y": [],
            "hit_z": [],
            "hit_e": [],
            "hit_system": [],
            "hit_side": [],
            "hit_layer": [],
        }

    def mask(self, nbits: int) -> int:
        """e.g. mask(4) returns 0b1111"""
        return (1 << nbits) - 1

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
    yoke = [yoke_barrel, yoke_endcap]

if __name__ == "__main__":
    main()
