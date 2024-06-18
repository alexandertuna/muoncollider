import pyLCIO

import argparse
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

COL_NAME = "ECalEndcapCollection"

def main():
    ops = options()
    plotter = LayerPlotter(ops.i, ops.o, int(ops.l))
    plotter.plot()


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", help="Input filename", required=True)
    parser.add_argument("-o", help="Output filename", required=True)
    parser.add_argument("-l", help="Layer to plot", required=True)
    return parser.parse_args()


class LayerPlotter:
    def __init__(self, filename: str, pdfname: str, layer: int):
        self.reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
        self.reader.open(filename)
        self.layer = layer
        self.pdfname = pdfname


    def plot(self) -> None:
        with PdfPages(self.pdfname) as pdf:
            for event in self.reader:
                df = EventDecoder(event).data
                self.plot_event(df, pdf)
                break


    def plot_event(self, df: pd.DataFrame, pdf: PdfPages) -> None:
        layer = df[df["hit_layer"] == self.layer]
        event = layer["event"].mean()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(layer["hit_x"], layer["hit_y"])
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"Event {int(event)}, layer {self.layer}")
        ax.grid()
        ax.set_axisbelow(True)
        ax.tick_params(top=True, right=True)
        ax.set_xlim([-75, 75])
        ax.set_ylim([800, 1000])
        fig.subplots_adjust(bottom=0.12, left=0.17, right=0.95, top=0.93)
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
            d["hit_layer"].append((id0 >> 19) & self.mask(9))
        self.data = pd.DataFrame(d)

    def default_dict(self) -> dict:
        return {
            "event": [],
            "hit_x": [],
            "hit_y": [],
            "hit_z": [],
            "hit_e": [],
            "hit_layer": [],
        }

    def mask(self, nbits: int) -> int:
        """e.g. mask(4) returns 0b1111"""
        return (1 << nbits) - 1





if __name__ == "__main__":
    main()
