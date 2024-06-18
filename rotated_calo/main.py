import pyLCIO

import argparse
import numpy as np

def main():
    ops = options()
    plotter = LayerPlotter(ops.i)
    plotter.plot()


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", help="Input filename", required=True)
    return parser.parse_args()


class LayerPlotter:
    def __init__(self, filename: str):
        self.reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
        self.reader.open(filename)

    def plot(self):
        for event in self.reader:
            self.plot_event(event)

    def plot_event(self, event):
        for collection in event:
            if collection.getTypeName() == "TrackerHit":
                self.plot_hits(collection)

    def plot_hits(self, collection):
        for hit in collection:
            pos = np.array(hit.getPosition())
            print(pos)


if __name__ == "__main__":
    main()
