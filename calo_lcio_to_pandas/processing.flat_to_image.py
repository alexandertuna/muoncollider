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
    processor.write_data()


class ProcessFlatToImage:
    
    def __init__(self, input: str, output: str) -> None:
        self.input = input
        self.output = output
        if not os.path.isfile(self.input):
            raise Exception(f"{self.input} does not exist")
        self.df = pd.DataFrame()
        self.features = np.array([])
        self.labels = np.array([])
        self.window = [0, images.pixels, 0, images.pixels]
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
            (self.df.hit_xppp >= self.window[0]) &
            (self.df.hit_yppp >= self.window[2]) &
            (self.df.hit_xppp < self.window[1]) &
            (self.df.hit_yppp < self.window[3])
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


if __name__ == "__main__":
    main()
