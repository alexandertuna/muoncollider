"""
A script to convert a lcio file to a flat pandas DataFrame.
"""

import pyLCIO  # type: ignore
from pyLCIO import EVENT, UTIL

import argparse
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm

from typing import List, Tuple

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


@dataclass(frozen=True)
class TruthParticle:
    px: float
    py: float
    pz: float
    e: float
    pdgid: int


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", help="Input filename", required=True)
    parser.add_argument("-o", help="Output filename", required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        filename="log.log",
        format="%(asctime)s %(message)s",
        filemode="w",
        level=logging.DEBUG,
    )
    ops = options()
    processor = ProcessLcioToFlat(ops.i, ops.o)
    processor.read_hits()
    processor.write_hits()


class ProcessLcioToFlat:

    def __init__(self, input: str, output: str) -> None:
        self.input = input
        self.output = output
        self.df = pd.DataFrame()

    def read_hits(self) -> None:
        logger.info(f"Reading {self.input} ...")
        reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
        reader.open(self.input)
        results = [self.read_event(event) for event in tqdm(reader)]
        reader.close()
        logger.info(f"Merging events into a DataFrame ...")
        self.df = pd.concat([pd.DataFrame(res) for res in results])

    def read_event(self, event: pyLCIO.EVENT.LCEvent) -> dict:
        d = self.default_dict()
        truth = self.read_event_truth(event)
        event_number = event.getEventNumber()
        allnames = event.getCollectionNames()
        for colname in collection_names.reco:
            if colname not in allnames:
                continue
            col = event.getCollection(colname)
            for hit in col:
                id0 = hit.getCellID0()
                position = hit.getPosition()
                d["event"].append(event_number)
                d["hit_system"].append(id0 & self.mask(5))
                d["hit_side"].append((id0 >> 5) & self.mask(2))
                d["hit_layer"].append((id0 >> 19) & self.mask(9))
                d["hit_x"].append(position[0])
                d["hit_y"].append(position[1])
                d["hit_z"].append(position[2])
                d["hit_e"].append(hit.getEnergy())
                d["truth_px"].append(truth.px)
                d["truth_py"].append(truth.py)
                d["truth_pz"].append(truth.pz)
                d["truth_e"].append(truth.e)
                d["truth_pdgid"].append(truth.pdgid)
        return d

    def read_event_truth(self, event: pyLCIO.EVENT.LCEvent) -> TruthParticle:
        event_number = event.getEventNumber()
        n_stable = 0
        for obj in event.getCollection(collection_names.mc):
            if obj.getGeneratorStatus() == 1:
                obj_p = obj.getMomentum()
                obj_e = obj.getEnergy()
                pdgid = obj.getPDG()
                return TruthParticle(obj_p[0], obj_p[1], obj_p[2], obj_e, pdgid)
        raise Exception("Missing truth particle")

    def mask(self, nbits: int) -> int:
        """e.g. mask(4) returns 0b1111"""
        return (1 << nbits) - 1

    def write_hits(self) -> None:
        logger.info(f"Writing hits to file: {self.df.shape}")
        self.df.to_parquet(self.output)

    def default_dict(self) -> dict:
        return {
            "event": [],
            "hit_system": [],
            "hit_side": [],
            "hit_layer": [],
            "hit_x": [],
            "hit_y": [],
            "hit_z": [],
            "hit_e": [],
            "truth_px": [],
            "truth_py": [],
            "truth_pz": [],
            "truth_e": [],
            "truth_pdgid": [],
        }


# Constants: collection names
@dataclass(frozen=True)
class collection_names:
    mc = "MCParticle"
    reco = [
        "EcalBarrelCollectionRec",
        "EcalEndcapCollectionRec",
        "HcalBarrelCollectionRec",
        "HcalEndcapCollectionRec",
    ]
    all = reco + [mc]


if __name__ == "__main__":
    main()
