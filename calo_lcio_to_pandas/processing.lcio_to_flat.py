"""
A script to convert lcio files to a flat pandas DataFrame.

I avoid parallelism in handling lcio files because the format
(and ROOT) seem designed without parallelism in mind.
"""

import pyLCIO  # type: ignore
from pyLCIO import EVENT, UTIL

import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from tqdm import tqdm

from typing import Any, List, Tuple


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", help="Comma-separated input filenames", required=True)
    parser.add_argument("-o", help="Output filename", required=True)
    return parser.parse_args()


def main() -> None:
    ops = options()
    if not ops.i:
        raise Exception("Need input file with -i")
    inputs = ops.i.split(",")
    output = ops.o
    processor = ProcessLcioToFlat(inputs, output)
    processor.read_hits()
    processor.write_hits()


class ProcessLcioToFlat:

    def __init__(self, fnames: List[str], output: str) -> None:
        self.fnames = fnames
        self.output = output
        self.df = pd.DataFrame()

    def read_hits(self) -> None:
        results = [self.read_hits_serially(fname) for fname in self.fnames]
        self.df = pd.DataFrame(self.merge_results(results))

    def merge_results(self, results: List[dict]) -> dict:
        print("Merging dicts ...")
        result = self.default_dict()
        for res in results:
            for key in res:
                result[key].extend(res[key])
        return result

    def read_hits_serially(self, fname: str) -> dict:
        reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
        reader.open(fname)
        results = []
        for event in tqdm(reader):
            results.append(self.processEventCellView(event))
        reader.close()
        return self.merge_results(results)

    def default_dict(self, n: int = 0) -> dict:
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

    def processEventCellView(self, event: Any) -> dict:
        # print(f'On event {event}')
        d = self.default_dict()
        truth_px, truth_py, truth_pz, truth_e, truth_pdgid = self.processEventTruth(event)
        allnames = event.getCollectionNames()
        for colname in collection_names.reco:
            if colname not in allnames:
                continue
            col = event.getCollection(colname)
            cellIdEncoding = col.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
            cellIdDecoder = UTIL.BitField64(cellIdEncoding)
            for i_hit, hit in enumerate(col):
                cellIdDecoder.setValue(
                    (hit.getCellID0() & 0xFFFFFFFF) | (hit.getCellID1() << 32)
                )
                position = hit.getPosition()
                x, y, z = position[0], position[1], position[2]
                d["event"].append(event.getEventNumber())
                d["hit_system"].append(cellIdDecoder["system"].value())
                d["hit_side"].append(cellIdDecoder["side"].value())
                d["hit_layer"].append(cellIdDecoder["layer"].value())
                d["hit_x"].append(x)
                d["hit_y"].append(y)
                d["hit_z"].append(z)
                d["hit_e"].append(hit.getEnergy())
                d["truth_px"].append(truth_px)
                d["truth_py"].append(truth_py)
                d["truth_pz"].append(truth_pz)
                d["truth_e"].append(truth_e)
                d["truth_pdgid"].append(truth_pdgid)
        return d

    def processEventTruth(self, event: Any) -> Tuple[float, float, float, float, int]:
        event_number = event.getEventNumber()
        n_stable = 0
        for obj in event.getCollection(collection_names.mc):
            if obj.getGeneratorStatus() == 1:
                obj_p = obj.getMomentum()
                obj_e = obj.getEnergy()
                pdgid = obj.getPDG()
                ret = obj_p[0], obj_p[1], obj_p[2], obj_e, pdgid
                n_stable += 1
        if n_stable != 1:
            raise Exception("Unexpected truth particles")
        return ret

    def write_hits(self):
        print(f"Writing hits to file: {self.df.shape}")
        self.df.to_parquet(self.output)


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


if __name__ == "__main__":
    main()
