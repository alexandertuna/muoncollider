import pyLCIO # type: ignore
from pyLCIO import EVENT, UTIL

import os


FNAME = "/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10000.slcio"
COLLECTIONS = (
    "ECalBarrelCollection",
    "ECalEndcapCollection",
    "HCalBarrelCollection",
    "HCalEndcapCollection",
    # "EcalBarrelCollectionRec",
    # "EcalEndcapCollectionRec",
    # "HcalBarrelCollectionRec",
    # "HcalEndcapCollectionRec",
)
ENCODING = "system:0:5,side:5:-2,module:7:8,stave:15:4,layer:19:9,submodule:28:4,x:32:-16,y:48:-16"


def main() -> None:
    checkInput()
    encoding, id0, id1 = getFirstCalorimeterCell()
    if encoding != ENCODING:
        raise Exception(f"Encoding mismatch: {encoding} vs {ENCODING}")
    print(f"Encoding: {encoding}")
    print(f"id0: 0x{id0:08x}")
    print(f"id1: 0x{id1:08x}")
    d_lcio = decodeWithLCIO(id0, id1)
    d_hand = decodeByHand(id0, id1)
    agree = d_lcio == d_hand
    print("W/ LCIO:", d_lcio)
    print("By hand:", d_hand)
    print(f"LCIO and by-hand decoding are in agreement: {agree}")


def checkInput() -> None:
    if not os.path.isfile(FNAME):
        raise Exception(f"Input file does not exist: {FNAME}")


def getFirstCalorimeterCell() -> tuple[str, int, int]:
    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open(FNAME)
    for event in reader:
        for colname in COLLECTIONS:
            col = event.getCollection(colname)
            if not col:
                continue
            for obj in col:
                encoding = col.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
                return encoding, obj.getCellID0(), obj.getCellID1()
    raise Exception(f"Found no calorimeter cells in {FNAME}")
    return "", 0, 0


def decodeWithLCIO(id0: int, id1: int) -> dict[str, int]:
    decoder = UTIL.BitField64(ENCODING)
    decoder.setValue((id0 & 0xffffffff) |
                     (id1 << 32))
    values = decoder.valueString()
    d = {}
    for entry in values.split(","):
        key, value = entry.split(":")
        d[key] = int(value)
    return d


def decodeByHand(id0: int, id1: int) -> dict[str, int]:
    d = {}
    d["system"] = id0 & ((1 << 5) - 1)
    d["side"] = twos_complement((id0 >> 5) & ((1 << 2) - 1), 2)
    d["module"] = (id0 >> 7) & ((1 << 8) - 1)
    d["stave"] = (id0 >> 15) & ((1 << 4) - 1)
    d["layer"] = (id0 >> 19) & ((1 << 9) - 1)
    d["submodule"] = (id0 >> 28) & ((1 << 4) - 1)
    d["x"] = twos_complement(id1 & 0xff, 16)
    d["y"] = twos_complement(id1 >> 16, 16)
    return d


def twos_complement(val, bits):
    """ https://stackoverflow.com/questions/1604464/twos-complement-in-python """
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val                         # return positive value as is


if __name__ == "__main__":
    main()
