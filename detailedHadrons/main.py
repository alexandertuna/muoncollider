import pyLCIO
from pyLCIO import EVENT, UTIL

import argparse
import glob
import math
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ETAS = [
    (0.0, 1.1),
    (1.1, 1.2),
    (1.2, 2.0),
]

# Set up hit encoder/decoder
# encoding = col.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
encoding = "system:0:5,side:5:-2,module:7:8,stave:15:4,layer:19:9,submodule:28:4,x:32:-16,y:48:-16"
decoder = UTIL.BitField64(encoding)

# Command-line options
def options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Particle type to analyze (default: %(default)s)", choices=["ne", "pi"], default="ne")
    parser.add_argument("-n", help="Maximum number of events to analyze", default=0, type=int)
    parser.add_argument("-p", help="Parquet file to read and/or write (default: %(default)s)", default="detailedHadrons.parquet")
    parser.add_argument("-l", help="Load parquet file instead of lcio", action="store_true")
    return parser.parse_args()

@dataclass(frozen=True)
class systems:
    ecal_barrel = 20
    ecal_endcap = 29
    hcal_barrel = 10
    hcal_endcap = 11
    yoke_barrel = 13
    yoke_endcap = 14
    ecal = [ecal_barrel,
            ecal_endcap]
    hcal = [hcal_barrel,
            hcal_endcap]
    yoke = [yoke_barrel,
            yoke_endcap]

@dataclass(frozen=True)
class ppd:
    mipPe = 15.0
    npix = 2000.0

@dataclass(frozen=True)
class calib:
    class mip:
        ecal = 0.0001575
        hcal_barrel = 0.0004925
        hcal_endcap = 0.0004725
    class mip_to_reco:
        ecal = 0.0066150
        hcal_barrel = 0.024625
        hcal_endcap = 0.024625

@dataclass(frozen=True)
class sampling_scaling:
    ecal        = calib.mip_to_reco.ecal / calib.mip.ecal
    hcal_barrel = calib.mip_to_reco.hcal_barrel / calib.mip.hcal_barrel
    hcal_endcap = calib.mip_to_reco.hcal_endcap / calib.mip.hcal_endcap

def theta(x: float, y: float, z: float) -> float:
    return np.arccos(z / np.linalg.norm([x, y, z]))

def eta(x: float, y: float, z: float) -> float:
    return -np.log(np.tan(theta(x, y, z)/2.0))

def phi(x: float, y: float) -> float:
    return np.arctan2(y, x)

@dataclass
class particleFromTheGun:
    px: float
    py: float
    pz: float
    e: float

    def __post_init__(self):
        self.eta = eta(self.px, self.py, self.pz)
        self.phi = phi(self.px, self.py)

@dataclass
class particleReconstructed:
    tru_eta: float
    tru_phi: float
    tru_e: float
    sim_e: float
    rec_e: float
    dig_e: float
    clu_e: float
    pfo_e: float
    tru_n: int
    sim_n: int
    dig_n: int
    rec_n: int
    clu_n: int
    pfo_n: int


# Set up things for each object
settings = {
    "fnames": { "ne": "/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/neutronGun_E_250_1000*"
            },
    "pdgid":  { "ne": [2112],
                "pi": [211, 111],
            },
}

# Progress bar
def progress(time_diff, nprocessed, ntotal):
    nprocessed, ntotal = float(nprocessed), float(ntotal)
    rate = (nprocessed+1)/time_diff
    msg = "\r > %6i / %6i | %2i%% | %8.4fHz | %6.1fm elapsed | %6.1fm remaining"
    msg = msg % (nprocessed, ntotal, 100*nprocessed/ntotal, rate, time_diff/60, (ntotal-nprocessed)/(rate*60))
    print(msg)

# Script
def main() -> None:
    ops = options()
    study = DetailedHadronStudy(ops.i, ops.p, ops.l, ops.n)
    study.load_data()
    if not ops.l:
        study.write_data()
    study.plot_energy()

# Analysis class
class DetailedHadronStudy:

    def __init__(self, obj_type: str, parquet_name: str, load_parquet: bool, num_events: int) -> None:
        self.obj_type = obj_type
        self.parquet_name = parquet_name
        self.load_parquet = load_parquet
        self.num_events = num_events
        self.df = pd.DataFrame()
        self.mapping = {
            "sim": "Sim. Calorimeter",
            "dig": "Digi. Calorimeter",
            "rec": "Reco. Calorimeter",
            "clu": "Cluster",
            "pfo": "Reconstructed PF",
        }
        self.constants = {
            "sim": {
                "mip (ecal)": calib.mip.ecal,
                "mip (hcal, barrel)": calib.mip.hcal_barrel,
                "mip (hcal, endcap)": calib.mip.hcal_endcap,
                "mip->reco (ecal)": calib.mip_to_reco.ecal,
                "mip->reco (hcal, barrel)": calib.mip_to_reco.hcal_barrel,
                "mip->reco (hcal, endcap)": calib.mip_to_reco.hcal_endcap,
            },
            "dig": {
                "mip->reco (ecal)": calib.mip_to_reco.ecal,
                "mip->reco (hcal, barrel)": calib.mip_to_reco.hcal_barrel,
                "mip->reco (hcal, endcap)": calib.mip_to_reco.hcal_endcap,
                "ppd.mipPe": ppd.mipPe,
                "ppd.npix": ppd.npix,
            },
            "rec": {},
            "clu": {},
            "pfo": {},
        }


    def load_data(self) -> None:
        if self.load_parquet:
            if self.num_events:
                print("Ignoring -n option when using -l")
            self.df = pd.read_parquet(self.parquet_name)
        else:
            particles = self.read_lcio()
            self.df = pd.DataFrame([vars(p) for p in particles])
        print(self.df)


    def write_data(self) -> None:
        self.df.to_parquet(self.parquet_name)


    def plot_energy(self) -> None:
        print("Plotting energy ... ")

        binsx = binsy = np.arange(0, 1501, 15)
        linex = liney = [min(binsx), max(binsx)]

        with PdfPages("energy.pdf") as pdf:
            for source in ["sim", "dig", "rec", "clu", "pfo"]:
                for eta_min, eta_max in ETAS:
                    print(f"Plotting {source} energy in eta range {eta_min}, {eta_max} ... ")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    condition = (abs(self.df.tru_eta) > eta_min) & (abs(self.df.tru_eta) < eta_max)
                    counts, xedges, yedges, im = ax.hist2d(self.df.tru_e[condition],
                                                           self.df[f"{source}_e"][condition],
                                                           bins=(binsx, binsy),
                                                           cmap="rainbow",
                                                           cmin=0.1)
                    ax.plot(linex, liney, color="gray", linewidth=1, linestyle="dashed")
                    ax.set_xlabel("True energy [GeV]")
                    ax.set_ylabel(f"{self.mapping[source]} energy [GeV]")
                    ax.tick_params(top=True, right=True)
                    ax.text(0.05, 0.6, f"{eta_min} < |${{\eta}}$| < {eta_max}", transform=ax.transAxes, fontsize=12)
                    for it, (label, value) in enumerate(self.constants[source].items()):
                        ax.text(0.05, 0.95 - it*0.04, f"{label} = {value}", transform=ax.transAxes, fontsize=6, color="gray")
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.set_label("Number of entries")
                    fig.subplots_adjust(bottom=0.14, left=0.15, right=0.95, top=0.95)
                    pdf.savefig()


    def filenames(self) -> List[str]:
        samples = glob.glob(settings["fnames"][self.obj_type])
        fnames = []
        for s in samples:
            fnames += glob.glob(f"{s}/*.slcio")
        print("Found %i files." % len(fnames))
        return fnames


    def read_lcio(self) -> List[particleReconstructed]:

        # Announcements
        if self.num_events > 0:
            print(f"Running on {self.num_events} events at maximum")

        # Loop over events
        reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
        i_total = 0
        t0 = time.time()
        fnames = self.filenames()

        # List of particles
        particles = []

        # File loop
        for i_file, f in enumerate(fnames):

            reader.open(f)

            # Keep track of total events
            if self.num_events > 0 and i_total >= self.num_events:
                break

            # Event loop
            for i_event, event in enumerate(reader):

                # Keep track of total events
                if self.num_events > 0 and i_total >= self.num_events:
                    break
                i_total += 1

                # Get the collections we care about
                mcpCollection = getCollection(event, "MCParticle")

                # Make particle object
                particleTrue = None

                # Loop over the truth objects and fill histograms
                tru_e, tru_n = 0, 0
                for mcp in mcpCollection:
                    p, e = mcp.getMomentum(), mcp.getEnergy()
                    if all([abs(mcp.getPDG()) in settings['pdgid'][self.obj_type],
                            mcp.getGeneratorStatus()==1,
                            abs(eta(p[0], p[1], p[2])) < 2.0,
                        ]):
                        tru_n += 1
                        p, e = mcp.getMomentum(), mcp.getEnergy()
                        particleTrue = particleFromTheGun(p[0], p[1], p[2], e)
                        tru_e = e

                # If there's no good truth reference, move on
                if particleTrue is None:
                    continue

                # If there are too many truth references, get confused
                if tru_n > 1:
                    raise Exception(f"Found too many mcp references ({tru_n})")

                # Loop over digi hits and sum
                sim_e = getEnergySim(event)
                dig_e = getEnergyDig(event)
                rec_e = getEnergyRec(event)
                clu_e, clu_n = getEnergyAndNumberClu(event)
                pfo_e, pfo_n = getEnergyAndNumberPfo(event, self.obj_type)

                # Count number of items
                def countElements(event, collection_names):
                    return sum([len(getCollection(event, name)) for name in collection_names])
                sim_n = countElements(event, ["ECalBarrelCollection",
                                              "ECalEndcapCollection",
                                              "HCalBarrelCollection",
                                              "HCalEndcapCollection"])
                dig_n = countElements(event, ["EcalBarrelCollectionDigi",
                                              "EcalEndcapCollectionDigi",
                                              "HcalBarrelCollectionDigi",
                                              "HcalEndcapCollectionDigi"])
                rec_n = countElements(event, ["EcalBarrelCollectionRec",
                                              "EcalEndcapCollectionRec",
                                              "HcalBarrelCollectionRec",
                                              "HcalEndcapCollectionRec"])

                # Create a summary
                tru_eta, tru_phi = particleTrue.eta, particleTrue.phi
                particleSummary = particleReconstructed(
                    tru_eta,
                    tru_phi,

                    tru_e,
                    sim_e,
                    rec_e,
                    dig_e,
                    clu_e,
                    pfo_e,

                    tru_n,
                    sim_n,
                    dig_n,
                    rec_n,
                    clu_n,
                    pfo_n,
                )
                particles.append(particleSummary)

            # Bookkeeping
            progress(time.time() - t0, i_file + 1, len(fnames))
            reader.close()

        return particles

def reconstructHcalEnergy(hit):
    """
    https://www.desy.de/~dudarboh/marlinreco_doc/html/RealisticCaloDigiScinPpd_8cc_source.html
    """
    decoder.setValue((hit.getCellID0() & 0xffffffff) |
                     (hit.getCellID1() << 32))
    energy = hit.getEnergy()
    system = decoder['system'].value()
    assert system in systems.hcal
    barrel = system == systems.hcal_barrel
    calibrCoeff = calib.mip_to_reco.hcal_barrel if barrel else calib.mip_to_reco.hcal_endcap
    r = 0.95
    if (energy < r*ppd.npix):
        energy = -ppd.npix * math.log ( 1. - ( energy / ppd.npix ) )
    else:
        energy = 1/(1-r)*(energy-r*ppd.npix)-ppd.npix*math.log(1-r)
    energy /= ppd.mipPe
    energy *= calibrCoeff
    return energy

def getCollection(event, name):
    if name in event.getCollectionNames():
        return event.getCollection(name)
    return []

def getEnergySim(event):
    energy = 0
    ecal_b = getCollection(event, "ECalBarrelCollection")
    ecal_e = getCollection(event, "ECalEndcapCollection")
    hcal_b = getCollection(event, "HCalBarrelCollection")
    hcal_e = getCollection(event, "HCalEndcapCollection")
    for col, scaling in [
            (ecal_b, sampling_scaling.ecal),
            (ecal_e, sampling_scaling.ecal),
            (hcal_b, sampling_scaling.hcal_barrel),
            (hcal_e, sampling_scaling.hcal_endcap),
    ]:
        for sim in col:
            energy += sim.getEnergy()*scaling
    return energy

def getEnergyDig(event):
    energy = 0
    ecal_b = getCollection(event, "EcalBarrelCollectionDigi")
    ecal_e = getCollection(event, "EcalEndcapCollectionDigi")
    hcal_b = getCollection(event, "HcalBarrelCollectionDigi")
    hcal_e = getCollection(event, "HcalEndcapCollectionDigi")
    for col in [ecal_b, ecal_e]:
        for dig in col:
            energy += dig.getEnergy()*calib.mip_to_reco.ecal
    for col in [hcal_b, hcal_e]:
        for dig in col:
            energy += reconstructHcalEnergy(dig)
    return energy

def getEnergyRec(event):
    energy = 0
    ecal_b = getCollection(event, "EcalBarrelCollectionRec")
    ecal_e = getCollection(event, "EcalEndcapCollectionRec")
    hcal_b = getCollection(event, "HcalBarrelCollectionRec")
    hcal_e = getCollection(event, "HcalEndcapCollectionRec")
    for col in [ecal_b, ecal_e, hcal_b, hcal_e]:
        for rec in col:
            energy += rec.getEnergy()
    return energy

def getEnergyAndNumberClu(event):
    energy, num = 0, 0
    cluCollection = getCollection(event, "PandoraClusters")
    for clu in cluCollection:
        num += 1
        if clu.getEnergy() > energy:
            energy = clu.getEnergy()
    return energy, num

def getEnergyAndNumberPfo(event, obj_type):
    energy, num = 0, 0
    pfos = getCollection(event, "PandoraPFOs")
    for pfo in pfos:
        if abs(pfo.getType()) in settings['pdgid'][obj_type]:
            num += 1
            if pfo.getEnergy() > energy:
                energy = pfo.getEnergy()
    return energy, num

if __name__ == "__main__":
    main()
