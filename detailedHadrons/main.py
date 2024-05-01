import pyLCIO
from pyLCIO import EVENT, UTIL

import argparse
import glob
import collections
import ctypes
import math
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

exec(open("./plotHelper.py").read())
ROOT.gROOT.SetBatch()
ROOT.gErrorIgnoreLevel = ROOT.kWarning

# Set up hit encoder/decoder
# encoding = col.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
encoding = "system:0:5,side:5:-2,module:7:8,stave:15:4,layer:19:9,submodule:28:4,x:32:-16,y:48:-16"
decoder = UTIL.BitField64(encoding)

# Command-line options
def options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(usage=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", help="Particle type to analyze", choices=["ne", "pi"], default="ne")
    parser.add_argument("-n", help="Maximum number of events to analyze")
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
    mcp_n: int
    sim_n: int
    dig_n: int
    rec_n: int
    clu_n: int
    pfo_n: int

# Set up things for each object
settings = {
        "fnames": { "ne": "/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/neutronGun_E_250_1000*"
                },
        "labelname": { "ne": "Neutron",
                    },
        "plotdir":{ "ne": "neutrons",
                },
        "pdgid":  { "ne": [2112],
                    "pi": [211, 111],
                },
        "mass":   { "ne": 0.940,
                    "pi": 0.135,
                },
}

def filenames(obj_type: str) -> List[str]:
    samples = glob.glob(settings["fnames"][obj_type])
    fnames = []
    for s in samples:
        fnames += glob.glob(f"{s}/*.slcio")
    print("Found %i files."%len(fnames))
    return fnames

# Define good particle
def isGood(tlv):
    if abs(tlv.Eta()) < 2:
        return True
    return False

# Progress bar
def progress(time_diff, nprocessed, ntotal):
    nprocessed, ntotal = float(nprocessed), float(ntotal)
    rate = (nprocessed+1)/time_diff
    msg = "\r > %6i / %6i | %2i%% | %8.4fHz | %6.1fm elapsed | %6.1fm remaining"
    msg = msg % (nprocessed, ntotal, 100*nprocessed/ntotal, rate, time_diff/60, (ntotal-nprocessed)/(rate*60))
    print(msg)

# convert hcal digit energy to GeV
def reconstructHcalEnergy(hit):
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

def main():
    particles = readLcio()
    df = pd.DataFrame([vars(p) for p in particles])
    print(df)


def readLcio():

    # Command-line args
    ops = options()
    obj_type = ops.i
    max_events = int(ops.n) if ops.n else ops.n

    # Announcements
    print(f"Running on {settings['labelname'][obj_type]}")
    if max_events is not None:
        print(f"Running on {max_events} events at maximum")

    # ############## CREATE EMPTY HISTOGRAM OBJECTS  #############################
    # Set up histograms
    # This is an algorithmic way of making a bunch of histograms and storing them in a dictionary
    variables = {}
    #variables["pt"] =  {"nbins": 30, "xmin": 0, "xmax": 3000,   "title": "p_{T} [GeV]"}
    variables["E"] =   {"nbins": 50, "xmin": 0, "xmax": 1000,   "title": "E [GeV]"}
    #variables["eta"] = {"nbins": 30, "xmin": -3, "xmax": 3,     "title": "#eta"}
    #variables["phi"] = {"nbins": 30, "xmin": -3.5, "xmax": 3.5, "title": "#phi"}
    #variables["n"] =   {"nbins": 20, "xmin": 0, "xmax": 20,     "title": "n"}
    hists = {}

    objects = {}
    objects["mcp"] = f"True {settings['labelname'][obj_type]}"
    objects["sim"] = "Sim Calorimeter"
    objects["dig"] = "Digi Calorimeter"
    objects["rec"] = "Reco Calorimeter"
    objects["clu"] = "Matched Cluster"
    objects["pfo"] = f"Reconstructed {settings['labelname'][obj_type]}"

    for obj in objects:
        for var in variables:
            hists[obj+"_"+var] = ROOT.TH1F(obj+"_"+var, objects[obj],
                                           variables[var]["nbins"],
                                           variables[var]["xmin"],
                                           variables[var]["xmax"])

    ranges = ["_0to1p1", "_1p1to1p2", "_1p2to2"]

    # Initialize all the 2D histograms: the each of the above variables at each level vs the mcp value
    hists2d = {}
    for obj in objects:
        for var in variables:
            if obj == "mcp": continue
            for r in ranges:
                hists2d[obj+"_v_mcp_"+var+r] = ROOT.TH2F(obj+"_v_mcp_"+var+r,
                                                         obj+"_v_mcp_"+var+r,
                                                         variables[var]["nbins"],
                                                         variables[var]["xmin"],
                                                         variables[var]["xmax"],
                                                         variables[var]["nbins"],
                                                         variables[var]["xmin"],
                                                         variables[var]["xmax"])


    # ############## LOOP OVER EVENTS AND FILL HISTOGRAMS  #############################
    # Loop over events
    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    # reader.setReadCollectionNames(["MCParticle", "PandoraPFOs", "ECalBarrelCollection", "ECalEndcapCollection", "EcalBarrelCollectionDigi", "EcalEndcapCollectionDigi", "EcalBarrelCollectionRec", "EcalEndcapCollectionRec", "PandoraClusters"])
    i_total = 0
    t0 = time.time()
    fnames = filenames(obj_type)
    n_clu_ob_dict = collections.defaultdict(int)
    n_pfo_ob_dict = collections.defaultdict(int)

    # List of particles
    particles = []

    # File loop
    for i_file, f in enumerate(fnames):

        reader.open(f)

        # Keep track of total events
        if max_events is not None and i_total >= max_events:
            break

        # Event loop
        for i_event, event in enumerate(reader):

            # Keep track of total events
            if max_events is not None and i_total >= max_events:
                break
            i_total += 1

            # Get the collections we care about
            mcpCollection = getCollection(event, "MCParticle")
            simCollection_b = getCollection(event, "ECalBarrelCollection")
            simCollection_e = getCollection(event, "ECalEndcapCollection")
            hcalSimCollection_b = getCollection(event, "HCalBarrelCollection")
            hcalSimCollection_e = getCollection(event, "HCalEndcapCollection")
            yokeSimCollection_b = getCollection(event, "YokeBarrelCollection")
            yokeSimCollection_e = getCollection(event, "YokeEndcapCollection")
            digCollection_b = getCollection(event, "EcalBarrelCollectionDigi")
            digCollection_e = getCollection(event, "EcalEndcapCollectionDigi")
            recCollection_b = getCollection(event, "EcalBarrelCollectionRec")
            recCollection_e = getCollection(event, "EcalEndcapCollectionRec")
            hcalDigCollection_b = getCollection(event, "HcalBarrelCollectionDigi")
            hcalDigCollection_e = getCollection(event, "HcalEndcapCollectionDigi")
            hcalRecCollection_b = getCollection(event, "HcalBarrelCollectionRec")
            hcalRecCollection_e = getCollection(event, "HcalEndcapCollectionRec")
            cluCollection = getCollection(event, "PandoraClusters")
            pfoCollection = getCollection(event, "PandoraPFOs")

            # Make counter variables
            n_mcp_ob = 0
            sim_n = 0
            dig_n = 0
            rec_n = 0
            n_pfo_ob = 0
            n_clu_ob = 0
            has_pfo_ob = False
            has_clu_ob = False
            my_pfo_ob = None
            my_clu_ob = None
            my_mcp_ob = None

            # Make particle object
            particleTrue = None

            # Loop over the truth objects and fill histograms
            tru_E = 0
            for mcp in mcpCollection:
                mcp_tlv = getTLV(mcp)
                if all([abs(mcp.getPDG()) in settings['pdgid'][obj_type],
                        mcp.getGeneratorStatus()==1,
                        isGood(mcp_tlv)]):
                    n_mcp_ob += 1
                    my_mcp_ob = mcp_tlv
                    p, e = mcp.getMomentum(), mcp.getEnergy()
                    particleTrue = particleFromTheGun(p[0], p[1], p[2], e)
                    tru_E = mcp.getEnergy()

            # If there's no good truth reference, move on
            if particleTrue is None:
                continue

            # If there are too many truth references, get confused
            if n_mcp_ob > 1:
                raise Exception(f"Foudn too many mcp references ({n_mcp_ob})")

            # Loop over digi hits and sum
            sim_E = getEnergySim(event)
            dig_E = getEnergyDig(event)
            rec_E = getEnergyRec(event)
            clu_E = getEnergyClu(event)
            pfo_E = getEnergyPfo(event, obj_type)

            # Count number of items
            def countElements(event, collection_names):
                return sum([len(getCollection(event, name)) for name in collection_names])
            sim_n = countElements(event, ["ECalBarrelCollection", "ECalEndcapCollection", "HCalBarrelCollection", "HCalEndcapCollection"])
            dig_n = countElements(event, ["EcalBarrelCollectionDigi", "EcalEndcapCollectionDigi", "HcalBarrelCollectionDigi", "HcalEndcapCollectionDigi"])
            rec_n = countElements(event, ["EcalBarrelCollectionRec", "EcalEndcapCollectionRec", "HcalBarrelCollectionRec", "HcalEndcapCollectionRec"])

            # Create a summary
            tru_eta, tru_phi = particleTrue.eta, particleTrue.phi
            particleSummary = particleReconstructed(
                tru_eta,
                tru_phi,

                tru_E,
                sim_E,
                rec_E,
                dig_E,
                clu_E,
                pfo_E,

                n_mcp_ob,
                sim_n,
                dig_n,
                rec_n,
                n_clu_ob,
                n_pfo_ob,
            )
            particles.append(particleSummary)

            # Only make plots for events with isGood mcps
            if particleTrue is not None:
                n_clu_ob_dict[n_clu_ob] += 1
                n_pfo_ob_dict[n_pfo_ob] += 1
                hists["mcp_E"].Fill(my_mcp_ob.E())
                if has_pfo_ob:
                    hists["pfo_E"].Fill(my_pfo_ob.E())
                if has_clu_ob:
                    hists["clu_E"].Fill(my_clu_ob.getEnergy())
                hists["sim_E"].Fill(sim_E)
                hists["dig_E"].Fill(dig_E)
                hists["rec_E"].Fill(rec_E)

                # Print out 2D distributions per eta range
                for r in ranges:
                    r1 = r.replace("p", ".").strip("_")
                    low_eta = r1.split("to")[0]
                    high_eta = r1.split("to")[1]
                    selection_string = f"my_mcp_ob.Eta()>={low_eta} and my_mcp_ob.Eta()<{high_eta}"
                    if eval(selection_string):
                        if has_pfo_ob:
                            hists2d["pfo_v_mcp_E"+r].Fill(my_mcp_ob.E(), my_pfo_ob.E())
                        if has_clu_ob:
                            hists2d["clu_v_mcp_E"+r].Fill(my_mcp_ob.E(), my_clu_ob.getEnergy())
                        hists2d["sim_v_mcp_E"+r].Fill(my_mcp_ob.E(), sim_E)
                        hists2d["dig_v_mcp_E"+r].Fill(my_mcp_ob.E(), dig_E)
                        hists2d["rec_v_mcp_E"+r].Fill(my_mcp_ob.E(), rec_E)

        # Bookkeeping
        progress(time.time() - t0, i_file + 1, len(fnames))
        reader.close()


    # mention cluster and PFO multiplicity
    total = sum(n_clu_ob_dict.values())
    print(f"Total events = {total}")
    # for n_clu_ob, n_occurences in sorted(n_clu_ob_dict.items()):
    #     print(f"N(clu) = {n_clu_ob} occurred {n_occurences}x ({n_occurences/total * 100:.1f}%)")
    # for n_pfo_ob, n_occurences in sorted(n_pfo_ob_dict.items()):
    #     print(f"N(pfo) = {n_pfo_ob} occurred {n_occurences}x ({n_occurences/total * 100:.1f}%)")

    # ############## MANIPULATE, PRETTIFY, AND SAVE HISTOGRAMS #############################

    # Draw basic distributions
    for var in variables:
        h_to_plot = {}
        for obj in objects:
            h_to_plot[obj] = hists[obj+"_"+var]
        plotHistograms(h_to_plot, f"plots/calo/comp_{var}.png", variables[var]["title"], "Count")

    # Make 2D plots comparing true v reco quantities
    for hist in hists2d:
        c = ROOT.TCanvas("c_%s"%hist, "c")
        hists2d[hist].Draw("colz")
        var = hist.split("_")[-2]
        obj = hist.split("_")[0]
        hists2d[hist].GetXaxis().SetTitle("True "+settings['labelname'][obj_type]+" "+variables[var]["title"])
        hists2d[hist].GetYaxis().SetTitle(objects[obj]+" "+variables[var]["title"])
        c.SetRightMargin(0.18)
        c.SetLogz()
        c.SaveAs(f"plots/calo/{hist}.png")

    return particles

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

def getEnergyClu(event):
    energy = 0
    cluCollection = getCollection(event, "PandoraClusters")
    for clu in cluCollection:
        if clu.getEnergy() > energy:
            energy = clu.getEnergy()
    return energy

def getEnergyPfo(event, obj_type):
    energy = 0
    pfoCollection = getCollection(event, "PandoraPFOs")
    for pfo in pfoCollection:
        pfo_tlv = getTLV(pfo)
        if abs(pfo.getType()) in settings['pdgid'][obj_type]:
            if pfo_tlv.E() > energy:
                energy = pfo_tlv.E()
    return energy

if __name__ == "__main__":
    main()
