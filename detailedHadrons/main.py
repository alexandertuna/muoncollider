import pyLCIO
from pyLCIO import EVENT, UTIL
import glob
import collections
import ctypes
import math
import time
from dataclasses import dataclass

exec(open("./plotHelper.py").read())

# ############## SETUP #############################

# Prevent ROOT from drawing while you're running -- good for slow remote servers
# Instead, save files and view them with an sftp client like Fetch (feel free to ask me for my UTK license)
ROOT.gROOT.SetBatch()

# Set up some options
max_events = -1
obj_type = "ne"
magnetic_field = 5.00
calibration_mip = 0.0001575
calibration_mip_to_reco = 0.0066150 # 0.00641222630095
sampling_scaling = calibration_mip_to_reco/calibration_mip

hcal_calibration_mip_b = 0.0004925
hcal_calibration_mip_to_reco_b = 0.024625 # 0.0287783798145
hcal_sampling_scaling_b = hcal_calibration_mip_to_reco_b / hcal_calibration_mip_b
print(f"hcal_sampling_scaling_b = {hcal_sampling_scaling_b}")

hcal_calibration_mip_e = 0.0004725
hcal_calibration_mip_to_reco_e = 0.024625 # 0.0285819096797
hcal_sampling_scaling_e = hcal_calibration_mip_to_reco_e / hcal_calibration_mip_e
print(f"hcal_sampling_scaling_e = {hcal_sampling_scaling_e}")

append = "tmp" # time.strftime("%Y_%m_%d_%Hh%Mm%Ss")

# Set up hit encoder/decoder
encoding = 'system:0:5,side:5:-2,module:7:8,stave:15:4,layer:19:9,submodule:28:4,x:32:-16,y:48:-16' # col.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
decoder = UTIL.BitField64(encoding)

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
print("Running on", settings["labelname"][obj_type])

# Gather input files
# Note: these are using the path convention from the singularity command in the MuCol tutorial (see README)
samples = glob.glob(settings["fnames"][obj_type])
fnames = []
for s in samples:
    fnames += glob.glob(f"{s}/*.slcio")
fnames = fnames[:5]
print("Found %i files."%len(fnames))

# Define good particle
def isGood(tlv):
    if abs(tlv.Eta()) < 2:
        return True
    return False

# Perform matching between two TLVs
def isMatched(tlv1, tlv2, req_pt = True):
    if tlv1.DeltaR(tlv2) > 0.1: return False
    if req_pt:
        drelpt = abs(tlv1.Perp()-tlv2.Perp())/tlv2.Perp()
        if drelpt > 0.1*tlv2.Perp()/100: return False # Require 10% at 100, 20% at 200, ...
    return True

def getClusterEta(cluster):
    theta = cluster.getITheta()
    return -1*math.ln(math.tan(theta/2))

# Progress bar
def progress(time_diff, nprocessed, ntotal):
    nprocessed, ntotal = float(nprocessed), float(ntotal)
    rate = (nprocessed+1)/time_diff
    msg = "\r > %6i / %6i | %2i%% | %8.4fHz | %6.1fm elapsed | %6.1fm remaining"
    msg = msg % (nprocessed, ntotal, 100*nprocessed/ntotal, rate, time_diff/60, (ntotal-nprocessed)/(rate*60))
    print(msg)

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
        hists[obj+"_"+var] = ROOT.TH1F(obj+"_"+var, objects[obj], variables[var]["nbins"], variables[var]["xmin"], variables[var]["xmax"])

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

# convert hcal digit energy to GeV
def reconstructHcalEnergy(hit):
    decoder.setValue((hit.getCellID0() & 0xffffffff) |
                     (hit.getCellID1() << 32))
    energy = hit.getEnergy()
    system = decoder['system'].value()
    assert system in systems.hcal
    barrel = system == systems.hcal_barrel
    # calibrCoeff = hcal_calibration_mip_to_reco_b if barrel else hcal_calibration_mip_to_reco_e
    calibrCoeff = calib.mip_to_reco.hcal_barrel if barrel else calib.mip_to_reco.hcal_endcap
    r = 0.95
    if (energy < r*ppd.npix):
        energy = -ppd.npix * math.log ( 1. - ( energy / ppd.npix ) )
    else:
        energy = 1/(1-r)*(energy-r*ppd.npix)-ppd.npix*math.log(1-r)
    energy /= ppd.mipPe
    energy *= calibrCoeff
    return energy

# ############## LOOP OVER EVENTS AND FILL HISTOGRAMS  #############################
# Loop over events
reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
# reader.setReadCollectionNames(["MCParticle", "PandoraPFOs", "ECalBarrelCollection", "ECalEndcapCollection", "EcalBarrelCollectionDigi", "EcalEndcapCollectionDigi", "EcalBarrelCollectionRec", "EcalEndcapCollectionRec", "PandoraClusters"])
i = 0
boop = 0
t0 = time.time()
n_clu_ob_dict = collections.defaultdict(int)
n_pfo_ob_dict = collections.defaultdict(int)
for i_file, f in enumerate(fnames):
    reader.open(f)
    if max_events > 0 and i >= max_events: break

    for event in reader:

        if max_events > 0 and i >= max_events: break
        if i%100 == 0: progress(time.time() - t0, i_file + 1, len(fnames))

        # Get the collections we care about
        mcpCollection = event.getCollection("MCParticle")
        pfoCollection = event.getCollection("PandoraPFOs")
        simCollection_b = event.getCollection("ECalBarrelCollection")
        simCollection_e = event.getCollection("ECalEndcapCollection")
        hcalSimCollection_b = event.getCollection("HCalBarrelCollection")
        hcalSimCollection_e = event.getCollection("HCalEndcapCollection")
        yokeSimCollection_b = event.getCollection("YokeBarrelCollection")
        yokeSimCollection_e = event.getCollection("YokeEndcapCollection")
        try: digCollection_b = event.getCollection("EcalBarrelCollectionDigi")
        except: digCollection_b = None
        try: digCollection_e = event.getCollection("EcalEndcapCollectionDigi")
        except: digCollection_e = None
        try: recCollection_b = event.getCollection("EcalBarrelCollectionRec")
        except: recCollection_b = None
        try: recCollection_e = event.getCollection("EcalEndcapCollectionRec")
        except: recCollection_e = None
        cluCollection = event.getCollection("PandoraClusters")

        try: hcalDigCollection_b = event.getCollection("HcalBarrelCollectionDigi")
        except: hcalDigCollection_b = None
        try: hcalDigCollection_e = event.getCollection("HcalEndcapCollectionDigi")
        except: hcalDigCollection_e = None
        try: hcalRecCollection_b = event.getCollection("HcalBarrelCollectionRec")
        except: hcalRecCollection_b = None
        try: hcalRecCollection_e = event.getCollection("HcalEndcapCollectionRec")
        except: hcalRecCollection_e = None

        # Make counter variables
        n_mcp_ob = 0
        n_pfo_ob = 0
        n_clu_ob = 0
        has_mcp_ob = False
        has_pfo_ob = False
        has_clu_ob = False
        my_pfo_ob = None
        my_clu_ob = None
        my_mcp_ob = None

        # Loop over the truth objects and fill histograms
        for mcp in mcpCollection:
            mcp_tlv = getTLV(mcp)
            if abs(mcp.getPDG()) in settings['pdgid'][obj_type] and mcp.getGeneratorStatus()==1 and isGood(mcp_tlv):
                has_mcp_ob = True
                n_mcp_ob += 1
                my_mcp_ob = mcp_tlv

        # Loop over the reconstructed clusters and fill histograms
        # If there are multiple, it'll keep the one with the higher pT
        for clu in cluCollection:
            #clu_tlv = ROOT.TLorentzVector()
            #clu_tlv.SetE(clu.getEnergy())
            #clu_tlv.SetTheta(clu.getITheta())
            #clu_tlv.SetPhi(clu.getIPhi())
            if has_mcp_ob: # and isMatched(clu_tlv, my_mcp_ob, req_pt = False):
                n_clu_ob += 1
                has_clu_ob = True
                if n_clu_ob == 1:
                    my_clu_ob = clu
                elif n_clu_ob > 1 and clu.getEnergy() > my_clu_ob.getEnergy():
                    my_clu_ob = clu

        # Loop over the reconstructed objects and fill histograms
        # If there are multiple, it'll keep the one with the higher pT
        for pfo in pfoCollection:
            pfo_tlv = getTLV(pfo)

            if abs(pfo.getType()) in settings['pdgid'][obj_type]:
                if has_mcp_ob: # and isMatched(pfo_tlv, my_mcp_ob, req_pt = False):
                    n_pfo_ob += 1
                    has_pfo_ob = True
                    if n_pfo_ob == 1:
                        my_pfo_ob = pfo_tlv
                    elif n_pfo_ob > 1 and pfo_tlv.E() > my_pfo_ob.E():
                        my_pfo_ob = pfo_tlv

        # Loop over sim calo hits and sum
        sim_E = 0
        for sim in simCollection_b or []:
            sim_E += sim.getEnergy()*sampling_scaling.ecal
        for sim in simCollection_e or []:
            sim_E += sim.getEnergy()*sampling_scaling.ecal
        for sim in hcalSimCollection_b or []:
            sim_E += sim.getEnergy()*sampling_scaling.hcal_barrel
        for sim in hcalSimCollection_e or []:
            sim_E += sim.getEnergy()*sampling_scaling.hcal_endcap

        yoke_E = 0
        for col in [yokeSimCollection_b,
                    yokeSimCollection_e]:
            if col and len(col) > 0:
                yoke_E += 70.1 * sum(sim.getEnergy() for sim in col)

        # Loop over digi hits and sum
        dig_E = 0
        if digCollection_b:
            # for dig in digCollection_b: dig_E += dig.getEnergy()*calibration_mip_to_reco
            for dig in digCollection_b: dig_E += dig.getEnergy()*calib.mip_to_reco.ecal
        if digCollection_e:
            # for dig in digCollection_e: dig_E += dig.getEnergy()*calibration_mip_to_reco
            for dig in digCollection_e: dig_E += dig.getEnergy()*calib.mip_to_reco.ecal
        for dig in hcalDigCollection_b or []:
            dig_E += reconstructHcalEnergy(dig)
        for dig in hcalDigCollection_e or []:
            dig_E += reconstructHcalEnergy(dig)

        # Loop over reco hits and sum
        rec_E = 0
        if recCollection_b:
            for rec in recCollection_b: rec_E += rec.getEnergy()
        if recCollection_e:
            for rec in recCollection_e: rec_E += rec.getEnergy()
        for rec in hcalRecCollection_b or []: rec_E += rec.getEnergy()
        for rec in hcalRecCollection_e or []: rec_E += rec.getEnergy()

        # tuna debug
        if boop < 20 and my_mcp_ob is not None:
            E, eta, phi = my_mcp_ob.E(), my_mcp_ob.Eta(), my_mcp_ob.Phi()
            if abs(eta) < 1.0:
                print(f"i = {i}, E = {int(E)}, sim_E = {int(sim_E)}, dig_E = {int(dig_E)}, rec_E = {int(rec_E)}, dig_E/rec_E = {int(dig_E/rec_E)}, eta = {eta:.2f}, phi = {phi:.2f}, npfo = {n_pfo_ob}, clu_ob = {my_clu_ob is not None}, yoke_E = {yoke_E:.6f}")
            if i <= 9:
                n_sim_b = 0 if not simCollection_b else len(simCollection_b)
                n_sim_e = 0 if not simCollection_e else len(simCollection_e)
                n_dig_b = 0 if not digCollection_b else len(digCollection_b)
                n_dig_e = 0 if not digCollection_e else len(digCollection_e)
                n_hcalSim_b = 0 if not hcalSimCollection_b else len(hcalSimCollection_b)
                n_hcalSim_e = 0 if not hcalSimCollection_e else len(hcalSimCollection_e)
                n_hcalDig_b = 0 if not hcalDigCollection_b else len(hcalDigCollection_b)
                n_hcalDig_e = 0 if not hcalDigCollection_e else len(hcalDigCollection_e)
                n_hcalRec_b = 0 if not hcalRecCollection_b else len(hcalRecCollection_b)
                n_hcalRec_e = 0 if not hcalRecCollection_e else len(hcalRecCollection_e)
                print(f"N(ecal sim b) = {n_sim_b}")
                print(f"N(ecal sim e) = {n_sim_e}")
                print(f"N(hcal sim b) = {n_hcalSim_b}")
                print(f"N(hcal sim e) = {n_hcalSim_e}")
                print(f"N(ecal dig b) = {n_dig_b}")
                print(f"N(ecal dig e) = {n_dig_e}")
                print(f"N(hcal dig b) = {n_hcalDig_b}")
                print(f"N(hcal dig e) = {n_hcalDig_e}")
                print(f"N(hcal rec b) = {n_hcalRec_b}")
                print(f"N(hcal rec e) = {n_hcalRec_e}")
            # if i == 9:
            #     break
            boop += 1

        # Only make plots for events with isGood mcps
        if has_mcp_ob:
            n_clu_ob_dict[n_clu_ob] += 1
            n_pfo_ob_dict[n_pfo_ob] += 1
            hists["mcp_E"].Fill(my_mcp_ob.E())
            if has_pfo_ob:
                hists["pfo_E"].Fill(my_pfo_ob.E())
                #hists2d["pfo_v_mcp_E"].Fill(my_mcp_ob.E(), my_pfo_ob.E())
            if has_clu_ob:
                hists["clu_E"].Fill(my_clu_ob.getEnergy())
                #hists2d["clu_v_mcp_E"].Fill(my_mcp_ob.E(), my_clu_ob.getEnergy())
            hists["sim_E"].Fill(sim_E)
            hists["dig_E"].Fill(dig_E)
            hists["rec_E"].Fill(rec_E)
            #hists2d["sim_v_mcp_E"].Fill(my_mcp_ob.E(), sim_E)
            #hists2d["dig_v_mcp_E"].Fill(my_mcp_ob.E(), dig_E)
            #hists2d["rec_v_mcp_E"].Fill(my_mcp_ob.E(), rec_E)

            # Print out 2D distributions per eta range
            for r in ranges:
                r1 = r.replace("p", ".").strip("_")
                low_eta = r1.split("to")[0]
                high_eta = r1.split("to")[1]
                selection_string = f"my_mcp_ob.Eta()>={low_eta} and my_mcp_ob.Eta()<{high_eta}"
                if eval(selection_string):
                    if has_pfo_ob: hists2d["pfo_v_mcp_E"+r].Fill(my_mcp_ob.E(), my_pfo_ob.E())
                    if has_clu_ob: hists2d["clu_v_mcp_E"+r].Fill(my_mcp_ob.E(), my_clu_ob.getEnergy())
                    hists2d["sim_v_mcp_E"+r].Fill(my_mcp_ob.E(), sim_E)
                    hists2d["dig_v_mcp_E"+r].Fill(my_mcp_ob.E(), dig_E)
                    hists2d["rec_v_mcp_E"+r].Fill(my_mcp_ob.E(), rec_E)

        i+=1
    reader.close()


# mention cluster and PFO multiplicity
total = sum(n_clu_ob_dict.values())
print(f"Total events = {total}")
for n_clu_ob, n_occurences in sorted(n_clu_ob_dict.items()):
    print(f"N(clu) = {n_clu_ob} occurred {n_occurences}x ({n_occurences/total * 100:.1f}%)")
for n_pfo_ob, n_occurences in sorted(n_pfo_ob_dict.items()):
    print(f"N(pfo) = {n_pfo_ob} occurred {n_occurences}x ({n_occurences/total * 100:.1f}%)")


# ############## MANIPULATE, PRETTIFY, AND SAVE HISTOGRAMS #############################


# Draw basic distributions
for var in variables:
    h_to_plot = {}
    for obj in objects:
        h_to_plot[obj] = hists[obj+"_"+var]
    plotHistograms(h_to_plot, f"plots/calo/comp_{var}_{append}.png", variables[var]["title"], "Count")


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
    c.SaveAs(f"plots/calo/{hist}_{append}.png")

