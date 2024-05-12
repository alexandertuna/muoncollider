"""
Recreating the digitization process by-hand.

This closely follows the actual digitization, which is described in Marlin:
https://www.desy.de/~dudarboh/marlinreco_doc/html/RealisticCaloDigi_8cc_source.html
https://www.desy.de/~dudarboh/marlinreco_doc/html/RealisticCaloDigiSilicon_8cc_source.html
https://www.desy.de/~dudarboh/marlinreco_doc/html/RealisticCaloDigiScinPpd_8cc_source.html

And our configuration of these processes:
https://github.com/madbaron/SteeringMacros/blob/master/k4Reco/steer_reco_CONDOR.py
"""

import pyLCIO # type: ignore
from pyLCIO import EVENT, UTIL

import math
import os
from numpy.random import poisson
from numpy.random import binomial
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

FNAMES = (
    "/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10000.slcio",
    # "/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10100.slcio",
    # "/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10200.slcio",
    # "/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10300.slcio",
    # "/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10400.slcio",
    # "/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10500.slcio",
    # "/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10600.slcio",
    # "/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10700.slcio",
    # "/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10800.slcio",
    # "/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10900.slcio",
)

MIP = 0
GEVDEP = 1
NPE = 2

def main() -> None:

    parquet_name = "compare_sim_digi.parquet"

    study = SimDigiComparison()
    for fname in FNAMES:
        study.analyze(fname)

    print("Converting dict to pandas.DataFrame ...")
    df = pd.DataFrame(study.data)

    print(f"Writing {parquet_name} ...")
    df.to_parquet(parquet_name)

    study.plot()


class systems:
    ecal_barrel = 20
    ecal_endcap = 21
    hcal_barrel = 10
    hcal_endcap = 11
    ecal = [ecal_barrel, ecal_endcap]
    hcal = [hcal_barrel, hcal_endcap]


class DigitizedHit:

    def __init__(self, hit):
        self.hit = hit
        self.system = hit.getCellID0() & 0b11111
        self.time, self.energy_ = self.StandardIntegration()
        self.energy = self.EnergyDigi(self.energy_)
        self.passed = self.energy > self.threshold()


    def StandardIntegration(self) -> [float, float]:
        _time_correctForPropagation = True if self.system in systems.ecal else False
        _time_windowMin = -0.5 if self.system in systems.ecal else -10.0
        _time_windowMax = 10 if self.system in systems.ecal else 100.0
        c_light = 299.792458 # mm/ns

        timeCorrection = 0.0
        if _time_correctForPropagation:
            r = 0
            for i in range(3):
                r += pow(self.hit.getPosition()[i], 2.0)
            timeCorrection = math.sqrt(r) / c_light # [speed of light in mm/ns]

        energySum = 0
        earliestTime = 9e9
        for i in range(self.hit.getNMCContributions()):
            timei   = self.hit.getTimeCont(i)
            energyi = self.hit.getEnergyCont(i)
            relativetime = timei - timeCorrection
            if relativetime > _time_windowMin and relativetime < _time_windowMax:
                energySum += energyi
                if relativetime < earliestTime:
                    earliestTime = relativetime

        if earliestTime > _time_windowMin and earliestTime < _time_windowMax: #accept this hit
            return self.SmearTime(earliestTime), energySum
        return 0, 0


    def SmearTime(self, time: float) -> float:
        _time_resol = 0
        if _time_resol > 0:
            raise Exception("Not implemented")
        return time


    def EnergyDigi(self, energy: float) -> float:
        _misCalib_uncorrel = 0.0
        _misCalib_correl = 0.0
        _elec_rangeMip = 15000.0 if self.system in systems.ecal else 0.0
        _elec_noiseMip = 0.0
        _deadCell_fraction = 0.0

        e_out = energy
        e_out = self.digitiseDetectorEnergy(energy)

        oneMipInMyUnits = self.convertEnergy(1.0, MIP)
        if _elec_rangeMip > 0.0:
            e_out = min(e_out, _elec_rangeMip * oneMipInMyUnits)

        if _misCalib_uncorrel > 0:
            raise Exception("Not implemented")
        if _misCalib_correl > 0:
            raise Exception("Not implemented")
        if _elec_noiseMip > 0.0:
            raise Exception("Not implemented")
        if _deadCell_fraction > 0:
            raise Exception("Not implemented")
        return e_out


    def digitiseDetectorEnergy(self, energy: float) -> float:
        func = self.digitiseDetectorEnergy_RealisticCaloDigiSilicon if self.system in systems.ecal else self.digitiseDetectorEnergy_RealisticCaloDigiScinPpd
        return func(energy)


    def digitiseDetectorEnergy_RealisticCaloDigiSilicon(self, energy: float) -> float:
        if energy == 0:
            return 0
        _ehEnergy = 3.6
        _calib_mip = 0.0001575
        smeared_energy = energy
        if _ehEnergy > 0:
            nehpairs = 1e9 * energy / _ehEnergy
            smeared_energy *= poisson(nehpairs) / nehpairs
        return smeared_energy / _calib_mip


    def digitiseDetectorEnergy_RealisticCaloDigiScinPpd(self, energy: float) -> float:
        _PPD_pe_per_mip = 15.0
        _calib_mip = 0.0004925 # barrel ### 0.0004725 endcap
        _PPD_n_pixels = 2000.0
        _pixSpread = 0.0
        npe = energy * _PPD_pe_per_mip / _calib_mip
        if _PPD_n_pixels > 0:
            npe = _PPD_n_pixels * (1.0 - math.exp(-npe/_PPD_n_pixels))
            p = npe / _PPD_n_pixels
            # npe = binomial(_PPD_n_pixels, p)
            if _pixSpread > 0:
                raise Exception("Not implemented")
        return npe


    def convertEnergy(self, energy: float, inUnit: int) -> float:
        func = self.convertEnergy_RealisticCaloDigiSilicon if self.system in systems.ecal else self.convertEnergy_RealisticCaloDigiScinPpd
        return func(energy, inUnit)


    def convertEnergy_RealisticCaloDigiSilicon(self, energy: float, inUnit: int) -> float:
        _calib_mip = 0.0001575
        if inUnit == MIP:
            return energy
        elif inUnit == GEVDEP:
            return energy / _calib_mip;
        raise Exception("Not implemented")


    def convertEnergy_RealisticCaloDigiScinPpd(self, energy: float, inUnit: int) -> float: # NPE
        _PPD_pe_per_mip = 15.0
        _calib_mip = 0.0004925 # barrel ### 0.0004725 endcap
        if inUnit == NPE:
            return energy
        elif inUnit == MIP:
            return _PPD_pe_per_mip * energy
        elif inUnit == GEVDEP:
            return _PPD_pe_per_mip * energy / _calib_mip
        raise Exception("Not implemented")


    def threshold(self) -> float:
        func = self.threshold_RealisticCaloDigiSilicon if self.system in systems.ecal else self.threshold_RealisticCaloDigiScinPpd
        return func()


    def threshold_RealisticCaloDigiSilicon(self) -> float:
        _threshold_value = 5e-05
        _threshold_iunit = GEVDEP
        return self.convertEnergy_RealisticCaloDigiSilicon(_threshold_value, _threshold_iunit)


    def threshold_RealisticCaloDigiScinPpd(self) -> float:
        _threshold_value = 0.5
        _threshold_iunit = MIP
        return self.convertEnergy_RealisticCaloDigiScinPpd(_threshold_value, _threshold_iunit)


class SimDigiComparison:


    def __init__(self):
        self.data = {
            "n_mc_contributions": [],
            "system": [],
            "digit_energy_manual": [],
            "digit_energy_pandora": [],
            "digit_time_manual": [],
            "digit_time_pandora": [],
            "digit_exists_manual": [],
            "digit_exists_pandora": [],
        }
        self.parquet_name = "compare_sim_digi.parquet"


    def analyze(self, fname: str) -> None:
        print(f"Analyzing {fname} ...")
        reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
        reader.open(fname)
        for i_event, event in enumerate(reader):

            event_of_interest = [0, 1, 2, 3, 4, 5, 6]
            # event_of_interest = range(1000)
            if i_event not in event_of_interest:
                continue
            if i_event > max(event_of_interest):
                break

            print(f"Event {i_event}")

            for calo in ["E", "H"]:

                colname = f"{calo}CalBarrelCollection"
                try:
                    col = event.getCollection(colname)
                    relations_sim = [el.getTo()   for el in event.getCollection(f"{calo}calBarrelRelationsSimDigi")]
                    relations_dig = [el.getFrom() for el in event.getCollection(f"{calo}calBarrelRelationsSimDigi")]
                except:
                    continue

                for i_el, el in enumerate(col):
                    digit_exp = DigitizedHit(el)
                    digit_energy, digit_time = -1, -1
                    if el in relations_sim:
                        index = relations_sim.index(el)
                        digit_energy, digit_time = relations_dig[index].getEnergy(), relations_dig[index].getTime()

                    for name, var in [
                            ("n_mc_contributions", el.getNMCContributions()),
                            ("system", digit_exp.system),
                            ("digit_energy_manual", digit_exp.energy),
                            ("digit_energy_pandora", digit_energy),
                            ("digit_time_manual", digit_exp.time),
                            ("digit_time_pandora", digit_time),
                            ("digit_exists_manual", digit_exp.passed),
                            ("digit_exists_pandora", el in relations_sim),
                    ]:
                        self.data[name].append(var)


    def plot(self):
        print(f"Reading {self.parquet_name} ...")
        df = pd.read_parquet(self.parquet_name)
        print(df)

        with PdfPages("digitization.pdf") as pdf:
            for system in [10, 20]:
                for var in ["energy", "time"]:

                    condition = (df.system == system) & df.digit_exists_manual & df.digit_exists_pandora
                    subset = df[condition]

                    # 1D comparison
                    fig, ax = plt.subplots(figsize=(4, 4))
                    bins = np.linspace(-0.5, 0.5, 200)
                    ax.hist((subset[f"digit_{var}_manual"] - subset[f"digit_{var}_pandora"]) / subset[f"digit_{var}_pandora"], bins=bins)
                    ax.set_xlabel(f"System {system} {var} (byhand - pandora) / pandora")
                    ax.tick_params(top=True, right=True)
                    pdf.savefig()
                    plt.close()

                    # 2D comparison
                    fig, ax = plt.subplots(figsize=(4, 4))
                    min_val = min([subset[f"digit_{var}_pandora"].min(), subset[f"digit_{var}_manual"].min()])
                    max_val = max([subset[f"digit_{var}_pandora"].max(), subset[f"digit_{var}_manual"].max()])
                    if var == "energy":
                        bins = np.logspace(math.log(min_val, 10), math.log(max_val, 10), 100)
                    else:
                        bins = np.linspace(min_val, max_val, 100)
                    ax.hist2d(subset[f"digit_{var}_pandora"], subset[f"digit_{var}_manual"], bins=[bins, bins], cmap="rainbow", cmin=1e-7)
                    ax.set_xlabel(f"System {system} {var} pandora")
                    ax.set_ylabel(f"System {system} {var} byhand")
                    ax.tick_params(top=True, right=True)
                    if var == "energy":
                        ax.semilogx()
                        ax.semilogy()
                    pdf.savefig()
                    plt.close()
                

if __name__ == "__main__":
    main()
