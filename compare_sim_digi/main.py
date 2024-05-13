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

from dataclasses import dataclass
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
    "/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10100.slcio",
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

    study = SimDigiComparison(FNAMES)
    study.analyze()
    study.write_dataframe()
    study.plot()


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
    yoke = [yoke_barrel, yoke_endcap]


class SimDigiComparison:

    def __init__(self, fnames: list[str]) -> None:
        self.fnames = fnames
        self.data = self.default_dict()
        self.parquet_name = "compare_sim_digi.parquet"


    def default_dict(self):
        return {
            "n_mc_contributions": [],
            "system": [],
            "digit_energy_manual": [],
            "digit_energy_pandora": [],
            "digit_time_manual": [],
            "digit_time_pandora": [],
            "digit_exists_manual": [],
            "digit_exists_pandora": [],
        }


    def analyze(self) -> None:
        self.data = self.default_dict()
        for fname in self.fnames:
            data = self.analyze_file(fname)
            for key in data:
                self.data[key].extend(data[key])
        print("Converting dict to pandas.DataFrame ...")
        self.df = pd.DataFrame(self.data)


    def write_dataframe(self) -> None:
        print(f"Writing {self.parquet_name} ...")
        self.df.to_parquet(self.parquet_name)


    def analyze_file(self, fname: str) -> None:
        print(f"Analyzing {fname} ...")

        data = self.default_dict()
        reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
        reader.open(fname)

        for i_event, event in enumerate(reader):

            # events_of_interest = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            # # events_of_interest = range(1000)
            # if i_event not in events_of_interest:
            #     continue
            # if i_event > max(events_of_interest):
            #     break

            print(f"Event {i_event}")

            for calo, part in [
                    ("E", "Barrel"),
                    ("E", "Endcap"),
                    ("H", "Barrel"),
                    ("H", "Endcap"),
            ]:
                colname = f"{calo}Cal{part}Collection"
                try:
                    col = event.getCollection(colname)
                    sim2digit = {rel.getTo(): rel.getFrom() for rel in event.getCollection(f"{calo}cal{part}RelationsSimDigi")}
                except:
                    continue

                for i_sim, sim in enumerate(col):
                    digit_exp = DigitizedHit(sim)
                    digit_energy, digit_time = -1, -1
                    if sim in sim2digit:
                        digit_energy, digit_time = sim2digit[sim].getEnergy(), sim2digit[sim].getTime()

                    for name, var in [
                            ("n_mc_contributions", sim.getNMCContributions()),
                            ("system", digit_exp.system),
                            ("digit_energy_manual", digit_exp.energy),
                            ("digit_energy_pandora", digit_energy),
                            ("digit_time_manual", digit_exp.time),
                            ("digit_time_pandora", digit_time),
                            ("digit_exists_manual", digit_exp.passed),
                            ("digit_exists_pandora", sim in sim2digit),
                    ]:
                        data[name].append(var)

        print("Closing ...")
        reader.close()
        return data


    def human_readable(self, system):
        if system == systems.hcal_barrel:
            return "HCal Barrel"
        elif system == systems.hcal_endcap:
            return "HCal Endcap"
        elif system == systems.ecal_barrel:
            return "ECal Barrel"
        elif system == systems.ecal_endcap:
            return "ECal Endcap"
        raise Exception(f"Unknown system: {system}")


    def unit(self, var, system):
        if var == "energy":
            if system in systems.ecal:
                return "MIP"
            else:
                return "NPE"
        elif var == "time":
            return "ns"
        raise Exception(f"Unknown variable: {var}")


    def plot(self):
        print(f"Reading {self.parquet_name} ...")
        df = pd.read_parquet(self.parquet_name)
        print(df)

        with PdfPages("digitization.pdf") as pdf:
            for system in systems.ecal + systems.hcal:
                for var in ["energy", "time"]:

                    condition = (df.system == system) & df.digit_exists_manual & df.digit_exists_pandora
                    subset = df[condition]
                    manual = subset[f"digit_{var}_manual"]
                    pandora = subset[f"digit_{var}_pandora"]
                    if var == "energy" and system in systems.hcal:
                        pandora = pandora + np.random.rand(len(pandora)) - 0.5

                    # 1D comparison
                    fig, ax = plt.subplots(figsize=(4, 4))
                    bins = np.linspace(-0.5, 0.5, 200)
                    ax.hist((manual - pandora) / pandora, bins=bins)
                    ax.set_xlabel(f"Digit {var}: (by hand - pandora) / pandora")
                    ax.set_ylabel(f"Number of hits")
                    ax.tick_params(top=True, right=True)
                    ax.text(0.08, 0.8, self.human_readable(system), transform=ax.transAxes)
                    fig.subplots_adjust(bottom=0.14, left=0.20, right=0.95, top=0.95)
                    pdf.savefig()
                    plt.close()

                    # 2D comparison
                    fig, ax = plt.subplots(figsize=(4, 4))
                    the_unit = self.unit(var, system)
                    min_val = min([pandora.min(), manual.min()])
                    max_val = max([pandora.max(), manual.max()])
                    linex = liney = [min_val, max_val]
                    if var == "energy":
                        bins = np.logspace(math.log(min_val, 10), math.log(max_val, 10), 100)
                    else:
                        bins = np.linspace(min_val, max_val, 100)
                    counts, xedges, yedges, im = ax.hist2d(pandora, manual, bins=[bins, bins], cmap="rainbow", cmin=1e-7)
                    ax.set_xlabel(f"Digit {var} [{the_unit}] (pandora)")
                    ax.set_ylabel(f"Digit {var} [{the_unit}] (by hand)")
                    ax.tick_params(top=True, right=True)
                    if var == "energy":
                        ax.semilogx()
                        ax.semilogy()
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.set_label("Number of hits")
                    ax.text(0.08, 0.8, self.human_readable(system), transform=ax.transAxes)
                    # ax.plot(linex, liney, color="gray", linewidth=1, linestyle="dashed")
                    fig.subplots_adjust(bottom=0.14, left=0.15, right=0.85, top=0.95)
                    pdf.savefig()
                    plt.close()
                

class DigitizedHit:

    def __init__(self, hit):
        self.hit = hit
        self.system = hit.getCellID0() & 0b11111
        self.time, self.energy_ = self.StandardIntegration()
        self.energy = self.EnergyDigi(self.energy_)
        self.passed = self.energy > self.threshold()
        if self.system not in systems.ecal + systems.hcal:
            raise Exception(f"Unknown system: {self.system}")


    def is_yoke(self):
        return self.system in systems.yoke


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
        _calib_mip = 0.0004925 if self.system == systems.hcal_barrel else 0.0004725
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



if __name__ == "__main__":
    main()
