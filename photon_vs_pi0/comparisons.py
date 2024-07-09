import argparse
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import logging
logging.basicConfig(level=logging.INFO)

PHOTON = 0
PIZERO = 1
PARTICLES = [
    PHOTON,
    PIZERO,
]

def main():
    ops = options()
    comp = ComparisonPlots2(ops.photon, ops.pizero, ops.output)
    comp.plot()


def options():
    parser = argparse.ArgumentParser(description="Event Displays")
    parser.add_argument("--photon", help="Input file (photons)", required=True)
    parser.add_argument("--pizero", help="Input file (pizeros)", required=True)
    parser.add_argument("-o", "--output", help="Output file", required=True)
    return parser.parse_args()


class ComparisonPlots2:
    def __init__(self, photon_glob: str, pizero_glob, output_file: str) -> None:
        self.files = {
            PHOTON: glob.glob(photon_glob),
            PIZERO: glob.glob(pizero_glob),
        }
        self.features = {
            PHOTON: np.concatenate([np.load(fname) for fname in self.files[PHOTON]]),
            PIZERO: np.concatenate([np.load(fname) for fname in self.files[PIZERO]]),
        }
        self.photon_files = glob.glob(photon_glob)
        self.pizero_files = glob.glob(pizero_glob)
        self.output_file = output_file
        self.photon_features = np.concatenate([np.load(fname) for fname in self.photon_files])
        self.pizero_features = np.concatenate([np.load(fname) for fname in self.pizero_files])
        print(self.photon_features.shape)
        print(self.pizero_features.shape)
        _, z, x, y = self.photon_features.shape
        self.x_all = slice(0, x)
        self.y_all = slice(0, y)
        self.z_all = slice(0, z)
        self.make_new_features()

    def calculate_ratios(
        self,
        images: np.ndarray,
        slice_numer_x: slice = None,
        slice_numer_y: slice = None,
        slice_numer_z: slice = None,
        slice_denom_x: slice = None,
        slice_denom_y: slice = None,
        slice_denom_z: slice = None,
    ):
        slices_numer = (
            slice(None),
            slice_numer_z or self.z_all,
            slice_numer_x or self.x_all,
            slice_numer_y or self.y_all,
        )
        slices_denom = (
            slice(None),
            slice_denom_z or self.z_all,
            slice_denom_x or self.x_all,
            slice_denom_y or self.y_all,
        )
        numer = np.sum(images[slices_numer], axis=(1, 2, 3))
        denom = np.sum(images[slices_denom], axis=(1, 2, 3))
        return numer / denom

    def make_new_features(self) -> None:
        self.make_feature_ratio_x()
        self.make_feature_r_y()
        self.make_feature_r_front()
        self.make_feature_r_back()
        self.make_feature_w_x()
        self.make_feature_w_y()
        self.make_feature_w_x_max()
        self.make_feature_w_y_max()
        self.make_feature_d_e()
        self.make_feature_r_e()

    def make_feature_ratio_x(self) -> None:
        self.ratio_x0208_y0808 = {}
        self.ratio_x0824_y0808 = {}
        for PARTICLE in PARTICLES:
            self.ratio_x0208_y0808[PARTICLE] = self.calculate_ratios(
                self.features[PARTICLE],
                slice_numer_x=slice(19, 21),
                slice_denom_x=slice(16, 24),
                slice_numer_y=slice(16, 24),
                slice_denom_y=slice(16, 24),
            )
            self.ratio_x0824_y0808[PARTICLE] = self.calculate_ratios(
                self.features[PARTICLE],
                slice_numer_x=slice(16, 24),
                slice_denom_x=slice( 8, 32),
                slice_numer_y=slice(16, 24),
                slice_denom_y=slice(16, 24),
            )

    def make_feature_r_y(self) -> None:
        pass

    def make_feature_r_front(self) -> None:
        pass

    def make_feature_r_back(self) -> None:
        pass

    def make_feature_w_x(self) -> None:
        pass

    def make_feature_w_y(self) -> None:
        pass

    def make_feature_w_x_max(self) -> None:
        pass

    def make_feature_w_y_max(self) -> None:
        pass

    def make_feature_d_e(self) -> None:
        pass

    def make_feature_r_e(self) -> None:
        pass


    def plot(self) -> None:
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        bins = np.linspace(0.0, 1.0, 101)
        ax.hist(self.ratio_x0208_y0808[PIZERO], bins=bins, histtype="stepfilled", alpha=0.75, edgecolor="black", color="red", label="Pi0")
        ax.hist(self.ratio_x0208_y0808[PHOTON], bins=bins, histtype="stepfilled", alpha=0.75, edgecolor="black", color="blue", label="Photon")
        ax.legend()
        plt.savefig(self.output_file)




class ComparisonPlots:
    def __init__(self, photon_file: str, pizero_file, output_file: str) -> None:
        self.photon_file = photon_file
        self.pizero_file = pizero_file
        self.output_file = output_file
        self.photon_df = self.get_df(photon_file)
        self.pizero_df = self.get_df(pizero_file)
        for df in [
            self.photon_df,
            self.pizero_df,
        ]:
            df["hit_r"] = np.sqrt(df["hit_x"] ** 2 + df["hit_y"] ** 2)
            df["hit_ex"] = df["hit_e"] * df["hit_x"]
            df["hit_ey"] = df["hit_e"] * df["hit_y"]
            df["hit_er"] = df["hit_e"] * df["hit_r"]

    def get_df(self, filename: str) -> pd.DataFrame:
        print(f"Reading {filename} ... ")
        def expand(input: str) -> List[str]:
            return [f for path in input.split(",") for f in glob.glob(path)]
        fnames = expand(filename)
        if len(fnames) == 1:
            return pd.read_parquet(fnames[0])
        return pd.concat([pd.read_parquet(f) for f in fnames])

    def plot(self) -> None:
        with PdfPages(self.output_file) as pdf:
            self.plot_radial_std(pdf)

    def plot_radial_std(self, pdf: PdfPages) -> None:
        for coord in ["x", "y", "r", "ex", "ey", "er"]:
            fig, ax = plt.subplots()
            bins = np.linspace(-20, 60, 100)
            ax.hist(self.photon_df.groupby("event")[f"hit_{coord}"].std(), histtype="step", label="Photon", bins=bins, color="blue")
            ax.hist(self.pizero_df.groupby("event")[f"hit_{coord}"].std(), histtype="step", label="Pizero", bins=bins, color="green")
            ax.set_xlabel(f"{coord} [mm]")
            ax.set_ylabel("Counts")
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)

    def plot_event(self, event: int, group: pd.DataFrame, pdf: PdfPages) -> None:
        nrows, ncols = NLAYERS // NCOLS_PER_ROW, NCOLS_PER_ROW
        fig, ax = plt.subplots(
            figsize=(4.5 * ncols, 4 * nrows), nrows=nrows, ncols=ncols
        )
        for i, (layer, layer_group) in enumerate(group.groupby("hit_layer")):
            energy = np.log10(layer_group["hit_e"] + 1e-6)
            i_ax = i // ncols, i % ncols
            scat = ax[i_ax].scatter(
                x=layer_group["hit_x"],
                y=layer_group["hit_y"],
                c=energy,
                vmin=plotting.vmin,
                vmax=plotting.vmax,
                cmap=plotting.cmap,
            )
            self.stylize(ax[i_ax], event, layer)
            fig.colorbar(scat, ax=ax[i_ax])
            fig.subplots_adjust(bottom=0.04, left=0.04, right=0.98, top=0.98)
            fig.subplots_adjust(wspace=0.25, hspace=0.25)

        pdf.savefig(fig)
        plt.close(fig)

    def stylize(self, ax: mpl.axes.Axes, event: int, layer: int) -> None:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_xlim(800, 950)
        ax.set_ylim(-75, 75)
        ax.grid(linewidth=0.1)
        ax.set_axisbelow(True)
        ax.tick_params(right=True, top=True)
        ax.text(
            0.100,
            1.02,
            f"{particle.name[self.pdgid]}, Event {event}, Layer {layer}",
            transform=ax.transAxes,
        )
        ax.text(0.956, 1.02, f"E [log(GeV)]", transform=ax.transAxes)


@dataclass(frozen=True)
class particle:
    name = {
        22: "Photon",
        111: "pizero",
    }


@dataclass(frozen=True)
class plotting:
    vmin, vmax = -3, -0.01
    cmap = "gist_heat_r"


if __name__ == "__main__":
    main()
