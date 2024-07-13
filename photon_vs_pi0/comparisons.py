import argparse
import glob
import numpy as np
# import pandas as pd
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

PLOTKW = {
    "histtype": "stepfilled",
    "edgecolor": "black",
    "alpha": 0.75,
}
PHOTONKW = {
    "color": "blue",
    "label": "Photon"
}
PIZEROKW = {
    "color": "red",
    "label": "Pi0",
}

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
        self.derived = {
            PHOTON: {},
            PIZERO: {},
        }
        self.output_file = output_file
        print("photon:", self.features[PHOTON].shape)
        print("pizero:", self.features[PIZERO].shape)
        _, z, x, y = self.features[PHOTON].shape
        self.x_all = slice(0, x)
        self.y_all = slice(0, y)
        self.z_all = slice(0, z)
        self.make_new_features()
        # self.write_to_file()

    def write_to_file(self):
        print("ffff")
        np.savez("photon." + self.output_file, **self.derived[PHOTON])
        np.savez("pizero." + self.output_file, **self.derived[PIZERO])
#         np.savez(self.output_file,
#                  ratio_x0208_y0808=self.ratio_x0208_y0808,
#                  ratio_x0408_y0808=self.ratio_x0408_y0808,
#                  ratio_x0808_y0208=self.ratio_x0808_y0208,
#                  ratio_x0808_y0408=self.ratio_x0808_y0408,
#                  ratio_z10=self.ratio_z10,
#                  ratio_z20=self.ratio_z20,
#                  width_x=self.width_x,
#                  width_y=self.width_y,
#                  width_about_max_x=self.width_about_max_x,
#                  width_about_max_y=self.width_about_max_y,
#                  e_ratio_x=self.e_ratio_x,
#                  delta_e_x=self.delta_e_x,
#                  e_ratio_y=self.e_ratio_y,
#                  delta_e_y=self.delta_e_y,
#                  )

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
        self.make_feature_ratio_y()
        self.make_feature_ratio_z()
        self.make_feature_width()
        self.make_feature_width_about_max()
        self.make_feature_two_maxima()

    def make_feature_ratio_x(self) -> None:
        for PARTICLE in PARTICLES:
            self.derived[PARTICLE]["ratio_x0208_y0808"] = self.calculate_ratios(
                self.features[PARTICLE],
                slice_numer_x=slice(19, 21),
                slice_denom_x=slice(16, 24),
                slice_numer_y=slice(16, 24),
                slice_denom_y=slice(16, 24),
            )
            self.derived[PARTICLE]["ratio_x0408_y0808"] = self.calculate_ratios(
                self.features[PARTICLE],
                slice_numer_x=slice(18, 22),
                slice_denom_x=slice(16, 24),
                slice_numer_y=slice(16, 24),
                slice_denom_y=slice(16, 24),
            )

    def make_feature_ratio_y(self) -> None:
        for PARTICLE in PARTICLES:
            self.derived[PARTICLE]["ratio_x0808_y0208"] = self.calculate_ratios(
                self.features[PARTICLE],
                slice_numer_x=slice(16, 24),
                slice_denom_x=slice(16, 24),
                slice_numer_y=slice(19, 21),
                slice_denom_y=slice(16, 24),
            )
            self.derived[PARTICLE]["ratio_x0808_y0408"] = self.calculate_ratios(
                self.features[PARTICLE],
                slice_numer_x=slice(16, 24),
                slice_denom_x=slice(16, 24),
                slice_numer_y=slice(18, 22),
                slice_denom_y=slice(16, 24),
            )

    def make_feature_ratio_z(self) -> None:
        for PARTICLE in PARTICLES:
            self.derived[PARTICLE]["ratio_z10"] = self.calculate_ratios(
                self.features[PARTICLE],
                slice_numer_z=slice(0, 10),
            )
            self.derived[PARTICLE]["ratio_z20"] = self.calculate_ratios(
                self.features[PARTICLE],
                slice_numer_z=slice(0, 20),
            )

    def calculate_width(self, images, axis, slice_x=slice(None), slice_y=slice(None), slice_z=slice(None)) -> None:
        if axis not in (2, 3):
            raise ValueError("Axis must be 2 (x-dimension) or 3 (y-dimension)")
        central_region = images[:, slice_z, slice_x, slice_y]
        mean_axes = tuple(i for i in range(central_region.ndim) if i != axis and i != 0)
        img1d = np.sum(central_region, axis=mean_axes)
        indices = np.arange(img1d.shape[-1])
        weighted_mean = np.sum(img1d * indices, axis=1) / np.sum(img1d, axis=1)
        weighted_square = np.sum(img1d * indices * indices, axis=1) / np.sum(img1d, axis=1)
        width = np.sqrt(weighted_square - weighted_mean**2)
        return width

    def make_feature_width(self) -> None:
        slice_x, slice_y = slice(15, 25), slice(15, 25)
        for PARTICLE in PARTICLES:
            self.derived[PARTICLE]["width_x"] = self.calculate_width(self.features[PARTICLE], axis=2, slice_x=slice_x, slice_y=slice_y)
            self.derived[PARTICLE]["width_y"] = self.calculate_width(self.features[PARTICLE], axis=3, slice_x=slice_x, slice_y=slice_y)


    def calculate_width_about_max(self,
                                      images,
                                      axis,
                                      slice_x=slice(None),
                                      slice_y=slice(None),
                                      slice_z=slice(None),
                                      ):
        if axis not in (2, 3):
            raise ValueError("Axis must be 2 (x-dimension) or 3 (y-dimension)")
        central_region = images[:, slice_z, slice_x, slice_y]
        mean_axes = tuple(i for i in range(central_region.ndim) if i != axis and i != 0)
        img1d = np.sum(central_region, axis=mean_axes)
        argmaxs = np.argmax(img1d, axis=1)
        indices = np.arange(img1d.shape[-1])
        weighted_pos = np.sqrt( (img1d * ((argmaxs[:, np.newaxis] - indices) ** 2)).sum(axis=1) / img1d.sum(axis=1) )
        return weighted_pos

    def make_feature_width_about_max(self) -> None:
        slice_x, slice_y = slice(15, 25), slice(15, 25)
        for PARTICLE in PARTICLES:
            self.derived[PARTICLE]["width_about_max_x"] = self.calculate_width_about_max(self.features[PARTICLE], axis=2, slice_x=slice_x, slice_y=slice_y)
            self.derived[PARTICLE]["width_about_max_y"] = self.calculate_width_about_max(self.features[PARTICLE], axis=3, slice_x=slice_x, slice_y=slice_y)


    def calculate_max_ratio(self, images, axis, slice_x=slice(None), slice_y=slice(None), slice_z=slice(None)):
        if axis not in (2, 3):
            raise ValueError("Axis must be 2 (x-dimension) or 3 (y-dimension)")

        central_region = images[:, slice_z, slice_x, slice_y]
        mean_axes = tuple(i for i in range(central_region.ndim) if i != axis and i != 0)
        arr = np.sum(central_region, axis=mean_axes)

        # Initialize arrays to store the results for each row
        max_values = np.max(arr, axis=1)
        max_indices = np.argmax(arr, axis=1)

        # Create a masked array to find the second max
        masked_arr = np.ma.masked_array(arr, mask=False)
        for i, idx in enumerate(max_indices):
            masked_arr.mask[i, idx] = True

        second_max_values = masked_arr.max(axis=1)
        second_max_indices = np.argmax(masked_arr, axis=1)

        # Ensure max_indices are less than second_max_indices for each row
        start_indices = np.minimum(max_indices, second_max_indices)
        end_indices = np.maximum(max_indices, second_max_indices)

        # Find the minimum values between max and second max for each row
        # Difficult to vectorize this! Sorry.
        min_between_max = []
        for i in range(arr.shape[0]):
            if end_indices[i] - start_indices[i] > 1:
                subarray = arr[i, start_indices[i] + 1:end_indices[i]]
                min_between_max.append(np.min(subarray))
            else:
                min_between_max.append(second_max_values[i])

        return (max_values - second_max_values) / (max_values + second_max_values), (second_max_values - min_between_max)

    def make_feature_two_maxima(self) -> None:
        for PARTICLE in PARTICLES:
            self.derived[PARTICLE]["e_ratio_x"], self.derived[PARTICLE]["delta_e_x"] = self.calculate_max_ratio(self.features[PARTICLE], axis=2)
            self.derived[PARTICLE]["e_ratio_y"], self.derived[PARTICLE]["delta_e_y"] = self.calculate_max_ratio(self.features[PARTICLE], axis=3)

    def plot(self) -> None:
        pass
        with PdfPages(self.output_file) as pdf:
            self.plot_ratio_x0208_y0808(pdf)
#            self.plot_ratio_x0408_y0808(pdf)
#            self.plot_ratio_x0808_y0208(pdf)
#            self.plot_ratio_x0808_y0408(pdf)
#            self.plot_ratio_z10(pdf)
#            self.plot_ratio_z20(pdf)
#            self.plot_width(pdf)
#            self.plot_width_about_max(pdf)
#            self.plot_e_ratio(pdf)
#            self.plot_delta_e(pdf)

    def plot_ratio_x0208_y0808(self, pdf: PdfPages) -> None:
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        bins = np.linspace(0.0, 1.01, 102)
        ax.hist(self.derived[PIZERO]["ratio_x0208_y0808"], bins=bins, histtype="stepfilled", alpha=0.75, edgecolor="black", color="red", label="Pi0")
        ax.hist(self.derived[PHOTON]["ratio_x0208_y0808"], bins=bins, histtype="stepfilled", alpha=0.75, edgecolor="black", color="blue", label="Photon")
        ax.legend()
        ax.set_xlabel("Sum $x_2$,$y_8$ / sum $x_8$,$y_8$")
        ax.set_ylabel("Number of events")
        pdf.savefig()
        plt.close()

    def plot_ratio_x0408_y0808(self, pdf: PdfPages) -> None:
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        bins = np.linspace(0.0, 1.01, 102)
        ax.hist(self.ratio_x0408_y0808[PIZERO], bins=bins, histtype="stepfilled", alpha=0.75, edgecolor="black", color="red", label="Pi0")
        ax.hist(self.ratio_x0408_y0808[PHOTON], bins=bins, histtype="stepfilled", alpha=0.75, edgecolor="black", color="blue", label="Photon")
        ax.legend()
        ax.set_xlabel("Sum $x_4$,$y_8$ / sum $x_8$,$y_8$")
        ax.set_ylabel("Number of events")
        pdf.savefig()
        plt.close()

    def plot_ratio_x0808_y0208(self, pdf: PdfPages) -> None:
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        bins = np.linspace(0.0, 1.01, 102)
        ax.hist(self.ratio_x0808_y0208[PIZERO], bins=bins, histtype="stepfilled", alpha=0.75, edgecolor="black", color="red", label="Pi0")
        ax.hist(self.ratio_x0808_y0208[PHOTON], bins=bins, histtype="stepfilled", alpha=0.75, edgecolor="black", color="blue", label="Photon")
        ax.legend()
        ax.set_xlabel("Sum $x_8$,$y_2$ / sum $x_8$,$y_8$")
        ax.set_ylabel("Number of events")
        pdf.savefig()
        plt.close()

    def plot_ratio_x0808_y0408(self, pdf: PdfPages) -> None:
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        bins = np.linspace(0.0, 1.01, 102)
        ax.hist(self.ratio_x0808_y0408[PIZERO], bins=bins, histtype="stepfilled", alpha=0.75, edgecolor="black", color="red", label="Pi0")
        ax.hist(self.ratio_x0808_y0408[PHOTON], bins=bins, histtype="stepfilled", alpha=0.75, edgecolor="black", color="blue", label="Photon")
        ax.legend()
        ax.set_xlabel("Sum $x_8$,$y_4$ / sum $x_8$,$y_8$")
        ax.set_ylabel("Number of events")
        pdf.savefig()
        plt.close()

    def plot_ratio_z10(self, pdf: PdfPages) -> None:
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        bins = np.linspace(0.0, 1.01, 102)
        ax.hist(self.ratio_z10[PIZERO], bins=bins, histtype="stepfilled", alpha=0.75, edgecolor="black", color="red", label="Pi0")
        ax.hist(self.ratio_z10[PHOTON], bins=bins, histtype="stepfilled", alpha=0.75, edgecolor="black", color="blue", label="Photon")
        ax.legend()
        ax.set_xlabel("Sum $z_{10}$ / sum $z_{all}$")
        ax.set_ylabel("Number of events")
        pdf.savefig()
        plt.close()

    def plot_ratio_z20(self, pdf: PdfPages) -> None:
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        bins = np.linspace(0.0, 1.01, 102)
        ax.hist(self.ratio_z20[PIZERO], bins=bins, histtype="stepfilled", alpha=0.75, edgecolor="black", color="red", label="Pi0")
        ax.hist(self.ratio_z20[PHOTON], bins=bins, histtype="stepfilled", alpha=0.75, edgecolor="black", color="blue", label="Photon")
        ax.legend()
        ax.set_xlabel("Sum $z_{20}$ / sum $z_{all}$")
        ax.set_ylabel("Number of events")
        pdf.savefig()
        plt.close()

    def plot_width(self, pdf: PdfPages) -> None:
        labels = ["$x$-width", "$y$-width"]
        widths = [self.width_x, self.width_y]
        for width, label in zip(widths, labels):
            fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
            bins = np.linspace(0.5, 3.5, 100)
            ax.hist(width[PIZERO], bins=bins, **PIZEROKW, **PLOTKW)
            ax.hist(width[PHOTON], bins=bins, **PHOTONKW, **PLOTKW)
            ax.legend()
            ax.set_xlabel(label)
            ax.set_ylabel("Number of events")
            pdf.savefig()
            plt.close()

    def plot_width_about_max(self, pdf: PdfPages) -> None:
        labels = ["$x$-width about maximum", "$y$-width about maximum"]
        widths = [self.width_about_max_x, self.width_about_max_y]
        for width, label in zip(widths, labels):
            fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
            bins = np.linspace(0.5, 3.5, 100)
            ax.hist(width[PIZERO], bins=bins, **PIZEROKW, **PLOTKW)
            ax.hist(width[PHOTON], bins=bins, **PHOTONKW, **PLOTKW)
            ax.legend()
            ax.set_xlabel(label)
            ax.set_ylabel("Number of events")
            pdf.savefig()
            plt.close()

    def plot_e_ratio(self, pdf: PdfPages) -> None:
        labels = ["$E$-ratio ($x$)", "$E$-ratio ($y$)"]
        ratios = [self.e_ratio_x, self.e_ratio_y]
        for ratio, label in zip(ratios, labels):
            fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
            bins = np.linspace(0.0, 1.01, 102)
            ax.hist(ratio[PIZERO], bins=bins, **PIZEROKW, **PLOTKW)
            ax.hist(ratio[PHOTON], bins=bins, **PHOTONKW, **PLOTKW)
            ax.legend()
            ax.set_xlabel(label)
            ax.set_ylabel("Number of events")
            pdf.savefig()
            plt.close()

    def plot_delta_e(self, pdf: PdfPages) -> None:
        labels = ["delta-$E$ ($x$)", "delta-$E$ ($y$)"]
        deltas = [self.delta_e_x, self.delta_e_y]
        for delta, label in zip(deltas, labels):
            fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
            bins = np.linspace(0.0, 30.0, 100)
            ax.hist(delta[PIZERO], bins=bins, **PIZEROKW, **PLOTKW)
            ax.hist(delta[PHOTON], bins=bins, **PHOTONKW, **PLOTKW)
            ax.legend()
            ax.set_xlabel(label)
            ax.set_ylabel("Number of events")
            ax.semilogy()
            pdf.savefig()
            plt.close()


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

    def get_df(self, filename: str):
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

    def plot_event(self, event: int, group, pdf: PdfPages) -> None:
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
