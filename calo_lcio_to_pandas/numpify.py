import math
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import List

# FNAME = "writeCaloHits.parquet"
# FNAME = "straightening_cells.rotation_p00.parquet"
# FNAME = "pgun_neutron.reco.1000.parquet"
FNAME = "pgun_neutron.reco.withZeroRotation.parquet"
ECAL_ENDCAP = 29
HCAL_ENDCAP = 11
EVENT = 0
PDF = FNAME + ".pdf"

ECal_inner_radius = 310.0 + 1.0 # endcap only. why is 1.1 necessary?
ECal_outer_radius = 2124.5
HCal_outer_radius = 4112.5
ECal_cell_size = 5.1
HCal_cell_size = 30.0




def layers(system: int) -> range:
    return range(50) if system == ECAL_ENDCAP else range(75)

def subtraction_x(system: int) -> float:
    return 0.0 # ECal_inner_radius if system == ECAL_ENDCAP else 0.0

def subtraction_y(system: int) -> float:
    return 0.0

def cell_size(system: int) -> float:
    return ECal_cell_size if system == ECAL_ENDCAP else HCal_cell_size

def main():
    df = pd.read_parquet(FNAME)

    # rotate?
    # theta = 0.0 * math.pi / 180.0
    # df["hit_xp"] = math.cos(theta) * df.hit_x - math.sin(theta) * df.hit_y
    # df["hit_yp"] = math.sin(theta) * df.hit_x + math.cos(theta) * df.hit_y

    theta = 20.0 * math.pi / 180.0
    # tan(theta) = x / z
    # x = z * tan(theta)

    # z_ecal = set(df.hit_z[df.hit_system == ECAL_ENDCAP])
    # print(len(z_ecal))
    # for z in sorted(z_ecal):
    #     print(f"{z:.3f} -> {int(z * math.tan(theta)):.2f}")
    # print(sorted(z_ecal))


    # df["hit_xp"] = (df.hit_x - np.vectorize(subtraction_x)(df.hit_system)) / np.vectorize(cell_size)(df.hit_system)
    # df["hit_yp"] = (df.hit_y - np.vectorize(subtraction_y)(df.hit_system)) / np.vectorize(cell_size)(df.hit_system)

    df["hit_xp"] = np.rint(df.hit_x / ECal_cell_size)
    df["hit_yp"] = np.rint(df.hit_y / ECal_cell_size)
    # df["hit_xp"] = (df.hit_x / np.vectorize(cell_size)(df.hit_system)).astype(int)
    # df["hit_yp"] = (df.hit_y / np.vectorize(cell_size)(df.hit_system)).astype(int)

    df["hit_xpp"] = np.rint((df.hit_x - (df.hit_z * np.tan(theta))) / np.vectorize(cell_size)(df.hit_system))
    df["hit_ypp"] = np.rint((df.hit_y)                              / np.vectorize(cell_size)(df.hit_system))
    # print(df)

    with PdfPages(PDF) as pdf:

        # scatter plots
        # for system in [ECAL_ENDCAP, HCAL_ENDCAP]:
        for system in [ECAL_ENDCAP]:
            print(f"Scatter plots for {system} ... ")
            for layer in layers(system):
                this = df[(df.event == 0) & (df.hit_system == system) & (df.hit_layer == layer)]
                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
                size = 10 if system == ECAL_ENDCAP else 50

                axes = [
                    [872, 974, -51, 51],
                    [171, 191, -10, 10],
                    [-10, 10, -10, 10],
                ]

                in_frame = ((this.hit_x > axes[0][0]) & (this.hit_x < axes[0][1]) & (this.hit_y > axes[0][2]) & (this.hit_y < axes[0][3]))
                mean_0_x, mean_0_y = this.hit_x[in_frame].mean(), this.hit_y[in_frame].mean()
                mean_1_x, mean_1_y = this.hit_xp[in_frame].mean(), this.hit_yp[in_frame].mean()
                mean_2_x, mean_2_y = this.hit_xpp[in_frame].mean(), this.hit_ypp[in_frame].mean()

                ax[0].scatter(this.hit_x, this.hit_y, s=size)
                ax[1].scatter(this.hit_xp, this.hit_yp, s=size)
                ax[2].scatter(this.hit_xpp, this.hit_ypp, s=size)

                # ax[0].scatter([mean_0_x], [mean_0_y], c="red")
                # ax[1].scatter([mean_1_x], [mean_1_y], c="red")
                # ax[2].scatter([mean_2_x], [mean_2_y], c="red")

                for i_ax in range(len(ax)):
                    ax[i_ax].tick_params(right=True, top=True)
                    # axes = [800, 1100, -150, 150] if system == ECAL_ENDCAP else [700, 2000, -400, 400]
                    # axes = [110, 130, -10, 10] if system == ECAL_ENDCAP else [700, 2000, -400, 400]

                    # if i_ax == 0:
                    #     axes = [872, 974, -51, 51] if system == ECAL_ENDCAP else [700, 2000, -400, 400]
                    # elif i_ax == 1:
                    #     axes = [171, 191, -10, 10] if system == ECAL_ENDCAP else [700, 2000, -400, 400]
                    # else:
                    #     axes = [-10, 10, -10, 10] if system == ECAL_ENDCAP else [700, 2000, -400, 400]

                    ax[i_ax].axis(axes[i_ax])
                    # ax[i_ax].set_xticks(range(axes[0], axes[1], 2))
                    # ax[i_ax].set_yticks(range(axes[2], axes[3], 2))
                    ax[i_ax].grid(True)

                pdf.savefig()
                plt.close()

        # histograms
        for system in [ECAL_ENDCAP, HCAL_ENDCAP]:
            continue
            print(f"Histograms for {system} ... ")
            for layer in layers(system):
                this = df[(df.hit_system == system) & (df.hit_layer == layer)]
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.hist2d(this.hit_xp, this.hit_yp, weights=this.hit_e, cmin=1e-6, bins=(50, 50))
                ax.tick_params(right=True, top=True)
                axes = [-ECal_outer_radius, ECal_outer_radius, -ECal_outer_radius, ECal_outer_radius] if system == ECAL_ENDCAP else [-HCal_outer_radius, HCal_outer_radius, -HCal_outer_radius, HCal_outer_radius]
                ax.axis(axes)
                ax.grid(True)
                pdf.savefig()
                plt.close()

def rotate(theta, x, y):
    xp = math.cos(theta) * x + math.sin(theta) * y
    yp = -math.sin(theta) * x + math.cos(theta) * y
    return xp, yp

if __name__ == "__main__":
    main()
