import math
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
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

THETA = math.radians(20.0)

ECal_inner_radius = 310.0 + 1.0 # endcap only. why is 1.1 necessary?
ECal_outer_radius = 2124.5
HCal_outer_radius = 4112.5
ECal_cell_size = 5.1
HCal_cell_size = 30.0

def layers(system: int) -> range:
    return range(35, 40)
    return range(50) if system == ECAL_ENDCAP else range(75)

def subtraction_x(system: int) -> float:
    return 0.0 # ECal_inner_radius if system == ECAL_ENDCAP else 0.0

def subtraction_y(system: int) -> float:
    return 0.0

def cell_size(system: int) -> float:
    return ECal_cell_size if system == ECAL_ENDCAP else HCal_cell_size

def main():
    df = pd.read_parquet(FNAME)

    OFFSET = 10

    df["hit_xp"] = np.rint(df.hit_x / ECal_cell_size)
    df["hit_yp"] = np.rint(df.hit_y / ECal_cell_size)
    df["hit_xpp"] = np.rint((df.hit_x - (df.hit_z * np.tan(THETA))) / np.vectorize(cell_size)(df.hit_system))
    df["hit_ypp"] = np.rint((df.hit_y)                              / np.vectorize(cell_size)(df.hit_system))
    df["hit_xppp"] = np.rint((df.hit_x - (df.hit_z * np.tan(THETA))) / np.vectorize(cell_size)(df.hit_system) + OFFSET)
    df["hit_yppp"] = np.rint((df.hit_y)                              / np.vectorize(cell_size)(df.hit_system) + OFFSET)
    print(df)

    axes = {
        ECAL_ENDCAP: [
            [872, 974, -51, 51],
            [171, 191, -10, 10],
            [-OFFSET, OFFSET, -OFFSET, OFFSET],
            [0, 2*OFFSET, 0, 2*OFFSET],
            [0, 2*OFFSET, 0, 2*OFFSET],
        ],
        HCAL_ENDCAP: [
            [872, 974, -51, 51],
            [171, 191, -10, 10],
            [-OFFSET, OFFSET, -OFFSET, OFFSET],
            [0, 2*OFFSET, 0, 2*OFFSET],
        ],
    }
    size = {
        ECAL_ENDCAP: 10,
        HCAL_ENDCAP: 50,
    }
    unit = [
        "mm",
        "cell",
        "cell, offset (centered)",
        "cell, offset",
        "pixel",
    ]
    xys = [
        ("hit_x", "hit_y"),
        ("hit_xp", "hit_yp"),
        ("hit_xpp", "hit_ypp"),
        ("hit_xppp", "hit_yppp"),
        (None, None),
    ]

    with PdfPages(PDF) as pdf:

        event = 0

        # scatter plots
        # for system in [ECAL_ENDCAP, HCAL_ENDCAP]:
        for system in [ECAL_ENDCAP]:
            print(f"Scatter plots for {system} ... ")
            for layer in layers(system):

                vmin, vmax = 0, 3
                cmap = "gist_heat_r" # "jet"
                GeV_to_MeV = 1000.0

                this = df[(df.event == event) & (df.hit_system == system) & (df.hit_layer == layer)]

                # coo = coo_matrix((DATA.val, (DATA.row, DATA.col)), shape=(ROWS, COLS))
                # arr = coo.toarray()
                coo_df = this[
                    (this.hit_xppp >= 0) &
                    (this.hit_yppp >= 0) &
                    (this.hit_xppp < 2*OFFSET) &
                    (this.hit_yppp < 2*OFFSET)
                ]
                shape = (2*OFFSET, 2*OFFSET)
                coo = coo_matrix((np.log10(GeV_to_MeV * coo_df.hit_e), (coo_df.hit_yppp, coo_df.hit_xppp)), shape=shape)
                print(layer, len(this))
                # if layer == 40:
                #     print(coo.toarray())
                #     print(coo.toarray().shape)

                scat = [None]*len(xys)
                cbar = [None]*len(xys)
                fig, ax = plt.subplots(nrows=1, ncols=len(xys), figsize=(24, 4))
                for i_ax, (x, y) in enumerate(xys):
                    if i_ax == len(xys) - 1 and x==None and y==None:
                        arr = coo.toarray()
                        # arr[arr == 0] = -6
                        im = ax[i_ax].imshow(
                            arr,
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax,
                        )
                        cbar[i_ax] = fig.colorbar(im, ax=ax[i_ax])
                    else:
                        scat[i_ax] = ax[i_ax].scatter(
                            this[x],
                            this[y],
                            s=size[system],
                            c=np.log10(GeV_to_MeV * this.hit_e),
                            vmin=vmin,
                            vmax=vmax,
                            cmap=cmap,
                        )
                        cbar[i_ax] = fig.colorbar(scat[i_ax], ax=ax[i_ax])
                    ax[i_ax].tick_params(right=True, top=True)
                    ax[i_ax].axis(axes[system][i_ax])
                    ax[i_ax].grid(True)
                    ax[i_ax].set_xlabel(f"x [{unit[i_ax]}]")
                    ax[i_ax].set_ylabel(f"y [{unit[i_ax]}]")
                    ax[i_ax].set_axisbelow(True)
                    cbar[i_ax].set_label("log$_{10}$(Energy [MeV])")
                    fig.subplots_adjust(bottom=0.14, left=0.05, right=0.96, top=0.95)
                    fig.subplots_adjust(wspace=0.25)

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
                axes = [
                -ECal_outer_radius,
                ECal_outer_radius,
                -ECal_outer_radius,
                ECal_outer_radius,
                ] if system == ECAL_ENDCAP else [
                    -HCal_outer_radius,
                    HCal_outer_radius,
                    -HCal_outer_radius,
                    HCal_outer_radius,
                ]
                ax.axis(axes)
                ax.grid(True)
                pdf.savefig()
                plt.close()


if __name__ == "__main__":
    main()
