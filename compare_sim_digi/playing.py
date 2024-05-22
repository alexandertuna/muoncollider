import math
import numpy as np
from numpy.random import normal
from numpy.random import binomial
from scipy.stats import skew
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
    
    # GeV -> NPE:
    #  _PPD_pe_per_mip*energy/_calib_mip

    # Thus
    # NPE -> GeV:
    # npe * _calib_mip / _PPD_pe_per_mip
    
    _PPD_pe_per_mip = 15.0
    _calib_mip = 0.0004925
    _PPD_n_pixels = 2000.0

    # energy = 10 * _calib_mip / _PPD_pe_per_mip
    # npe = energy * _PPD_pe_per_mip / _calib_mip

    functions = "binomial", "gaussian"

    with PdfPages("binomial.pdf") as pdf:

        # ndata = 10_000_000
        ndata = 10_000_000

        for function in functions:

            for npe in [10, 20, 40, 80, 160, 320]:
            # for npe in [10, 20, 40]:

                print(f"Plotting {npe} ...")
                p = npe / _PPD_n_pixels
                if function == "binomial":
                    color = "blue"
                    data = binomial(_PPD_n_pixels, np.zeros(ndata) + p)
                else:
                    color = "green"
                    data = normal(loc=npe, scale=npe/10, size=ndata)
                n_t = len(data)
                n_l = (data < npe).sum()
                n_r = (data > npe).sum()
                f_l = n_l / n_t
                f_r = n_r / n_t

                fig, ax = plt.subplots()
                ax.hist(data, bins=np.arange(0, 2 * npe) + 0.5, color=color, density=True)
                ax.set_xlabel("x")
                ax.set_ylabel("f(x) (PDF)")
                ax.tick_params(top=True, right=True)
                ax.set_axisbelow(True)
                ax.grid()
                ax.text(0.02, 1.05, f"{function} with mean={int(npe)}", transform=ax.transAxes)
                ax.text(0.48, 1.03, f"{int(f_l*100)}% < peak", transform=ax.transAxes)
                ax.text(0.48, 1.08, f"{int(f_r*100)}% > peak", transform=ax.transAxes)
                ax.text(0.78, 1.05, f"skew = {skew(data):.2f}", transform=ax.transAxes)
                pdf.savefig()
                ax.semilogy()
                pdf.savefig()
                plt.close()


if __name__ == "__main__":
    main()
