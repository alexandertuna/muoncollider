import math
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

XMIN = 1
XMAX = 200
NBINS = XMAX - XMIN
XTEXT = 0.1
YTEXT = 1.05

def main():
    data = np.random.exponential(10, size=1000000)
    with PdfPages("logbins.pdf") as pdf:
        for astype in [float, int]:
            data_astype = data.astype(astype)
            basic_bins(data_astype, pdf)
            linear_bins(data_astype, pdf)
            log_bins(data_astype, pdf)
            linear_log_bins(data_astype, pdf)
            linear_log_bins_density(data_astype, pdf)
            log_bins_floatized(data_astype, pdf)


def basic_bins(data, pdf):
    fig, ax = plt.subplots(figsize=(4, 4))
    bins = np.linspace(XMIN, XMAX, NBINS)
    ax.hist(data, bins=bins)
    ax.text(XTEXT, YTEXT, "Linear bins", transform=ax.transAxes)
    pdf.savefig()
    ax.semilogy()
    pdf.savefig()
    plt.close()


def linear_bins(data, pdf):
    fig, ax = plt.subplots(figsize=(4, 4))
    bins = np.linspace(XMIN, XMAX, NBINS)
    ax.hist(data, bins=bins)
    ax.semilogx()
    ax.semilogy()
    ax.text(XTEXT, YTEXT, "Linear bins", transform=ax.transAxes)
    pdf.savefig()
    plt.close()


def log_bins(data, pdf):
    fig, ax = plt.subplots(figsize=(4, 4))
    bins = np.logspace(math.log(XMIN, 10), math.log(XMAX, 10), NBINS)
    ax.hist(data, bins=bins)
    ax.text(XTEXT, YTEXT, "Log bins", transform=ax.transAxes)
    ax.semilogx()
    ax.semilogy()
    pdf.savefig()
    plt.close()


def linear_log_bins(data, pdf):
    fig, ax = plt.subplots(figsize=(4, 4))
    bins = np.logspace(math.log(XMIN, 10), math.log(XMAX, 10), NBINS)
    bins = np.unique(bins.astype(int))
    ax.hist(data, bins=bins)
    ax.text(XTEXT, YTEXT, "Linear and log bins", transform=ax.transAxes)
    ax.semilogx()
    ax.semilogy()
    pdf.savefig()
    plt.close()


def linear_log_bins_density(data, pdf):
    fig, ax = plt.subplots(figsize=(4, 4))
    bins = np.logspace(math.log(XMIN, 10), math.log(XMAX, 10), NBINS)
    bins = np.unique(bins.astype(int))
    ax.hist(data, bins=bins, density=True)
    ax.text(XTEXT, YTEXT, "Linear and log bins (density)", transform=ax.transAxes)
    ax.semilogx()
    ax.semilogy()
    pdf.savefig()
    plt.close()


def log_bins_floatized(data, pdf):
    data_floatized = data + np.random.rand(len(data)) - 0.5
    fig, ax = plt.subplots(figsize=(4, 4))
    bins = np.logspace(math.log(XMIN, 10), math.log(XMAX, 10), NBINS)
    ax.hist(data_floatized, bins=bins)
    ax.text(XTEXT, YTEXT, "Log bins (floatized)", transform=ax.transAxes)
    ax.semilogx()
    ax.semilogy()
    pdf.savefig()
    plt.close()


if __name__ == "__main__":
    main()
