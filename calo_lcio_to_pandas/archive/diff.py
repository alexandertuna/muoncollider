import numpy as np

def main():
    fn_0 = "/work/tuna/data/generating_photons_zeroth_attempt/5000_events/pgun_photon.5000.reco.slcio.parquet.features.npy"
    fn_1 = "/work/tuna/muoncollider/data.features.npy"

    arr_0 = np.load(fn_0)
    arr_1 = np.load(fn_1)

    print("0:", arr_0.shape)
    print("1:", arr_1.shape)

    n = 1000
    print("All close:", np.allclose(arr_0[:n], arr_1[:n]))
    print(arr_0[:n].sum())
    print(arr_1[:n].sum())

if __name__ == "__main__":
    main()
