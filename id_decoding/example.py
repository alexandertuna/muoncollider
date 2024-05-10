import pyLCIO
from pyLCIO import EVENT, UTIL

fname = "/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10000.slcio"
reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
reader.open(fname)

for event in reader:
    collection = event.getCollection("ECalBarrelCollection")
    encoding = collection.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
    decoder = UTIL.BitField64(encoding)
    for i_hit, hit in enumerate(collection):
        decoder.setValue((hit.getCellID0() & 0xFFFFFFFF) | (hit.getCellID1() << 32))
        layer = decoder["layer"].value()
        print(f"hit {i_hit} is on layer {layer}")
        if i_hit > 10:
            break
    break
