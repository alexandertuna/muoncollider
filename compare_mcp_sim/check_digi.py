import pyLCIO # type: ignore
from collections.abc import Sized

def main() -> None:
    fname = "/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10000.slcio"
    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open(fname)
    for i_event, event in enumerate(reader):
        ecals_b = getCollection(event, "EcalBarrelCollectionDigi")
        ecals_e = getCollection(event, "EcalEndcapCollectionDigi")
        hcals_b = getCollection(event, "HcalBarrelCollectionDigi")
        hcals_e = getCollection(event, "HcalEndcapCollectionDigi")
        print(f"Event {i_event}")
        print(f"len(ecals_b): {len(ecals_b)}")
        print(f"len(ecals_e): {len(ecals_e)}")
        print(f"len(hcals_b): {len(hcals_b)}")
        print(f"len(hcals_e): {len(hcals_e)}")
        break

def getCollection(event: pyLCIO.EVENT.LCEvent, name: str) -> Sized:
    if name in event.getCollectionNames():
        return event.getCollection(name)
    return []

if __name__ == "__main__":
    main()
