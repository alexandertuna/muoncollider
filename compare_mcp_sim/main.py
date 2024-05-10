import pyLCIO
from pyLCIO import EVENT, UTIL

def main():

    fname = "/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10000.slcio"

    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open(fname)

    for i_event, event in enumerate(reader):
        mcps = getCollection(event, "MCParticle")
        hcals_b = getCollection(event, "HCalBarrelCollection")
        hcals_e = getCollection(event, "HCalEndcapCollection")
        print(f"len(mcps): {len(mcps)}")
        print(f"len(hcals_b): {len(hcals_b)}")
        print(f"len(hcals_e): {len(hcals_e)}")

        # print all MCP
        if i_event == 2:
            for i_mcp, mcp in enumerate(mcps):
                print(i_mcp, pretty(mcp.getVertex(), mcp.getEnergy(), mcp.getTime()))

        # x, y, z, E, t
        if i_event == 2:
            for i_hit, hit in enumerate(hcals_b):
                print("*", i_event, pretty(hit.getPosition(), hit.getEnergy(), 0))
                for i_mcp in range(hit.getNMCContributions()):
                    print("-", i_mcp, pretty(hit.getStepPosition(i_mcp),
                                             hit.getEnergyCont(i_mcp),
                                             hit.getTimeCont(i_mcp)))
                print("")
                if i_hit > 10:
                    break
        if i_event > 2:
            break

def pretty(pos, e, t):
    x, y, z = pos[0], pos[1], pos[2]
    return f"x={int(x):5}, y={int(y):5}, z={int(z):5}, E={e:.8f}, t={int(t)}"

def getCollection(event, name):
    if name in event.getCollectionNames():
        return event.getCollection(name)
    return []


if __name__ == "__main__":
    main()
