import pyLCIO

# fname = "/work/tuna/muoncollider/pgun_mu.slcio"
fname = "/work/tuna/muoncollider/output_sim.slcio"
reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
reader.open(fname)

for event in reader:
    print(event)
    print(event.getCollectionNames())
    break
