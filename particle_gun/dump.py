import pyLCIO

# fname = "/work/tuna/muoncollider/pgun_neutron.slcio"
fname = "/work/tuna/muoncollider/output_sim.slcio"
reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
reader.open(fname)

for event in reader:
    print(event)
    names = event.getCollectionNames()
    print(names)
    for name in names:
        print(name, len(event.getCollection(name)))
    # col = event.getCollection("MCParticle")
    # for ele in col:
    #     mom = ele.getMomentum()
    #     print(ele, mom[0], mom[1], mom[2])
    
