EVENTS="10"
PARTICLE="2112"
ENERGY="50 500"
THETA=20
PHI=0
LCIO="pgun_neutron.slcio"
SIM="pgun_neutron.sim.slcio"

time python ../mucoll-benchmarks/generation/pgun/pgun_lcio.py -e ${EVENTS} -p 1 --pdg ${PARTICLE} --p ${ENERGY} --theta ${THETA} --phi ${PHI} -- ${LCIO}

time ddsim --steeringFile ../mucoll-benchmarks/simulation/ilcsoft/steer_baseline.py --inputFile ${LCIO} --outputFile ${SIM}

time Marlin --global.LCIOInputFiles=${SIM} --DD4hep.DD4hepXMLFile=${MUCOLL_GEO} ../mucoll-benchmarks/digitisation/marlin/digi_steer.xml
