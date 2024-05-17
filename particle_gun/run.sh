EVENTS="10"
PARTICLE="2112"
ENERGY="50 500"
THETA=20
PHI=0
DATA="/work/tuna/data"
LCIO="${DATA}/pgun_neutron.slcio"
SIM="${DATA}/pgun_neutron.sim.slcio"
CODE="/code"

export MUCOLL_GEO="${CODE}/detector-simulation/geometries/MuColl_10TeV_v0A/MuColl_10TeV_v0A.xml"

time python ${CODE}/mucoll-benchmarks/generation/pgun/pgun_lcio.py -o -e ${EVENTS} -p 1 --pdg ${PARTICLE} --p ${ENERGY} --theta ${THETA} --phi ${PHI} -- ${LCIO} || return

time ddsim --steeringFile ${CODE}/mucoll-benchmarks/simulation/ilcsoft/steer_baseline.py --inputFile ${LCIO} --outputFile ${SIM} || return
# time ddsim --inputFile ${LCIO} --steeringFile /code/SteeringMacros/Sim/sim_steer_GEN_CONDOR.py --outputFile ${SIM}

time k4run ${CODE}/SteeringMacros/k4Reco/steer_reco_CONDOR.py || return
# gaudirun.py ../SteeringMacros/k4Reco/steer_reco_CONDOR.py
# time Marlin --global.LCIOInputFiles=${SIM} --DD4hep.DD4hepXMLFile=${MUCOLL_GEO} ../mucoll-benchmarks/digitisation/marlin/digi_steer.xml
