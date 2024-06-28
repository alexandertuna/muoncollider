EVENTS="1000"
PARTICLE="pi0"
PDGID="111"
# PARTICLE="photon"
# PDGID="22"
# PARTICLE="neutron"
# PDGID="2112"
# ENERGY="50 500"
ENERGY="100"
THETA=20
PHI=0

NOW=$(date +%Y_%m_%d_%Hh%Mm%Ss)

CODE="/code"
TMP="/tmp/tuna/${NOW}"
DATA="/scratch/tuna/data"

GEN=${DATA}/${PARTICLE}/${PARTICLE}_gen_${NOW}.slcio
SIM=${DATA}/${PARTICLE}/${PARTICLE}_sim_${NOW}.slcio
STEER=${TMP}/steer_reco_CONDOR.py

export MUCOLL_GEO="${CODE}/detector-simulation/geometries/MuColl_10TeV_v0A/MuColl_10TeV_v0A.xml"

echo "mkdir"
mkdir $TMP

echo "cp and sed"
cp ${CODE}/SteeringMacros/k4Reco/steer_reco_CONDOR.py ${STEER} || return
sed -i "s/INFILENAME/${NOW}/g" ${STEER} || return
sed -i "s/TYPEEVENT/${PARTICLE}/g" ${STEER} || return

echo "pgun"
time python ${CODE}/mucoll-benchmarks/generation/pgun/pgun_lcio.py -o -e ${EVENTS} -p 1 --pdg ${PDGID} --p ${ENERGY} --theta ${THETA} --phi ${PHI} -- ${GEN} || return

time ddsim --steeringFile ${CODE}/mucoll-benchmarks/simulation/ilcsoft/steer_baseline.py --inputFile ${GEN} --outputFile ${SIM} || return
# time ddsim --inputFile ${GEN} --steeringFile /code/SteeringMacros/Sim/sim_steer_GEN_CONDOR.py --outputFile ${SIM}

time k4run ${STEER} # || return
# https://github.com/key4hep/k4FWCore/issues/125
# time k4run ${CODE}/SteeringMacros/k4Reco/steer_reco_CONDOR.py || return
# time Marlin --global.LCIOInputFiles=${SIM} --DD4hep.DD4hepXMLFile=${MUCOLL_GEO} ../mucoll-benchmarks/digitisation/marlin/digi_steer.xml

echo "rm -f ${SIM}"
rm -f ${SIM}
