EVENTS="2000"
PARTICLE="photon"
PDGID="22"
# PARTICLE="neutron"
# PDGID="2112"
ENERGY="50 500"
THETA=20
PHI=0

NOW=$(date +%Y_%m_%d_%Hh%Mm%Ss)

CODE="/code"
TMP="/tmp/tuna/${NOW}"
DATA="/work/tuna/data"

GEN=${DATA}/${PARTICLE}/${PARTICLE}_gen_${NOW}.slcio
SIM=${DATA}/${PARTICLE}/${PARTICLE}_sim_${NOW}.slcio

export MUCOLL_GEO="${CODE}/detector-simulation/geometries/MuColl_10TeV_v0A/MuColl_10TeV_v0A.xml"

echo "mkdir and cd"
mkdir $TMP
cd $TMP

echo "cp and sed"
cp ${CODE}/SteeringMacros/k4Reco/steer_reco_CONDOR.py ./ || return
sed -i "s/INFILENAME/${NOW}/g" steer_reco_CONDOR.py || return
sed -i "s/TYPEEVENT/${PARTICLE}/g" steer_reco_CONDOR.py || return

echo "pgun"
time python ${CODE}/mucoll-benchmarks/generation/pgun/pgun_lcio.py -o -e ${EVENTS} -p 1 --pdg ${PDGID} --p ${ENERGY} --theta ${THETA} --phi ${PHI} -- ${GEN} || return

time ddsim --steeringFile ${CODE}/mucoll-benchmarks/simulation/ilcsoft/steer_baseline.py --inputFile ${GEN} --outputFile ${SIM} || return
# time ddsim --inputFile ${GEN} --steeringFile /code/SteeringMacros/Sim/sim_steer_GEN_CONDOR.py --outputFile ${SIM}

time k4run ./steer_reco_CONDOR.py || return
# time k4run ${CODE}/SteeringMacros/k4Reco/steer_reco_CONDOR.py || return
# time Marlin --global.LCIOInputFiles=${SIM} --DD4hep.DD4hepXMLFile=${MUCOLL_GEO} ../mucoll-benchmarks/digitisation/marlin/digi_steer.xml

echo "rm -f ${SIM}"
