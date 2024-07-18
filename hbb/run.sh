GEN="mumu_H_bb_3TeV.hepmc"
SIM="mumu_H_bb_3TeV.slcio"

# time whizard /code/mucoll-benchmarks/generation/signal/whizard/mumu_H_bb_3TeV.sin

export MUCOLL_GEO="/code/detector-simulation/geometries/MuColl_10TeV_v0A/MuColl_10TeV_v0A.xml"

# time ddsim --steeringFile /code/mucoll-benchmarks/simulation/ilcsoft/steer_baseline.py --inputFile ${GEN} --outputFile ${SIM} -N 100

sed -i "s|/data/sim/TYPEEVENT/TYPEEVENT_sim_INFILENAME.slcio|mumu_H_bb_3TeV.slcio|g" steer_reco_CONDOR.py
sed -i "s|/data/reco/TYPEEVENT/TYPEEVENT_reco_INFILENAME.slcio|mumu_H_bb_3TeV.reco.slcio|g" steer_reco_CONDOR.py
### /code/ has 2 instances for geo and pandora
time k4run steer_reco_CONDOR.py
