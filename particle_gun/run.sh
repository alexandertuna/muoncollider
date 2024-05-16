LCIO="pgun_mu.slcio"

python ../mucoll-benchmarks/generation/pgun/pgun_lcio.py -e 100 -p 1 --pdg -13 13 --p 10 --theta 10 170 -- ${LCIO}

ddsim --steeringFile ../mucoll-benchmarks/simulation/ilcsoft/steer_baseline.py --inputFile ${LCIO}
