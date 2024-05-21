for FI in steer_baseline.py sim_steer_GEN_CONDOR.py; do
    grep -v "^#" ${FI} > ${FI}.0
    grep "\S" ${FI}.0 > ${FI}.1
    python -m black ${FI}.1
done
diff *.1
