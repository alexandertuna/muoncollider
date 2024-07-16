for BATCH in {0..9}; do
    time for ITER in {0..4}; do
        time source all_calo_cells/run.sh &
        sleep 10s
    done
    sleep 30m
done
