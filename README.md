# muoncollider

Heavily influenced by https://github.com/trholmes/mucolstudies.

Running the apptainer container:

```
apptainer run --no-home -B /tank/data/snowmass21/muonc:/data -B /home/$USER -B /work/$USER /home/$USER/k4toroid.sif

> source /opt/spack/share/spack/setup-env.sh
> spack env activate -V k4prod
> source /opt/spack/opt/spack/linux-ubuntu22.04-x86_64/gcc-11.3.0/mucoll-stack-2023-07-30-ysejlaccel4azxh3bxzsdb7asuzxbfof/setup.sh
> export MARLIN_DLL=$MARLIN_DLL:/opt/MyBIBUtils/lib/libMyBIBUtils.so
```

A few python modules are installed as user packages:

```
python -m pip install mypy
python -m pip install pyarrow
python -m pip install pandas-stubs
python -m pip install plotly
```

Note to self: need to double-check those.
