from glob import glob
import os
from typing import List
import time

NOW = time.strftime("%Y_%m_%d_%Hh%Mm%Ss")
DIRNAME = "/ospool/uc-shared/project/futurecolliders/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/photonGun_E_250_1000"
# SIFNAME = "osdf:///ospool/uc-shared/project/futurecolliders/tuna/k4toroid.sif" # crashes due to /.singularity/env/10-blah.sh
SIFNAME = "osdf:///ospool/uc-shared/project/futurecolliders/tuna/k4toroid2.sif"
# SIFNAME = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/infnpd/mucoll-ilc-framework:1.6-centos8"
# SIFNAME = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/infnpd/mucoll-ilc-framework:1.7-almalinux9"


def main():
    launcher = Launcher(sorted(glob(f"{DIRNAME}/*.slcio")))
    launcher.run()


class Launcher:
    def __init__(self, fnames: List[str]):
        self.fnames = fnames
        print(f"Found {len(self.fnames)} files")

    def run(self):
        for id, fname in enumerate(self.fnames):
            self.run_one(id, fname)
            if id >= 1:
                break

    def run_one(self, id: int, fname: str):
        print(f"Processing {fname}")
        fname_sub = f"condor_submit_{id:06}.sub"
        fname_exe = f"condor_exe_{id:06}.sh"
        output = f"/ospool/uc-shared/project/futurecolliders/tuna/grid/{NOW}/job_{id:06}"
        cfg = {
            "IMAGE": SIFNAME,
            "EXECUTABLE": fname_exe,
            "OUTDIR": f"osdf://{output}",
            "OUTPUT": f"condor_out_{id:06}.$(Cluster)_$(Process).txt",
            "LOG": f"condor_log_{id:06}.$(Cluster)_$(Process).txt",
            "ERROR": f"condor_err_{id:06}.$(Cluster)_$(Process).txt",
        }
        os.makedirs(output, exist_ok=True)
        sub = template_submit() % cfg
        exe = template_executable() % cfg
        with open(fname_sub, "w") as fi:
            fi.write(sub)
        with open(fname_exe, "w") as fi:
            fi.write(exe)


def template_submit():
    return f"""Universe = Vanilla
+SingularityImage = "%(IMAGE)s"
Executable     = %(EXECUTABLE)s
Requirements = ( HAS_SINGULARITY ) && ( HAS_CVMFS_unpacked_cern_ch )
should_transfer_files = YES
Output  = %(OUTPUT)s
Log     = %(LOG)s
Error   = %(ERROR)s
transfer_input_files = %(OUTDIR)s
when_to_transfer_output = ON_EXIT
request_cpus = 1
request_disk = 15 GB
request_memory = 10 GB
+ProjectName="collab.futurecolliders"
Queue 1
"""

def template_executable():
    return f"""#!/bin/bash
echo $HOSTNAME
echo "<<<APPTAINER_NAME:" $APPTAINER_NAME
echo "Sourcing setup scripts"
# source /opt/ilcsoft/muonc/ILCSoft.cmake.env.sh
# source /opt/ilcsoft/muonc/init_ilcsoft.sh
source /setup.sh
python -m pip install pandas --user
python -m pip install tqdm --user
python -m pip install dataclasses --user
python -m pip install pyarrow --user
python -c 'import pandas as pd; print(pd.__version__)'
echo "<<<Setup some environment"
echo "source /setup.sh"
which ddsim
which Marlin
echo "<<<Check if test input file was copied from the origin"
echo "<<<stat a file from the cvmfs public repo"
echo "stat /cvmfs/public-uc.osgstorage.org/ospool/uc-shared/public/futurecolliders/BIB10TeV/sim_mm_pruned/BIB_sim_26.slcio"
stat /cvmfs/public-uc.osgstorage.org/ospool/uc-shared/public/futurecolliders/BIB10TeV/sim_mm_pruned/BIB_sim_26.slcio
echo ">>>completed"
echo "<<<Copy some public data to the worker node"

echo "Downloading wget"
apt-get download wget
dpkg -x wget*.deb wget
export WGET=./wget/usr/bin/wget

echo "wget"
$WGET http://osdf-public.tempest.uchicago.edu:1094/ospool/uc-shared/public/futurecolliders/BIB10TeV/sim_mp_pruned/BIB_sim_662.slcio
echo ">>> transfer completed"
echo "<<<stat the local file"
stat BIB_sim_662.slcio
rm -f BIB_sim_662.slcio
echo ">>> Deletions complete. Trying to stashcp something"
export STASHCP=/cvmfs/oasis.opensciencegrid.org/osg-software/osg-wn-client/23/current/el8-x86_64/usr/bin/stashcp
$STASHCP -d osdf:///ospool/uc-shared/project/futurecolliders/data/fmeloni/DataMuC_MuColl10_v0A/v2/reco/photonGun_E_250_1000/photonGun_E_250_1000_reco_0.slcio ./photonGun_E_250_1000_reco_0.slcio
$WGET http://osdf-public.tempest.uchicago.edu:1094/ospool/uc-shared/public/futurecolliders/tuna/code/processing.lcio_to_flat.py
python processing.lcio_to_flat.py -i photonGun_E_250_1000_reco_0.slcio -o tmp.parquet
echo "Job complete"
"""

if __name__ == "__main__":
    main()
