# Intro

Here are tips on using VSCode within apptainer running on a remote connection, with the specific example of the snowmass machine.

NB: My connection is called `k4toroid_snowmass`, but you can call it whatever you like.

# Links

- https://github.com/microsoft/vscode-remote-release/issues/3066
- https://linuxize.com/post/using-the-ssh-config-file/

# Laptop

On my laptop, in the `${HOME}/.ssh/config` file, I added:

```
Host k4toroid_*
  RemoteCommand apptainer run -B /tank/data/snowmass21/muonc:/data -B /work/$USER:/code -B /home/$USER -B /work/$USER /home/$USER/k4toroid.sif
  RequestTTY yes
Host snowmass k4toroid_snowmass
  HostName login.snowmass21.io
  User tuna
```

The key is `RemoteCommand`, which can execute a command upon login. The other lines are hopefully self-explanatory.

You can check if this works by running `ssh k4toroid_snowmass` from the command line. If successful, it should log into snowmass and immediately start `apptainer`. If that doesn't work, you can also try running `ssh snowmass` to see if a connection without `RemoteCommand` works.

# Snowmass

On `login.snowmass21.io`, in the `${HOME}/.bashrc` file, I added:

```
if [ "$APPTAINER_NAME" == "k4toroid.sif" ]; then
    source /opt/spack/share/spack/setup-env.sh
    spack env activate -V k4prod
    source /opt/spack/opt/spack/linux-ubuntu22.04-x86_64/gcc-11.3.0/mucoll-stack-2023-07-30-ysejlaccel4azxh3bxzsdb7asuzxbfof/setup.sh
    export MARLIN_DLL=$MARLIN_DLL:/opt/MyBIBUtils/lib/libMyBIBUtils.so
fi
```

This sets up the mucoll-stack software whenever the apptainer environment starts, including for VSCode.

# VSCode

- Install the `Remote - SSH` extension
- In Settings, enable "Remote.SSH: Enable Remote Command" (`remote.SSH.enableRemoteCommand`)
- `Connect to...` -> `Connect to Host` -> `k4toroid_snowmass`

