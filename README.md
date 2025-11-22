# Create environment

Download and install Conda/Mamba in your system.

Then use the following command to create a new environment:

```bash
mamba create -f ./environment.yml
mamba activate ai
```

Notice: `ai` is the name defined in `environment.yml` and create as mamba env.

If environment already exist, and want to update require env in it, just run:

```bash
 mamba env update -n ai -f environment.yml
```