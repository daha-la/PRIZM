# Installations
This directory contains all tools required for running PRIZM. We recommend using the [installer script](./install_tools.sh), which automatically downloads and sets up all dependencies in this directory:
```bash
bash install_tools.sh
```
On some systems (e.g. HPC clusters), TLS certificate verification may fail when downloading GEMME or JET2. In this case, the installer will automatically retry the download without certificate verification and issue a warning.

If you prefer to install the tools manually, please follow the official instructions:
- GEMME: https://www.lcqb.upmc.fr/GEMME/download.html
- JET2: https://www.lcqb.upmc.fr/JET2/JET2.html
- Foldseek: https://mmseqs.com/foldseek/
If installing manually, please remember to update paths in the GEMME and JET2 `default.conf` files.

After installation, this directory should have the following structure:
installation/
├── install_tools.sh
├── README.md
├── GEMME/
│   ├── gemme.py
│   └── ...
├── JET2/
│   ├── jet
│   └── ...
└── foldseek/
    └── bin/
        └── foldseek