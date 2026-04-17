# Cellpose-SAM quick test on Ubuntu

Minimal headless smoke test for Cellpose-SAM on an Ubuntu server.

## Clone

    git clone git@github.com:YOUR_GITHUB_USERNAME/cellpose-sam-test.git
    cd cellpose-sam-test

## GitHub SSH key reminder

If needed:

    ssh-keygen -t ed25519 -C "your_email@example.com"
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519
    cat ~/.ssh/id_ed25519.pub

Then paste the printed public key into GitHub:

    Settings -> SSH and GPG keys -> New SSH key

Test:

    ssh -T git@github.com

## Install Conda on Ubuntu 

Miniforge is a good minimal default:

    curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh
    bash Miniforge3-Linux-$(uname -m).sh -b
    ~/miniforge3/bin/conda init bash
    source ~/.bashrc
    conda --version

## Build the environment

The setup script creates the conda environment and installs the required packages:

    chmod +x setup_cellpose_sam.sh
    ./setup_cellpose_sam.sh
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cellpose-sam

## Optional GPU setup (ALREADY INSTALLED, NOT NECESSARY)

First check that the machine has an NVIDIA GPU:

    lspci | grep -i nvidia

If `nvidia-smi` does not work yet, install the Ubuntu NVIDIA driver:

    sudo apt update
    sudo apt install -y ubuntu-drivers-common linux-headers-$(uname -r)
    sudo ubuntu-drivers install
    sudo reboot

After reboot:

    nvidia-smi

If the driver works, reinstall PyTorch with CUDA support inside the conda environment:

    conda activate cellpose-sam
    pip uninstall -y torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

Check CUDA visibility:

    python - <<'PY'
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch cuda:", torch.version.cuda)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    PY

## Run the test

CPU:

    python test_cellpose_sam.py --outdir results_cpu

GPU:

    python test_cellpose_sam.py --outdir results_gpu --use-gpu

## Expected outputs

The script writes:

    results_cpu/
    ├── synthetic_cells.tif
    ├── synthetic_gt_mask.tif
    ├── cellpose_sam_mask.tif
    ├── preview.png
    └── summary.txt

## Notes

This is only a smoke test. It simulates simple microscopy-like data, runs the `cpsam` pretrained model, and checks that inference completes and outputs are saved.

If `--use-gpu` does not work:

- if `nvidia-smi` fails, the NVIDIA driver is missing or broken
- if `nvidia-smi` works but `torch.cuda.is_available()` is `False`, PyTorch was installed without CUDA support
- if Torch sees CUDA but Cellpose still fails, make sure you are running inside the `cellpose-sam` conda environment
