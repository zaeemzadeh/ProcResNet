

 th main.lua -netType resnet-flat -depth 164 -batchSize 128 -nGPU 1 -nThreads 4 -dataset cifar10 -nEpochs 300 -shareGradInput false -optnet true |& tee log



## Install Torch
1. Install the Torch dependencies:
  ```bash
  curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash -e
  ```

2. Install Torch in a local folder:
  ```bash
  git clone https://github.com/torch/distro.git ~/torch --recursive
  cd ~/torch; ./install.sh
  ```

If you want to uninstall torch, you can use the command: `rm -rf ~/torch`

* NOTE: https://github.com/torch/cutorch/issues/797

## Install the Torch cuDNN v5 bindings
```bash
git clone -b R5 https://github.com/soumith/cudnn.torch.git
cd cudnn.torch; luarocks make
```

* Install cudnn v5


# Install Magma

* http://jinjiren.github.io/blog/gpu-math-using-torch-and-magma/

* make.inc.openblas

* dpkg -L libopenblas-base

* add export OPENBLASDIR=/usr/lib and export CUDADIR=/usr/local/cuda to ~/.bashrc

* source ~./bashrc

* make (if error, run fix_magma first)

