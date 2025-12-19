# Model installation guide

Uses the model from https://github.com/crozier-del/UDVD-MF-Denoising/tree/v1.0

Instructions to get this working on local device. Instructions are on Crozier's github, but I ran into a few problems - so this includes how I got around them.

## tmux - optional

If using SSH I would recommend using a resumable terminal, as the model takes many hours to train. This means you can run the model from command line without keeping the terminal open.

Steps for this follow

install tmux

` sudo apt install tmux `

activate tmux resumable terminal

` tmux new -s denoise `

exit tmux terminal 

` ctrl + b ` then ` d `

resume tmax 

` tmux attach -t denoise `

## Repo and dependencies

I ran into some version problems while cloning the model and installing packages. Here are the steps to install locally (hopefully without issues).

` git clone https://github.com/crozier-del/UDVD-MF-Denoising `

` cd UDVD-MF-Denoising `

enter environment.yaml and change ` imagecodecs==2024.1.1 ` to ` imagecodecs==2024.6.1 `

create the environment using this updated environment

`  conda env create -n denoise-HDR -f environment.yaml `

download the example data, this is not included in the initial clone due to gitignore

` curl -L -o PtCeO2_030303.tif https://github.com/crozier-del/UDVD-MF-Denoising/raw/v1.0/examples/PtCeO2_030303.tif `

run the model on the example data

` python denoise_mf.py --data "./examples/PtCeO2_030303.tif" --num-epochs 50  --batch-size 1 --image-size 256 `
