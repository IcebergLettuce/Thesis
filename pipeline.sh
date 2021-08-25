#!/bin/bash
python main.py train --name myrun --config example/configuration.yaml --tdata example/mock-image-label-pairs.npz --host local
python main.py generate --name myrun -N 100 --host local
python main.py report --name myrun --host local
python main.py utility  --name myrun --host local --config example/configuration.yaml --patients example/patients/ --tdata seiton/myrun/synthetic/synthetic.npz
python main.py report --name myrun --host local
python main.py distance --name myrun --host local --ref example/mock-image-label-pairs.npz
python main.py report --name myrun --host local