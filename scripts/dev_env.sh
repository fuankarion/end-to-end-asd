#EASEE env
conda create -n easee_env -c pytorch -c conda-forge pytorch=1.8.1 torchvision python cudatoolkit=11.1

source activate easee_env

pip uninstall Pillow -y
pip install Pillow-SIMD

pip install python_speech_features
pip install natsort
pip install scipy
pip install sklearn

conda install pytorch-geometric -c rusty1s -c conda-forge