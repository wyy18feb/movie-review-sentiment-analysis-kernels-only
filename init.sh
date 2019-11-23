#!/bin/bash

apt-get update
apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo "eval $(pyenv init -)" >> ~/.bash_profile
echo "eval $(pyenv virtualenv-init -)" >> ~/.bash_profile
source ~/.bash_profile

pyenv install 3.6.8
pyenv virtualenv 3.6.8 pytorch-env
pyenv local pytorch-env

pip install -r requirements.txt
pip install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
