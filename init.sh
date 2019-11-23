#!/bin/bash

apt-get update


# install pyenv and create virtualenv

apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile
source ~/.bash_profile

pyenv install 3.6.8
pyenv virtualenv 3.6.8 pytorch-env
pyenv local pytorch-env

pip install -r requirements.txt
pip install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html


# install kaggle-api and download data from kaggle

pip install kaggle
apt-get install unzip

export KAGGLE_USERNAME=wyy18feb
export KAGGLE_KEY=156af0c38e1395d5db8f5fe3c61c7c89

kaggle competitions download -c sentiment-analysis-on-movie-reviews -p ~/.kaggle
unzip -o ~/.kaggle/sentiment-analysis-on-movie-reviews.zip -d .data/
