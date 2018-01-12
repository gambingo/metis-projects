#!/bin/bash

# Download MiniConda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh
# Install MiniConda
bash ~/anaconda.sh -b -p $HOME/anaconda
# Update path
echo -e '\nexport PATH=$HOME/anaconda/bin:$PATH' >> $HOME/.bashrc && source $HOME/.bashrc
echo "Message: Installed MiniCoda."

# matloblib on ubuntu
pip uninstall matplotlib
apt-get build-dep matplotlib
pip install -U matplotlib
echo "Message: Installed matplotlib."

# packages
conda install pandas -y
conda install shapely -y
pip install descartes
sudo apt install imagemagick -y
#sudo apt install gcc -y
#pip install Cython
echo "Message: Installed all packages."

# Make Directories
cd
cd metis-projects/metis-05-kojak
mkdir images
cd images
mkdir frames
mkdir frames_stage_one
mkdir frames_stage_two
mkdir frames_stage_three
mkdir gifs
mkdir logs
mkdir pickled_objects
mkdir final_plots
cd ..
mkdir data
echo "Message: Made directories."

# Download data
cd data
wget https://s3.amazonaws.com/mike.teczno.com-img/redistricting/North-Carolina-2014.geojson.gz
gunzip North-Carolina-2014.geojson.gz
echo "Message: Downloaded data."

cd ../python-scripts

# Compile Cython code
#source complile_cython.sh
