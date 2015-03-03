#!/bin/bash -x

sudo apt-get install swig
sudo apt-get install python-joblib
sudo pip install cython
sudo pip install h5py


cd ~/git 
git clone https://github.com/PyMVPA/PyMVPA.git
cd PyMVPA
python setup.py build_ext
sudo python setup.py install

cd ~/svn
svn checkout svn://svn.code.sf.net/p/ghmm/code/trunk/ghmm ghmm
cd ghmm
./autogen.sh
./configure
make 
sudo make install

cd ~/svn
svn co https://code.ros.org/svn/ros-pkg/stacks/laser_drivers/trunk/hokuyo_node/

# select java7
sudo update-alternatives --config java
sudo pip install pysmac