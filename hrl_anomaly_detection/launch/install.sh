#!/bin/bash -x

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