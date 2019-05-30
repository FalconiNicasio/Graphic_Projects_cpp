#!/bin/bash

sudo apt-get install cmake
sudo apt-get install opengl
cd /usr/bin/
sudo wget https://www.vtk.org/files/release/8.1/VTK-8.1.2.tar.gz
sudo tar -xvf VTK-8.1.2.tar.gz
sudo cmake .
sudo make 