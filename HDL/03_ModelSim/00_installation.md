
#
# ModelSim on Ubuntu 22.04 (ver20.1.1)
#
https://gist.github.com/Razer6/cafc172b5cffae189b4ecda06cf6c64f


# Installation requirements
The free version of Modelsim is a 32-bit binary and therefore requires certain 32-bit libraries in order to work correctly. For Ubunutu, install the following packages

sudo dpkg --add-architecture i386
sudo apt-get update
sudo apt-get install libc6:i386 libncurses5:i386 libstdc++6:i386 lib32ncurses6 libxft2 libxft2:i386 libxext6 libxext6:i386


# Installation
Download the ModelSim - Intel FPGA edition installer (both packages) from the Intel homepage.
https://download.altera.com/akdlm/software/acdsinst/20.1std.1/720/ib_installers/ModelSimSetup-20.1.1.720-linux.run

Make the installer executable

chmod +x ModelSimSetup-20.1.1.720-linux.run
Run the installer and install ModelSim:

./ModelSimSetup-20.1.1.720-linux.run


# Path
/home/john/intelFPGA/20.1/modelsim_ase/bin


# Run
./vsim