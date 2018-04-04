load('~/Downloads/hrl-assistive/hrl_dressing/src/hrl_dressing/calibration/singlesensor_armmount_newholder_combined.mat');
capneg = -capacitance;
capallpos = -capacitance - min(-capacitance) + 0.001;
disp(min(capallpos));