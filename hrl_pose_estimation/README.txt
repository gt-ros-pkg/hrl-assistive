Instructions on how to do pose estimation:

Original datasets are located on /hrl_file_server/
Format: bagfiles

Steps for pose estimation:

1. Read the bagfiles and make pickle files that are easier to address in python
  a. Visualizing labels in rviz
    i. autobed_bagreading_republishpose.py: Publishes topics with labels from a bagfile
    ii. In the config folder, rosrun rviz rviz -d view_bed_mocap.rviz: Subscribes to topics and visualizes labels in 3D
  b. bag_to_p.py: associates latest mocap with next pressure mat, trashs mats that do not have all updated mocaps
    i. bag_to_p_sitting.py: use this for the same thing, but with sitting datasets

2. Create training and testing datasets from the pickle files
  a. create_basic_dataset.py: read files created by bag_to_p.py and clumps together in a single folder. Pops random sets.
   
3. Use various learning methods to perform pose estimation
  a. trainer_convnet.py: does neural net training
    i. convnet.py: configure CNN layers and hyperparameters
    
4. Test subjects on the model you've built.
  a. plot_loss.py: plots losses and prints errors/standard deviations for entire validation datasets


Note: The kinematics convnet will ONLY work with Pytorch version 0.3.1. It will not work with Pytorch version 0.4.1.


