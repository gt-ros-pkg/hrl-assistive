Instructions on how to do pose estimation:

Original datasets are located on /hrl_file_server/
Format: bagfiles

Steps for pose estimation:

1. Read the bagfiles and make pickle files that are easier to address in python
  a. Visualizing labels in rviz
    i. autobed_bagreading_republishpose.py: Publishes topics with labels from a bagfile
    ii. In the config folder, rosrun rviz rviz -d view_bed_mocap.rviz: Subscribes to topics and visualizes labels in 3D

2. Create training and testing datasets from the pickle files
  a. create_basic_dataset.py: description
  b. create_sliced_dataset.py: description
  c. create_test_dataset.py: description
  
3. Use various learning methods to perform pose estimation
