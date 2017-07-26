Instructions on how to do pose estimation:

Original datasets are located on /hrl_file_server/
Format: bagfiles

Steps for pose estimation:

1. Read the bagfiles and make pickle files that are easier to address in python

2. Create training and testing datasets from the pickle files
  a. create_basic_dataset.py: description
  b. create_sliced_dataset.py: description
  c. create_test_dataset.py: description
  
3. Use various learning methods to perform pose estimation
