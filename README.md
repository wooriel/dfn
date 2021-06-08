# Deep Functional Net
Implementation of Deep Functional Net

Below link contains the data file directory.
I could not upload some files due to the memory limit.
Before running the training loop, you should have distance map.
To make that work, please download reg_100 data from the below link and make directory structure as below.
https://drive.google.com/drive/folders/1LnDfdcfmzkVQ-wBTZlccZQwQAzaZdFas?usp=sharing
ㄴtraining
  ㄴreg_100
  ㄴl2_dist
  ㄴreg_lb (it is already in git)
  ㄴreg_shot (it is already in git)
ㄴtest
  ㄴscan_d
  ㄴscan_lb
  
Then go to dist_map directory, and run run_euclidean.ipynb
This will generare the necessary distance map in euclidean distance.

It is possible to run run_geodesic.ipynb as well, but you need to install gdist
pip install tvb-gdist
(this is python binding for calculating geodesic distance written in C++)
