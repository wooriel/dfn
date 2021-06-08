# Deep Functional Net
Implementation of Deep Functional Net

Below link contains the data file directory.<br />
I could not upload some files due to the memory limit.<br />
Before running the training loop, you should have a distance map.<br />
To make that work, please download reg_100 data (registration data scaled by 100). <br />
Also, please make directory structure as below.<br />
https://drive.google.com/drive/folders/1LnDfdcfmzkVQ-wBTZlccZQwQAzaZdFas?usp=sharing<br />
<pre>
ㄴtraining
  ㄴreg_100
  ㄴl2_dist
  ㄴreg_lb (it is already in git)
  ㄴreg_shot (it is already in git)
ㄴtest
  ㄴscan_d
  ㄴscan_lb
</pre>
  
Then go to dist_map directory, and run run_euclidean.ipynb<br />
This will generare the necessary distance map in euclidean distance.<br />

It is possible to run run_geodesic.ipynb as well, but you need to install gdist<br />
pip install tvb-gdist<br />
(this is python binding for calculating geodesic distance written in C++)<br />
