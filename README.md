# Deep Functional Maps
Implementation of Deep Functional Maps Neural Net

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
  ㄴscan_shot
  ㄴscan_lb
</pre>
  
Then go to dist_map directory, and run run_euclidean.ipynb<br />
This will generare the necessary distance map in euclidean distance.<br />

It is possible to run run_geodesic.ipynb as well, but you need to install gdist.<br />
pip install tvb-gdist (this is python binding for calculating geodesic distance written in C++)<br />
Note that it takes 10+ min to get one geodesic distance map, and its size is enormously big.<br />

You may also run CMC.ipynb in the shot folder.<br />
In order to run that, you need scan_shot data and scan_d data.<br />
The scan_d data are available from the google drive link.<br />

For scan_shot data, I could not download those data one by one and upload it,<br />
so I thought it would be better to preprocess them.<br />

After cloning this git, stay at the parent directory and clone another git repo.<br />
git clone https://github.com/fedassa/SHOT.git<br />
Then go to dfn/shot/run_shot.ipynb.<br />
Without changing any variable, just run the entire file.<br />
It will take 10~20minutes creating the data.
