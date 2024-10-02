conda env create -f environment.yml

export QT_PLUGIN_PATH=/.../envs/sam2/lib/python3.10/site-packages/PyQt5/Qt5/plugins
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:/usr/local/cuda-12.1/extras/CUPTI/lib64:/.../envs/sam2/lib/python3.10/site-packages/PyQt5/Qt5/lib:$LD_LIBRARY_PATH