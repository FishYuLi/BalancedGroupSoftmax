HH=`pwd`
#
# cd $HH

conda create -n mmdet python=3.7 -y
# conda init bash
# bash
# conda activate mmdettry
export PATH=/opt/conda/envs/mmdet/bin:$PATH
echo $PATH
which python

pip install cython
pip install numpy
pip install torch
pip install torchvision
pip install pycocotools
pip install mmcv
pip install matplotlib
pip install terminaltables
# pip install lvis

cd lvis-api/
python setup.py develop

cd $HH
python setup.py develop

# cd $HH
# OMP_NUM_THREADS=3 ./tools/dist_train.sh configs/finalruns/4_htc_dconv_c3-c3_mstrain_400_1400_x101_32x4d_fpn_20e_lvis_insta_cos_is.py 8
