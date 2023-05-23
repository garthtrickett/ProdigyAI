#!/bin/bash
# echo 'preprocessing'

bash local_run_script.sh orderbook_preprocessing.py preemp-cpu-big-1 1 one_model without_shutdown pipeline us-central1-a orderbook_preprocessed NA
bash local_run_script.sh preprocessing.py preemp-cpu-big-1 1 one_model without_shutdown pipeline us-central1-a preprocessed NA
bash local_run_script.sh sync_data.py preemp-cpu-big-1 1 one_model without_shutdown pipeline us-central1-a preprocessed NA


bash local_run_script.sh tabl.py preemp-gpu-v100-1 1 one_model without_shutdown pipeline us-central1-a gpu_output
bash local_run_script.sh deeplob.py preemp-gpu-v100-1 1 one_model without_shutdown pipeline us-central1-a gpu_output NA
bash local_run_script.sh sync_data.py preemp-gpu-v100-1 1 one_model without_shutdown pipeline us-central1-a preprocessed NA


bash local_run_script.sh orderbook_preprocessing.py preemp-cpu-big-1 1 one_model without_shutdown pipeline
echo 'gpu'
bash local_run_script.sh t95_bl_gam_rhn.py preemp-gpu 1 one_model with_shutdown third_party_libraries/gam_rhn/95-FI2010

# bash local_run_script.sh preprocessing.py temp-instance-cpu 1 one_model without_shutdown pipeline
# bash local_run_script.sh gpu_stage_one.py temp-instance-gpu 1 one_model without_shutdown pipeline

# bash local_run_script.sh rocket_tests.py temp-instance-gpu 1 one_model without_shutdown pipeline
# bash local_run_script.sh t95_bl_gam_rhn.py temp-instance-gpu 1 one_model without_shutdown third_party_libraries/gam_rhn/95-FI2010

# bash local_run_script.sh preprocessing.py temp-instance-cpu 1 one_model without_shutdown pipeline

# stuff in here can help 2 https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73
wget http://us.download.nvidia.com/tesla/440.64.00/NVIDIA-Linux-x86_64-440.64.00.run
sudo bash NVIDIA-Linux-x86_64-440.64.00.run
wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.39_linux.run
sudo bash cuda_10.1.105_418.39_linux.run
# tensor rt6 
wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/6.0/GA_6.0.1.5/tars/TensorRT-6.0.1.5.Ubuntu-18.04.x86_64-gnu.cuda-10.1.cudnn7.6.tar.gz?bGQ5KKdOEDgKqUxDvCzBYSdzdzutO0AYCq4zsn-46vXfSYvMWk9KciPslOJTDoxU57sZLzzYTGE7a6ea6l6X4usDOIofZiPfAQhFZHblnOghgfk-erM6tkQhtmrak7KdDyRucs6xPXKN3SU_oKnnu7aMuCINJZ1gGJie2DUNe1p-_Z7wmsOEKbPrJykUJfBHzV2GSsPyF4S6rQRUInCGwijR2Dg5Wy5AHC96LrKx3AD3f_QH_pKpVlCJ81ws_A
mv TensorRT-6.0.1.5.Ubuntu-18.04.x86_64-gnu.cuda-10.1.cudnn7.6.tar.gz?bGQ5KKdOEDgKqUxDvCzBYSdzdzutO0AYCq4zsn-46vXfSYvMWk9KciPslOJTDoxU57sZLzzYTGE7a6ea6l6X4usDOIofZiPfAQhFZHblnOghgfk-erM6tkQhtmrak7KdDyRucs6xPXKN3SU_oKnnu7aMuCINJZ1gGJie2DUNe TensorRT-6.0.1.5.Ubuntu-18.04.x86_64-gnu.cuda-10.1.cudnn7.6.tar.gz
version=6.0.1.5
os=Ubuntu-18.04
arch=$(uname -m)
cuda=cuda-10.1
cudnn=cudnn7.6
tar xzvf TensorRT-${version}.${os}.${arch}-gnu.${cuda}.${cudnn}.tar.gz
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TensorRT-${version}/lib>
cd TensorRT-${version}/python
source activate prodigyAI
sudo python3 -m pip install tensorrt-6.0.1.5-cp37-none-linux_x86_64.whl
cd TensorRT-${version}/uff
python3 -m pip install uff-0.6.5-py2.py3-none-any.whl
which convert-to-uff
cd TensorRT-${version}/graphsurgeon
python3 -m pip install graphsurgeon-0.4.1-py2.py3-none-any.whl

#cuddn
wget https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.1_20191031/Ubuntu18_04-x64/libcudnn7_7.6.5.32-1%2Bcuda10.1_amd64.deb?jEvrF0INH01y6aBaKQgiEAkq4wkUZuBKQ47diIIaUYsmSuk0TocuM0cqOHNTlqxwJB84cTjc6bfZmbb_UZPk2IBbKzUPFq0Z6C0wyg9cA6_-zkDBNljJL7sERQCcIAfZ5fCU20cAipJjcRjcIbAza1jG56QSlepSjnUMaUZ8QvvFJ15MRRwZM7bKyEbK3KUJ168p3f_VimDAjhF5bGaN606smuVBDTETHXZR-7tRx_6HF7S4FK5wYKh5fmCntFIJ
mv libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb?jEvrF0INH01y6aBaKQgiEAkq4wkUZuBKQ47diIIaUYsmSuk0TocuM0cqOHNTlqxwJB84cTjc6bfZmbb_UZPk2IBbKzUPFq0Z6C0wyg9cA6_-zkDBNljJL7sERQCcIAfZ5fCU20cAipJjcRjcIbAza1jG56QSlepSjnUMaUZ8QvvFJ15MRRwZM7bKyEbK3KUJ168p libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb
wget https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.1_20191031/Ubuntu18_04-x64/libcudnn7-dev_7.6.5.32-1%2Bcuda10.1_amd64.deb?_v6lHoIS29hJT6xLjcIAp_FtpEmmjnM9vM5rf3TBrmjxVEdLMcbQUsCFTf5QhprD3fFTSAgkA7e2-MA3rZqRf0gWSU8kDu-Qa1UQsV8ZeS8MM3l1Cwos0cE_gntM94p6CpbcO-6BXKeI5P_t3qq4e5UVMhbfyrT_-JZuJLOGVSVuFOsXsVl1mCZnp5KgSEHpONcalAaA_YuVrrxPjbQSQIhM99YApjUU0jiB4xHi0h7oRqtNva1S5NaBjFZ6tO07WtJvTQ
mv libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb?_v6lHoIS29hJT6xLjcIAp_FtpEmmjnM9vM5rf3TBrmjxVEdLMcbQUsCFTf5QhprD3fFTSAgkA7e2-MA3rZqRf0gWSU8kDu-Qa1UQsV8ZeS8MM3l1Cwos0cE_gntM94p6CpbcO-6BXKeI5P_t3qq4e5UVMhbfyrT_-JZuJLOGVSVuFOsXsVl1mCZnp5KgSEHp libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb
wget https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.1_20191031/Ubuntu18_04-x64/libcudnn7-doc_7.6.5.32-1%2Bcuda10.1_amd64.deb?usWu6ReaV5VuziGM6E3Y0ki3RFJdZecTmIuoEgSFM1sdAHPxslNdGl8uNu_uBS7RgcmdAFLAG8JoxK9xUAjbG-zuG143JTV6LZCjIrR_TgIn5KIDLt_0UB9MSu0hojG1XQfivQa-2prO5lPKJlXDmWRvNaolI6D_JDfWBSsVDG7zsrVE_509vXURQCmx0gB2kWSzc-iaxRYWTzoCNgcDHoaIboB7xk3fgxiGKePQxLo83dXK4wVDWDMsAxiWQ1KyKgIrog
mv libcudnn7-doc_7.6.5.32-1+cuda10.1_amd64.deb?usWu6ReaV5VuziGM6E3Y0ki3RFJdZecTmIuoEgSFM1sdAHPxslNdGl8uNu_uBS7RgcmdAFLAG8JoxK9xUAjbG-zuG143JTV6LZCjIrR_TgIn5KIDLt_0UB9MSu0hojG1XQfivQa-2prO5lPKJlXDmWRvNaolI6D_JDfWBSsVDG7zsrVE_509vXURQCmx0gB2 libcudnn7-doc_7.6.5.32-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-doc_7.6.5.32-1+cuda10.1_amd64.deb



