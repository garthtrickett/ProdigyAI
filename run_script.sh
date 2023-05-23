#!/bin/bash
# sudo rm -rf /root/.ssh/known_hosts
export PATH="~/miniconda3/bin:$PATH"
source activate ProdigyAI

# sync the folders /home/garthtrickett/ProdigyAI/spartan
#

sudo echo "X11Forwarding no" >> /etc/ssh/sshd_config
export DISPLAY=''
sudo /etc/init.d/ssh restart

echo 'SSH restarted'



echo 'syncing spartan files from preemp-cpu to temp instances start'
rsync -e "ssh -i /home/$4/.ssh/gpu_to_cpu -o StrictHostKeyChecking=no" -vrt --update garthtrickett@35.184.82.172:~/ProdigyAI ~/


echo 'syncing spartan files from preemp-cpu to temp instances finished'

whoami;
sudo su $4 << BASH
  whoami;
BASH
echo 'user switched'

cd ~/ProdigyAI



sudo chown -R $4:$4 *

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

cd $5/ProdigyAI/$7/


if [[ $9 == *"first_run"* ]]
    then
        rm -rf /home/$4/ProdigyAI/temp/$1_run_in_progress.txt
    fi


rm -rf nohup.out
rm -rf /home/$4/ProdigyAI/logs/*
chown -R $4:google-sudoers $5/ProdigyAI
whoami;
sudo su $4 << BASH
  whoami;
BASH
echo 'user before python script'
echo $4
echo 'Up to python script'

conda activate ProdigyAI
if [[ $9 == *"NA"* ]]
    then
        python $1 > ~/ProdigyAI/logs/$1.out
    fi
if [[ ! $9 == *"NA"* ]]
    then
        python $1 --resuming $9 --user $4 > ~/ProdigyAI/logs/$1.out
    fi
  

# python deeplob.py --resuming resuming --user garthtrickett 


echo 'Python script finished'

echo 'syncing new files back to preemp-cpu'

echo $4

whoami;
sudo su $4 << BASH
  whoami;
BASH
echo 'user switched'

echo 'syncing data'
rsync -e "ssh -i /home/$4/.ssh/gpu_to_cpu -o StrictHostKeyChecking=no" --update -vrt $5/ProdigyAI/data $4@10.128.0.3:$5/ProdigyAI/
echo 'syncing data finished, syncing models'
rsync -e "ssh -i /home/$4/.ssh/gpu_to_cpu -o StrictHostKeyChecking=no" --update -vrt $5/ProdigyAI/models $4@10.128.0.3:$5/ProdigyAI/
echo 'syncing models finished, syncing temp'
rsync -e "ssh -i /home/$4/.ssh/gpu_to_cpu -o StrictHostKeyChecking=no" --delete -vrt $5/ProdigyAI/temp $4@10.128.0.3:$5/ProdigyAI/
echo 'syncing temp finished, syncing logs'
rsync -e "ssh -i /home/$4/.ssh/gpu_to_cpu -o StrictHostKeyChecking=no" --update -vrt $5/ProdigyAI/logs $4@10.128.0.3:$5/ProdigyAI/
echo 'syncing logs finished'

echo 'syncing new files back to preemp-cpu: finished'

chown -R $4:google-sudoers $5/ProdigyAI

if [[ $6 == *"with_shutdown"* ]]; then
  echo "Shutdown initiated"
  sudo shutdown -h now
  sleep 35
fi


# rsync -e "ssh -i /home/garthtrickett/.ssh/gpu_to_cpu -o StrictHostKeyChecking=no" --update -vrt /home/garthtrickett/ProdigyAI/data/preprocessed garthtrickett@10.128.0.3:/home/garthtrickett/ProdigyAI/data/