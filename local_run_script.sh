#!/bin/bash


source support_files/home_path.txt



find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf




OUTPUT="$(gcloud compute instances describe $2 --zone $7)"

if [[ $OUTPUT == *"preemp-cpu-big-1"* ]]; then
  ip_address="10.128.0.4"
  # ssh-keygen -f "/home/garthtrickett/.ssh/known_hosts" -R "104.155.208.206"
fi

if [[ $OUTPUT == *"preemp-gpu-t4"* ]]; then
  ip_address="10.146.0.39"
  # ssh-keygen -f "/home/garthtrickett/.ssh/known_hosts" -R "35.229.130.39"
fi

if [[ $OUTPUT == *"preemp-gpu-v100-1"* ]]; then
  ip_address="10.128.0.7"
  # ssh-keygen -f "/home/garthtrickett/.ssh/known_hosts" -R "35.229.130.39"
fi

if [[ $OUTPUT == *"TERMINATED"* ]]; then
  echo "Instance is off so turn it on"
  gcloud compute instances start $2 --zone $7
  sleep 35
fi



#sudo rm -rf /garthtrickett/.ssh/known_hosts
rsync -e "ssh -i /home/$USER/.ssh/id_rsa -o StrictHostKeyChecking=no" --update -rt $home_path/ProdigyAI/run_script.sh $USER@$ip_address:$home_path/ProdigyAI/run_script.sh
echo 'run_script synced'
ssh -i /home/$USER/.ssh/id_rsa -X -v -L 5555:localhost:5555 $USER@$ip_address "export PATH="/home/$USER/miniconda3/bin:$PATH"; source activate ProdigyAI; sudo bash $home_path/ProdigyAI/run_script.sh $1 $3 $4 $USER $home_path $5 $6 $8 $9; chown -R $USER:google-sudoers $home_path/ProdigyAI"
echo 'run_script finished'