#!/bin/bash
rm -rf /home/$USER/ProdigyAI/spartan/temp/*
echo 'preprocessing part one'
bash local_run_script.sh preprocessing.py temp-instance-cpu 1 two_model with_shutdown
echo 'gpu part one'
bash local_run_script.sh gpu_stage_one.py temp-instance-gpu 1 two_model with_shutdown
echo 'preprocessing part two'
bash local_run_script.sh preprocessing.py temp-instance-cpu 2 two_model with_shutdown
echo 'gpu part two'
bash local_run_script.sh gpu_stage_two.py temp-instance-gpu 2 two_model with_shutdown



# bash local_run_script.sh preprocessing.py temp-instance-cpu 1 two_model without_shutdown
# bash local_run_script.sh gpu_stage_one.py temp-instance-gpu 1 two_model without_shutdown