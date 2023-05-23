
rm -rf ~/ProdigyAI/temp/cp.ckpt
rm -rf ~/ProdigyAI/temp/cp_end.ckpt
bash local_run_script.sh $1 preemp-gpu-v100-1 1 one_model without_shutdown pipeline us-central1-a gpu_output $2

while true
do
    if [[ -f "/home/$USER/ProdigyAI/temp/run_in_progress.txt" ]]
    then
        sleep 180
        bash local_run_script.sh $1 preemp-gpu-v100-1 1 one_model without_shutdown pipeline us-central1-a gpu_output resuming
    else
        echo "script finished"
        exit
    fi

done
