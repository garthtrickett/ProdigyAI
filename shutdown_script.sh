# gcloud compute instances add-metadata preemp-gpu-v100-1 --zone=us-central1-c --metadata-from-file shutdown-script=/home/garthtrickett/ProdigyAI/shutdown_script.sh

rsync -e "ssh -i /home/garthtrickett/.ssh/gpu_to_cpu -o StrictHostKeyChecking=no" --update -rt /home/garthtrickett/ProdigyAI/temp/run_in_progress.txt garthtrickett@10.128.0.3:/home/garthtrickett/ProdigyAI/temp/

