export DATASET_NAME="./datasets/blended_MVS"
accelerate launch --config_file ./configs/fsdp.yaml --main_process_port 10012 run.py \
  --output_dir="./outputs/debug" \
  --vis_vggt \
  --nframes 9 \
  --dataset_name=$DATASET_NAME \
  --train_batch_size 1 \
  --dataloader_num_workers 6 \
