LR=1e-5

OUTPUT_PATH=<PATH-TO-OUTPUT>
MODEL_PATH=<PATH-TO-MODEL>

# OUTPUT
MODEL_OUTPUT_PATH=$OUTPUT_PATH/model
LOG_OUTPUT_PATH=$OUTPUT_PATH/logs
TENSORBOARD_PATH=$OUTPUT_PATH/tensorboard

mkdir -p $MODEL_OUTPUT_PATH
mkdir -p $LOG_OUTPUT_PATH
mkdir -p $TENSORBOARD_PATH

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

deepspeed --num_gpus=1 --master_port $MASTER_PORT main.py \
    --deepspeed \
    --train_file "./data/bella_train_demo.json" \
    --prompt_column inputs \
    --response_column targets \
    --model_name_or_path $MODEL_PATH \
    --output_dir $MODEL_OUTPUT_PATH \
    --tensorboard_dir $TENSORBOARD_PATH \
    --seq_len 2048 \
    --batch_size_per_device 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --max_steps 9000 \
    --save_steps 4500 \
    --learning_rate $LR \
    --finetune \
    --lora_rank 32 \
    &> $LOG_OUTPUT_PATH/training.log