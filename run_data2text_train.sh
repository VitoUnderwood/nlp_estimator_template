export MODEL_DIR=checkpoints/data2text
export DATA_DIR=data/data2text

now=$(date "+%Y_%m_%d_%H:%M")

python run_data2text.py \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR/ \
  --vocab_file=$MODEL_DIR/vocab.pkl \
  --vector_file=$MODEL_DIR/vector.pkl \
  --key_file=$MODEL_DIR/key.pkl \
  --val_file=$MODEL_DIR/val.pkl \
  --model_config_file=$MODEL_DIR/config.json \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --output_dir=$MODEL_DIR \
  --max_early_stop_step=10 \
  --maximum_iterations=50
