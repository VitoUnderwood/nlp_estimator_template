export MODEL_DIR=checkpoints/data2text
export DATA_DIR=data/data2text

python run_data2text.py \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR/ \
  --vocab_file=$MODEL_DIR/vocab.pkl \
  --vector_file=$MODEL_DIR/vector.pkl \
  --key_file=$MODEL_DIR/key.pkl \
  --val_file=$MODEL_DIR/val.pkl \
  --model_config_file=$MODEL_DIR/config.json \
  --max_feat_num=8 \
  --max_seq_length=130 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --output_dir=$MODEL_DIR \
  --input_max_len=128 \
  --output_max_len=128 \
  --maximum_iterations=100 &> logs/data2text.train
