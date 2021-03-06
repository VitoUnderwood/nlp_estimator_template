export MODEL_DIR=checkpoints/data2text
export DATA_DIR=data/data2text

now=$(date "+%Y_%m_%d_%H:%M")

python run_data2text.py \
  --do_predict=true \
  --predict_batch_size=32 \
  --data_dir=$DATA_DIR/ \
  --vocab_file=$MODEL_DIR/vocab.pkl \
  --vector_file=$MODEL_DIR/vector.pkl \
  --key_file=$MODEL_DIR/key.pkl \
  --val_file=$MODEL_DIR/val.pkl \
  --model_config_file=$MODEL_DIR/config.json \
  --max_feat_num=5 \
  --max_seq_length=32 \
  --learning_rate=2e-5 \
  --output_dir=$MODEL_DIR \
  --beam_width=10 \
  --maximum_iterations=50

