export MODEL_DIR=checkpoints/NMT
export DATA_DIR=data/NMT
export OUTPUT_DIR=checkpoints/NMT

python run_NMT.py \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR/ \
  --vocab_file=$MODEL_DIR/vocab.txt \
  --model_config_file=$MODEL_DIR/NMT_config.json \
  --max_seq_length=128 \
  --train_batch_size=8 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR