export MODEL_DIR=checkpoints/QA_CVAE
export DATA_DIR=data/QA_CVAE

python run_QA_CVAE.py \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR/ \
  --vocab_file=$MODEL_DIR/vocab.pkl \
  --word2vec_file=$MODEL_DIR/vector.pkl \
  --model_config_file=$MODEL_DIR/config.json \
  --max_seq_length=50 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --output_dir=$MODEL_DIR \
  --input_max_len=128 \
  --output_max_len=128 \
  --maximum_iterations=100
