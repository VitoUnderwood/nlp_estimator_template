export CUDA_VISIBLE_DEVICES=0
export MODEL_DIR=checkpoints/QA_CVAE
export DATA_DIR=data/QA_CVAE

python run_QA_CVAE.py \
  --do_predict=true \
  --data_dir=$DATA_DIR/ \
  --vocab_file=$MODEL_DIR/vocab.txt \
  --model_config_file=$MODEL_DIR/config.json \
  --max_seq_length=128 \
  --predict_batch_size=4 \
  --learning_rate=2e-5 \
  --output_dir=$MODEL_DIR \
  --input_max_len=128 \
  --output_max_len=128 \
  --maximum_iterations=100 \
  --beam_width=5
