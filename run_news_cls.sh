export BERT_BASE_DIR=checkpoints/chinese_L-12_H-768_A-12
export DATA_DIR=data/News
export OUTPUT_DIR=checkpoints/news_cls

python run_news_cls.py \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=8 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR


#export BERT_BASE_DIR=checkpoints/chinese_L-12_H-768_A-12
#export DATA_DIR=data/News
#export OUTPUT_DIR=checkpoints/news_cls
#
#python run_news_cls.py \
#  --do_train=true \
#  --data_dir=$DATA_DIR/ \
#  --vocab_file=$BERT_BASE_DIR/vocab.txt \
#  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
#  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
#  --max_seq_length=128 \
#  --train_batch_size=8 \
#  --learning_rate=2e-5 \
#  --num_train_epochs=3.0 \
#  --output_dir=$OUTPUT_DIR

## eval
#export BERT_BASE_DIR=checkpoints/chinese_L-12_H-768_A-12
#export DATA_DIR=data/News
#export OUTPUT_DIR=results/news_cls
#
#python run_news_cls.py \
#  --do_eval=true \
#  --data_dir=$DATA_DIR/ \
#  --vocab_file=$BERT_BASE_DIR/vocab.txt \
#  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
#  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
#  --max_seq_length=128 \
#  --train_batch_size=8 \
#  --learning_rate=2e-5 \
#  --num_train_epochs=3.0 \
#  --output_dir=$OUTPUT_DIR

# predict
#export BERT_BASE_DIR=checkpoints/chinese_L-12_H-768_A-12
#export DATA_DIR=data/News
#export TRAINED_CLASSIFIER=checkpoints/news_cls
#export OUTPUT_DIR=results/news_cls
#
#python run_news_cls.py \
#  --do_predict=true \
#  --data_dir=$DATA_DIR/ \
#  --vocab_file=$BERT_BASE_DIR/vocab.txt \
#  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
#  --init_checkpoint=$TRAINED_CLASSIFIER \
#  --max_seq_length=128 \
#  --output_dir=$OUTPUT_DIR