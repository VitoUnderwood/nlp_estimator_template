export BERT_BASE_DIR=/Users/vito/PyCharmProjects/nlp_estimator_template/checkpoints/chinese_L-12_H-768_A-12
export DATA_DIR=/Users/vito/PyCharmProjects/nlp_estimator_template/data
export OUTPUT_DIR=/Users/vito/PyCharmProjects/nlp_estimator_template/checkpoints/news_cls

python run_news_cls.py \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR

# predict
#export BERT_BASE_DIR=/Users/vito/PyCharmProjects/nlp_estimator_template/checkpoints/uncased_L-12_H-768_A-12
#export GLUE_DIR=/Users/vito/PyCharmProjects/nlp_estimator_template/data
#export TRAINED_CLASSIFIER=/Users/vito/PyCharmProjects/nlp_estimator_template/checkpoints/tmp
#
#python run_classifier.py \
#  --task_name=MRPC \
#  --do_predict=true \
#  --data_dir=$GLUE_DIR/MRPC \
#  --vocab_file=$BERT_BASE_DIR/vocab.txt \
#  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
#  --init_checkpoint=$TRAINED_CLASSIFIER \
#  --max_seq_length=128 \
#  --output_dir=/tmp/mrpc_output/