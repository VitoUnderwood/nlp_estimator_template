python word2vec.py \
--corpus="data/phvm_data/processed/corpus.txt" \
--output_dir=checkpoints/word2vec \
--word2vec_name=word2vec.model \
--min_count=1 \
--vector_size=300 &> logs/word2vec.train
