#docker run -p 8502:8501 \
#  --mount type=bind,\source=/Users/vito/PyCharmProjects/nlp_estimator_template/checkpoints/QA_CVAE/save_demo/,\target=/models/QA \
#  -e MODEL_NAME=QA \
#  -t tensorflow/serving &

#docker run -t -p 8502:8501 \
#  -v "/Users/vito/PyCharmProjects/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu:/models/half_plus_two" \
#  -e MODEL_NAME=half_plus_two \
#  tensorflow/serving &


#docker run -t -p 8501:8501 \
#  -v "/Users/vito/PyCharmProjects/nlp_estimator_template/checkpoints/QA_CVAE:/models/QA" \
#  -e MODEL_NAME=QA \
#  tensorflow/serving:1.15.0 &


sudo docker run -t -p 8501:8501 \
  -v "/home/xwwang/nlp_estimator_template/checkpoints/QA_CVAE:/models/QA" \
  -e MODEL_NAME=QA \
  tensorflow/serving:1.15.0 &
