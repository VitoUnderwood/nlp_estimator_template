#docker run -p 8502:8501 \
#  --mount type=bind,\source=/Users/vito/PyCharmProjects/nlp_estimator_template/checkpoints/QA_CVAE/save_demo/,\target=/models/QA \
#  -e MODEL_NAME=QA \
#  -t tensorflow/serving &
docker run -t -p 8501:8501 \
  -v "/Users/vito/PyCharmProjects/nlp_estimator_template/checkpoints/QA_CVAE/save_demo:/models/QA" \
  -e MODEL_NAME=QA \
  tensorflow/serving &
