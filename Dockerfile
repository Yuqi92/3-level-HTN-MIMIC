FROM tensorflow/tensorflow:2.4.0-gpu
RUN pip install scikit-learn
RUN pip install tensorflow_hub
RUN mkdir /train_code
COPY . /train_code/
WORKDIR /train_code/
ENTRYPOINT ["python", "train.py"]
