FROM tensorflow/tensorflow:1.13.1-py3-jupyter
COPY requirements.txt TF-MNIST.ipynb /app/
WORKDIR /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
#ENTRYPOINT ["jupyter-notebook","--ip=127.0.0.1",  "TF-MNIST.ipynb", "--notebook-dir=]
