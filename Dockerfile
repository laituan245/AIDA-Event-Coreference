FROM huggingface/transformers-pytorch-gpu
COPY . /app
WORKDIR /app
RUN pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt
ENTRYPOINT ["python3.6", "aida_event_coref.py"]
