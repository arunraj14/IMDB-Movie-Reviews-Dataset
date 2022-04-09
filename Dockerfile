FROM python:3.7

WORKDIR /usr/app
COPY . /usr/app/

EXPOSE 8000

RUN pip3 install -r requirements.txt
RUN python3 -m nltk.downloader stopwords

CMD ["python","flask_app.py"]