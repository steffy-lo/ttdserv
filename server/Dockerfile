FROM python:3.10.1-slim

WORKDIR /app

RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "main.py"]