# Getting Started


## Using Docker
### 1. Setup host
Set `host=http://server:8080` in `.streamlit/secrets.toml`
### 2. Build and run the image
`docker-compose up -d --build`
### 3. Verify client is up
Visit http://localhost:8501

**A decent CPU and Memory (> 8GB) Usage for running it as a docker container is needed for WhisperX**

## Local Machine
### 1. Setup host
Set `host = "http://localhost:8080"` in `.streamlit/secrets.toml`
### 2. Run server
```
cd server
python main.py
```
### 3. Run client
```
cd client
streamlit run main.py
```