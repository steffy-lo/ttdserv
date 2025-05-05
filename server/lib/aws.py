from fastapi import File, UploadFile, HTTPException
import boto3
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read environment variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  # Default to us-east-1 if not set

# Initialize S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def upload_to_s3(file_obj, object_name):
    try:
        s3_client.upload_fileobj(file_obj, BUCKET_NAME, object_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file to S3: {e}")

def file_upload(file: UploadFile = File(...)):
    try:
        s3_client.upload_fileobj(file.file, BUCKET_NAME, file.filename)
        file_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{file.filename}"
        return {"message": "File uploaded successfully", "url": file_url, "object_key": file.filename}
    except Exception as e:
        return {"error": str(e)}

def file_download(object_key):
    try:
        # Download the file from S3
        s3_response = s3_client.get_object(Bucket=BUCKET_NAME, Key=object_key)
        data = s3_response['Body'].read()
        return data
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            raise FileNotFoundError(f"File not found in S3: {object_key}") from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading file from S3: {e}")
