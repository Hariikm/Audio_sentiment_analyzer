from fastapi import FastAPI, UploadFile, File
import shutil
import os
from typing import Annotated
from sentiment import sentiment
from aud_senti import audio_sentiment

app = FastAPI()

UPLOAD_DIR = "/tmp/uploads"  # Directory to save uploaded files

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # Define file path
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Use the file path with your functions
        sent_txt = sentiment(file_path)
        sent_aud = audio_sentiment(file_path)

        final_str = f"The text actually is {sent_txt} and the audio sentiment is {sent_aud}"

        return {"result": final_str}
    finally:
        # Clean up the file after processing
        os.remove(file_path)