from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the pre-trained model (replace "my_model.h5" with your actual model path)

# filepath to your JSON file
filepath = "dataset.json"

# Read the JSON data into a DataFrame
df = pd.read_json(filepath)

model = load_model("my_model.h5")


# Create a tokenizer for external status
tokenizer = Tokenizer(num_words=1000, oov_token="<UNK>")  # Adjust num_words as needed
tokenizer.fit_on_texts(df['externalStatus'])
external_status_sequences = tokenizer.texts_to_sequences(df['externalStatus'])
max_length = max(len(seq) for seq in external_status_sequences)
padded_sequences = keras.preprocessing.sequence.pad_sequences(external_status_sequences, maxlen=max_length, padding='post')

internal_status_encoded = pd.get_dummies(df['internalStatus'])

# Define an input schema for the API request
class ExternalStatus(BaseModel):
    external_status: str

app = FastAPI()
  
@app.post("/predict_internal_status")
def predict_internal_status(external_status:str):
    

    # Preprocess the text data
    new_sequence = tokenizer.texts_to_sequences([external_status])[0]
    new_sequence = pad_sequences([new_sequence], maxlen=max_length, padding='post')

    # Make prediction using the model
    prediction = model.predict(new_sequence)
    predicted_class = internal_status_encoded.columns[prediction.argmax()]

    return {"predicted_internal_status": predicted_class}
