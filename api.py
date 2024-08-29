from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch
import pandas as pd

from model.narx import NarxModel
from model.narxbuffer import NarxBuffer
from model.variablenames import endog_variable_names, exog_variable_names, model_in_variable_order, model_out_variable_order

# device = torch.device("cpu")



# Setup the model
model = NarxModel()
model.eval()
model.load_state_dict(
    torch.load("state_dicts/narx.pth",
               map_location=torch.device('cpu'), weights_only=True)
)

# example post data for the API docs
PAST = [
    [50, 50, 0, 0.001061799, 0, 0, 0.04269555, 1.995, 1.17383721, 0.6690872],
    [50, 50, 0, 0.00150618, 0, 0, 0.04369555, 2, 1.22023256, 0.6955326],
    [50, 50, 0, 0.00200618, 0, 0, 0.04269555, 2.05, 1.17465116, 0.6695512],
    [50, 50, 0, 0.00271623, 0, 0.001089387, 0.043696383, 2.1, 1.16000000, 0.6612000],
    [50, 50, 0, 0.003136011, 0.000381793, 0.000340433, 0.04389655, 2.24, 1.14081395, 0.6557399],
    [50, 50, 0, 0, 0.000411161, 0, 0.0317064, 2.3, 1.02802326, 0.5943002],
    [50, 50, 0, 0.001802589, 0, 0.000953213, 0.051723067, 2.38, 1.12941860, 0.6550628],
]

FUTURE = [90, 5, 5]

app = FastAPI()

# Define the request body
class PredictionRequest(BaseModel):
    past: list[list[float]] = Field(
        ...,
        example=PAST
    )
    future: list[float] = Field(
        ...,
        example=FUTURE
    )

# The predict endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    past = request.past
    future = request.future

    # Convert past data to DataFrame
    initial_data = pd.DataFrame(
        past, columns=endog_variable_names + exog_variable_names)

    # Initialize input data buffer
    buffer = NarxBuffer(
        endog_variable_names=endog_variable_names,
        exog_variable_names=exog_variable_names,
        model_in_variable_order=model_in_variable_order,
        model_out_variable_order=model_out_variable_order,
        t_endog=7,
        t_exog=7,
    )

    # Populate buffer with initial data
    buffer.populate_buffer_from_df(initial_data)
    
    # Predict for 7 days
    DAYS = 7
    predictions = []
    for _ in range(DAYS):
        with torch.inference_mode():
            # Unpack FUTURE list to pass as arguments C, P, L
            model_input_vars = buffer.feed_model(*future)
            # Get the prediction
            prediction = torch.clamp(model(model_input_vars), min=0)
            predictions.append(prediction.tolist())
            # Update buffer with the prediction
            buffer.update_buffer(prediction)

    methane = [row[-1] for row in predictions]
    return {"predictions": predictions, "methane": methane}



