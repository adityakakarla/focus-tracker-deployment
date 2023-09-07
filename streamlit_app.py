import streamlit as st
from camera_input_live import camera_input_live
from roboflow import Roboflow
from playsound import playsound
import time

rf = Roboflow(api_key="NBUvQ56DQdrvX3vNmz90")
project = rf.workspace().project("photo-detector-v3")
model = project.version(3).model

image = camera_input_live()
st.image(image)

with open('input_image.jpg', 'wb') as f:
    f.write(image.getvalue())

result = model.predict('input_image.jpg', confidence=40, overlap=30).json()

if len(result['predictions']) >= 1:
    playsound('phone.mp3')
    time.sleep(1)


    

# infer on a local image
#st.write(model.predict('input_image.jpg', confidence=40, overlap=30).json())

# visualize your prediction
#model.predict(image, confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
