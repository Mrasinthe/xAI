# -------------------------------------------------------------------------------
import uvicorn
from fastapi.openapi.utils import get_openapi
import json
from aiofile import async_open
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI, File, UploadFile, Depends, Form
from application import predict, read_imagefile, explain_lime  # ,read_model
from application import ShapModelExplainer
from application import OcclusionSensitityModelExplainer
from application import get_ClassName
from PIL import Image
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import io

ShapExplainer = ShapModelExplainer()
OcclusionExplainer = OcclusionSensitityModelExplainer()
from pydantic import BaseModel
import base64
import numpy as np
import matplotlib.pyplot as plt

description = """
xAI Microservices APIs helps you to understand the internal model structure and provide you explanation.
## Image Class Prediction Service

You can just pass an image to the Predict API and get prediction back as JSON

## LIME and SHAP Explainability Services

Just pass your image to the LIME Microservice and this service provide you the results in JSON

## Occlusion Sensitivity Explainability Service
* *Send Image True Label** (_cardboard,glass,metal,paper,plastic,trash_).
"""

app = FastAPI(
    openapi_url="/building/openapi.json",
    docs_url="/building/docs",
    title="XAI Microservices",
    description=description,
    version="0.0.1",
    terms_of_service="https://dps.cs.ut.ee/index.html",
    contact={
        "name": "Mehrdad Asadi, Ph.D.",
        "url": "https://dps.cs.ut.ee/people.html",
        "email": "mehrdad.asadi@ut.ee",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    servers=[{"url": "http://192.168.42.139"}],
    # routes=app.routes,
)


class ClassLabel(BaseModel):
    imagetype: str


@app.post("/explain_all/image")
async def explain_api(file: UploadFile = File(...), base: ClassLabel = Depends()):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    filename = file.filename.split(".")[0]
    input = base.dict()
    var = input['imagetype']
    classNum = get_ClassName(var)

    if not extension:
        return "Image must be jpg or png format!"
    label_number = get_ClassName(var)

    image = read_imagefile(await file.read())

    ####################### for Occlusion Explanation ########################
    Occexplanation = OcclusionExplainer.explain_occlusion(image, classNum)

    image_array_Occ = np.array(Occexplanation, dtype=np.uint8)
    image_Occ = Image.fromarray(image_array_Occ)

    # Convert the image to a base64-encoded string
    image_buffer_Occ = io.BytesIO()
    image_Occ.save(image_buffer_Occ, format='JPEG')
    Occlimage_base64 = base64.b64encode(image_buffer_Occ.getvalue()).decode('utf-8')

    ####################### for Shap Explanation ########################

    Shapexplanation, shap_V_plot_image, prediction = ShapExplainer.explain_shap(image, filename)

    # Convert the bar shap_V plot image to a base64-encoded string
    shap_V_plot_buffer = io.BytesIO()
    shap_V_plot_image.save(shap_V_plot_buffer, format='PNG')
    shap_V_plot_base64 = base64.b64encode(shap_V_plot_buffer.getvalue()).decode('utf-8')

    ####################### for Lime Explanation ########################

    Limeexplaination, top_T, top_T_plot_image, lime_explanation, segments, bar_plot_image, segment_overlay_array = explain_lime(
        image)

    image_array_Lime = np.array(Limeexplaination, dtype=np.uint8)
    image_Lime = Image.fromarray(image_array_Lime)

    # Convert the image to a base64-encoded string
    image_buffer_Lime = io.BytesIO()
    image_Lime.save(image_buffer_Lime, format='JPEG')
    Lime_image_base64 = base64.b64encode(image_buffer_Lime.getvalue()).decode('utf-8')

    # Convert the bar plot image to a base64-encoded string
    bar_plot_buffer = io.BytesIO()
    bar_plot_image.save(bar_plot_buffer, format='PNG')
    bar_plot_base64 = base64.b64encode(bar_plot_buffer.getvalue()).decode('utf-8')

    # Convert segment overlay array to an image
    segment_overlay_image_array = np.array(segment_overlay_array, dtype=np.uint8)
    segment_overlay_image = Image.fromarray(segment_overlay_image_array)

    # Convert the segment overlay image to a base64-encoded string
    segment_overlay_buffer = io.BytesIO()
    segment_overlay_image.save(segment_overlay_buffer, format='PNG')
    segment_overlay_base64 = base64.b64encode(segment_overlay_buffer.getvalue()).decode('utf-8')

    # Convert the top labels plot image to a base64-encoded string
    top_T_plot_buffer = io.BytesIO()
    top_T_plot_image.save(top_T_plot_buffer, format='PNG')
    top_T_plot_base64 = base64.b64encode(top_T_plot_buffer.getvalue()).decode('utf-8')

    return {"Occexplanation": Occexplanation, "Occlimage_base64": Occlimage_base64, "Shapexplanation": Shapexplanation,
            "shap_V_plot_base64": shap_V_plot_base64, "prediction": prediction, "lime_explanation": lime_explanation,
            "top_T": top_T, "Lime_image_base64": Lime_image_base64, "segments": segments,
            "bar_plot_base64": bar_plot_base64, "segment_overlay_base64": segment_overlay_base64,
            "top_T_plot_base64": top_T_plot_base64}


app.mount("/", StaticFiles(directory="."))

with open('openapi.json', 'w') as f:
    json.dump(app.openapi(), f)

if __name__ == "__main__":
    uvicorn.run(app, port=8081)
