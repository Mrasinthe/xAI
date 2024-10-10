#-------------------------------------------------------------------------------
import uvicorn
from fastapi.openapi.utils import get_openapi
import json
from aiofile import async_open
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI, File, UploadFile,Depends, Form, Body
from application import predict, read_imagefile, explain_lime, load_model#,read_model 
from application import ShapModelExplainer
from application import OcclusionSensitityModelExplainer
from application import get_ClassName
from application import explain_occ_sensitivity_raw
from PIL import Image
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import io
ShapExplainer = ShapModelExplainer()
OcclusionExplainer = OcclusionSensitityModelExplainer()
####################FOR REQUEST BODY####################
from pydantic import BaseModel
import base64
import numpy as np
import matplotlib.pyplot as plt
import sys
import gzip

description = """
xAI Microservices APIs helps you to understand the internal model structure and provide you explanation.
## Image Class Prediction Service

You can just pass an image to the Predict API and get prediction back as JSON

## LIME and SHAP Explainability Services

Just pass your image to the LIME Microservice and this service provide you the results in JSON

## Occlusion Sensitivity Explainability Service
* *Send Image True Label** (_cardboard,glass,metal,paper,plastic,trash_).
"""

# def create_application() -> FastAPI:
#     application = FastAPI(openapi_url="/building/openapi.json", docs_url="/building/docs")

#    application.include_router(create_building.router, prefix="/building", tags=["building"])
#    application.include_router(modify_building.router, prefix="/building", tags=["building"]
#    application.include_router(get_building.router, prefix="/building", tags=["building"])
#    application.include_router(get_buildings.router, prefix="/building", tags=["building"])
#     application.include_router(remove_building.router, prefix="/building", tags=["building"])
#     application.include_router(assign_users_to_building.router, prefix="/building", tags=["building"])
#     return application


# app = create_application()


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
     servers=[{"url":"http://192.168.42.139"}], 
   # routes=app.routes,
)




#app = FastAPI()

#def my_schema():
#    if app.openapi_schema:
#        return app.openapi_schema
#    openapi_schema = get_openapi(
#        description="""testing""",
#        title="XAI Microservices",
#        terms_of_service="https://dps.cs.ut.ee/index.html",
#        version="0.0.1",
#        servers=[{"url": "http://192.168.42.93"}],
#        routes=app.routes,
#    )
#    app.openapi_schema = openapi_schema
#    return app.openapi_schema

#app.openapi = my_schema

#with open('openai.json', 'w') as f:
#  json.dump(app.openapi(), f)
@app.get("/test")
def read_root():
    print("GET api is running")
    return {"Hello": "World"}


@app.get("/test2")
def view_image():
    image_path = 'dataset-resized/cardboard/cardboard1.jpg'
    image = Image.open(image_path)
    image_width, image_height = image.size
    image.close()

    return {"image_width": image_width, "image_height": image_height}

@app.get("/test3")
def view_image2():
    basePath = "dataset-resized"
    image_count = 0

    for filename in os.listdir(basePath):
        # if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_count += 1

    print("Number of images:", image_count)
    return image_count



@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    print("POST api is running after asyncdef")
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)
    return prediction





@app.post("/explain_lime/image") 
async def explain_api(file: UploadFile = File(...), mlModel:UploadFile = File(...), ImageFileBytes: bytes = File(...)):  
    print(file.filename)
    print(ImageFileBytes)


    # saved_model_filename = "uploaded_model.h5"
    # model_content = mlModel.file.read()

    # # Save the uploaded model
    # with open(saved_model_filename, "wb") as model_file:
    #     model_file.write(model_content)

    # # Load the saved model using the load_model function
    # loaded_model = load_model(saved_model_filename)



    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(ImageFileBytes)
    explaination, top_T, top_T_plot_image, segments, bar_plot_image, segment_overlay_array, pred,lime,segment__img,top_labels_names,scores= explain_lime(image) #loaded_model ,bar_plot_segments,bar_plot_importance


    image_array = np.array(explaination, dtype=np.uint8)
    image = Image.fromarray(image_array)

    # Convert the image to a base64-encoded string
    # image_buffer = io.BytesIO()
    # image.save(image_buffer, format='JPEG')
    # image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    # image_buffer.close() 

    image_buffer = io.BytesIO()
    lime.save(image_buffer, format='PNG')
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    image_buffer.close() 

    # Convert the bar plot image to a base64-encoded string
    bar_plot_buffer = io.BytesIO()
    bar_plot_image.save(bar_plot_buffer, format='PNG')
    bar_plot_base64 = base64.b64encode(bar_plot_buffer.getvalue()).decode('utf-8')
    bar_plot_buffer.close() 


    # Convert segment overlay array to an image
    segment_overlay_image_array = np.array(segment_overlay_array, dtype=np.uint8)
    segment_overlay_image = Image.fromarray(segment_overlay_image_array)

    # Convert the segment overlay image to a base64-encoded string
    # segment_overlay_buffer = io.BytesIO()
    # segment_overlay_image.save(segment_overlay_buffer, format='PNG')
    # segment_overlay_base64 = base64.b64encode(segment_overlay_buffer.getvalue()).decode('utf-8')
    # segment_overlay_buffer.close() 

    
    segment_overlay_buffer = io.BytesIO()
    segment__img.save(segment_overlay_buffer, format='PNG')
    segment_overlay_base64 = base64.b64encode(segment_overlay_buffer.getvalue()).decode('utf-8')
    segment_overlay_buffer.close() 


    # Convert the top labels plot image to a base64-encoded string
    top_T_plot_buffer = io.BytesIO()
    top_T_plot_image.save(top_T_plot_buffer, format='PNG')
    top_T_plot_base64 = base64.b64encode(top_T_plot_buffer.getvalue()).decode('utf-8')
    top_T_plot_buffer.close() 

    print("TESTINGGGGGGGGG") 

    return {"top_T": top_T, "image_base64": image_base64, "segments": segments,"bar_plot_base64": bar_plot_base64 ,"segment_overlay_base64": segment_overlay_base64, "top_T_plot_base64" : top_T_plot_base64 ,"pred":pred,"top_labels_names":top_labels_names,"scores":scores}
    #    ,"bar_plot_segments":bar_plot_segments,"bar_plot_importance":bar_plot_importance}


@app.post("/explain_shap/image")
async def explain_api(file: UploadFile = File(...), mlModel:UploadFile = File(...), ImageFileBytes: bytes = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    filename = file.filename.split(".")[0]
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(ImageFileBytes)
    explaination,shap_V_plot_image, prediction = ShapExplainer.explain_shap(image,filename)
    # shap_S_plot_image
   
   
   # Convert the bar shap_V plot image to a base64-encoded string
    shap_V_plot_buffer = io.BytesIO()
    shap_V_plot_image.save(shap_V_plot_buffer, format='PNG')
    shap_V_plot_base64 = base64.b64encode(shap_V_plot_buffer.getvalue()).decode('utf-8')

      
   # Convert the bar shap_S plot image to a base64-encoded string
    # shap_S_plot_buffer = io.BytesIO()
    # shap_S_plot_image.save(shap_S_plot_buffer, format='PNG')
    # shap_S_plot_base64 = base64.b64encode(shap_S_plot_buffer.getvalue()).decode('utf-8')

#     response_data = {
   
#     "shap_V_plot_base64": shap_V_plot_base64,
#     "prediction": prediction
# }
# # Assuming you have the JSON response data in a dictionary called 'response_data'
#     json_response = json.dumps(response_data, ensure_ascii=False)  # Convert dictionary to JSON string

# # Calculate the size of the JSON response in bytes
#     response_size_in_bytes = sys.getsizeof(json_response)

# # Print the size in bytes
#     print(f"Size of the JSON response: {response_size_in_bytes} bytes")

#     # Compress the JSON response using gzip
#     compressed_response = gzip.compress(json_response.encode('utf-8'))

# # Calculate the size of the compressed JSON response in bytes
#     compressed_response_size_in_bytes = sys.getsizeof(compressed_response)
#     print(f"Size of the compressed JSON response: {compressed_response_size_in_bytes} bytes")

    return  { "shap_V_plot_base64": shap_V_plot_base64}
    # "explaination": explaination, "shap_S_plot_base64": shap_S_plot_base64, "prediction":prediction }
# app.mount("/", StaticFiles(directory="."))

class ClassLabel(BaseModel):
     imagetype: str
@app.post("/explain_occlusion/image")
async def explain_api(file: UploadFile = File(...), base:ClassLabel = Depends(),mlModel:UploadFile = File(...), ImageFileBytes: bytes = File(...)):
     extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
     # input = base.model_dump()
     input = base.dict()
     var = input['imagetype']
     classNum = get_ClassName(var)
     ####################### FIND CLASS ########################
     if not extension:
         return "Image must be jpg or png format!"
     label_number = get_ClassName(var)
#     var1 = 4
     image =read_imagefile(ImageFileBytes)
     explanation, Occlus_Image = OcclusionExplainer.explain_occlusion(image,classNum)

     image_array = np.array(explanation, dtype=np.uint8)
     image = Image.fromarray(image_array)

    # Convert the image to a base64-encoded string
    #  image_buffer = io.BytesIO()
    #  image.save(image_buffer, format='JPEG')
    #  Occ_image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
     
    #  image_buffer.close() 

     
     Occ_buffer = io.BytesIO()
     Occlus_Image.save(Occ_buffer, format='PNG')
     Occ_GradCam_base64 = base64.b64encode(Occ_buffer.getvalue()).decode('utf-8')

    #  image2 = Image.fromarray(explanation.astype('uint8'))
    #  image2 = image2.resize((300, 300))  # Resize if needed

# # Convert the image to a base64-encoded string
#      buffer = io.BytesIO()
#      image2.save(buffer, format='JPEG')  # Save as PNG for lossless compression
#      image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

     occluding_size = 100
     occluding_pixel = 1
     occluding_stride = 10
     image = Image.open(io.BytesIO(ImageFileBytes))
     image = image.resize((300, 300))
        

     OCCE_plot_image = explain_occ_sensitivity_raw(occluding_size,occluding_pixel,occluding_stride,image)

     
     OCCE_plot_buffer = io.BytesIO()
     OCCE_plot_image.save(OCCE_plot_buffer, format='PNG')
     Occ_image_base64 = base64.b64encode(OCCE_plot_buffer.getvalue()).decode('utf-8')


     return {"Occ_image_base64": Occ_image_base64,"Occ_GradCam_base64":Occ_GradCam_base64} 
    #  , image_base64

app.mount("/", StaticFiles(directory="."))

#print(app.openapi())
#schema = app.openapi()

with open('openapi.json', 'w') as f:
  json.dump(app.openapi(), f)

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0", port=8090)
