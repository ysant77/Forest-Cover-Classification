from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from segmentation import deforestation_detection

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload", response_class=JSONResponse, tags=["image-processing"])
async def upload_files(
        file_list: str,
        first: UploadFile = File(...), second: UploadFile = File(...)
):
    # Read in raw bytes
    image_1_bytes = await first.read()
    image_2_bytes = await second.read()
    # Get places and dates
    raw_filenames = [elem[:-4] for elem in file_list.split(',')]
    places = [elem.split("_")[1] for elem in raw_filenames]
    dates = [elem.split("_")[2] for elem in raw_filenames]
    # Perform deforestation detection
    results = await deforestation_detection([image_1_bytes, image_2_bytes], filenames=raw_filenames)
    # Return
    response = {
        "status": "success",
        "places": places,
        "dates": dates,
        **results
    }

    return JSONResponse(content=response)


@app.get("/download/analysis", response_class=FileResponse, tags=["report-downloading"])
async def download_analysis():
    return FileResponse(
        path="files/analysis.zip",
        media_type="application/zip"
    )


@app.get("/download/report", response_class=FileResponse, tags=["report-downloading"])
async def download_report():
    return FileResponse(
        path="files/changes.csv",
        media_type="application/octet-stream",
        filename="changes.csv"
    )
