import io
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import uvicorn

from otracking.models import PeopleAnalytics

app = FastAPI()

@app.get('/')
async def read_root():
    return {"message": "Api People Analitycs - BlueLabs"}


@app.post('/process-video', status_code=200)
async def parse_request(camera_location:str, period_time:str, file: UploadFile):
    contents = await file.read()
    model = PeopleAnalytics(camera_location, period_time)
    response = model.process_video(contents, False)

    return response


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port= 8000)