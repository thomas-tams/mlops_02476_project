from fastapi import FastAPI
from http import HTTPStatus
from enum import Enum
from pydantic import BaseModel
import re
import numpy as np
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
import cv2

app = FastAPI()


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


@app.get("/restric_items/{item_id}")
def read_item(item_id: ItemEnum):
    return {"item_id": item_id}


class EmailItem(BaseModel):
    email: str
    domain_match: str


@app.get("/text_model/")
def contains_email(data: EmailItem):
    regex = r"@(hotmail|gmail)\b"
    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_acceptable_domain": re.fullmatch(regex, data.domain_match) is not None,
    }
    return response


@app.post("/cv_model/")
async def cv_model(image_file: UploadFile = File(...), resize_height: int = 28, resize_width: int = 28):
    # Read the uploaded image file
    contents = await image_file.read()
    np_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image file"}

    # Resize the image
    resized_img = cv2.resize(img, (resize_width, resize_height))

    # Save the resized image to disk
    output_path = "image_resize.jpg"
    cv2.imwrite(output_path, resized_img)

    return FileResponse(
        output_path, media_type="image/jpeg", headers={"Content-Disposition": f"attachment; filename={output_path}"}
    )


# POST method
database = {"username": [], "password": []}


@app.post("/login/")
def login(username: str, password: str):
    username_db = database["username"]
    password_db = database["password"]
    if username not in username_db and password not in password_db:
        with open("database.csv", "a") as file:
            file.write(f"{username},{password}\n")
        username_db.append(username)
        password_db.append(password)
    return "login saved"
