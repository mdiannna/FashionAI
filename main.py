from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from image_similarity.imagesimilarity import ImageSimilarity
from fastapi.templating import Jinja2Templates

app = FastAPI()
img_sim = ImageSimilarity()

# Mount the static files directory for serving CSS and JS
app.mount("/static", StaticFiles(directory="static"), name="static")
# Mount the directory containing your uploaded images
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request:Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    with open(f"uploads/{file.filename}", "wb") as buffer:
        buffer.write(await file.read())
    return {"Filename uploaded": file.filename, "TODO" : "integrate with similar clothes suggestion microservice"}


@app.post("/get-similar-img", response_class=HTMLResponse)
async def get_similar_img(request:Request, file: UploadFile = File(...)):
    # upload the new images in a separate folder, not in the "database"
    with open(f"new_imgs/{file.filename}", "wb") as buffer:
        buffer.write(await file.read())
    
    most_similar = img_sim.get_most_similar("new_imgs/"+file.filename, 'uploads/')

    print("most similar:", most_similar)

    return templates.TemplateResponse(
        request=request, name="index.html", context={"id": 1, "top_image": most_similar[1], "similar_images": most_similar[2]}
    )