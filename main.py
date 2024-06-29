from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/search/")
async def search(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        # For simplicity, we are returning a static message.
        # You can integrate your actual search logic here.
        return JSONResponse(content={"message": f"Search results for: {query}"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def main():
    return HTMLResponse(open("static/index.html").read())
