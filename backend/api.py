from fastapi import FastAPI, UploadFile, File, Form
from ai_worker import moderate_screenshot

app = FastAPI()

@app.post("/api/v1/ai/moderate-screenshot")
async def moderate(
    image: UploadFile = File(...),
    categories: str = Form(""),
    transaction_id: str = Form(None),
    device_id: str = Form(None),
):
    img_bytes = await image.read()
    cats = [c.strip() for c in categories.split(",") if c.strip()]
    report = moderate_screenshot(img_bytes, cats)
    report.update(transaction_id=transaction_id, device_id=device_id)
    return report
