import base64
import io
import requests
from PIL import Image as PILImage
from langchain_core.documents import Document
from langchain.chat_models import ChatOpenAI

def image_to_base64(image_obj):
    if hasattr(image_obj, "data"):
        image_bytes = image_obj.data
    elif hasattr(image_obj, "image"):
        image_bytes = image_obj.image
    else:
        raise ValueError("Unsupported image object")

    image = PILImage.open(io.BytesIO(image_bytes))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def summarize_all_images(image_list):
    if not image_list:
        return [], []

    img_base64s = []
    summaries = []

    for image in image_list:
        try:
            b64 = image_to_base64(image)
            img_base64s.append(b64)

            model = ChatOpenAI(model="gpt-4-vision-preview", temperature=0)
            response = model.invoke([
                {"type": "text", "text": "Summarize the content of this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ])
            summaries.append(response.content)
        except Exception as e:
            img_base64s.append("ERROR")
            summaries.append(f"Error summarizing image: {str(e)}")

    return img_base64s, summaries
