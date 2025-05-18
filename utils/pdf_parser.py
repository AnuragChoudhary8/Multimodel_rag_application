import fitz  # PyMuPDF

def extract_elements_from_pdf(path):
    doc = fitz.open(path)
    texts = []
    images = []

    for i, page in enumerate(doc):
        texts.append(page.get_text())

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            with open(f"extracted_data/image_page{i+1}_{img_index}.{ext}", "wb") as f:
                f.write(image_bytes)

    return texts
