from unstructured.partition.pdf import partition_pdf
import pytesseract
import shutil

tesseract_cmd = shutil.which("tesseract")
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
else:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_elements_from_pdf(path: str, output_dir="extracted_data"):
    return partition_pdf(
        filename=path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=output_dir,
        extract_image_block_to_payload=False
    )
