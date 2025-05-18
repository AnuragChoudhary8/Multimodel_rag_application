from unstructured.partition.pdf import partition_pdf

def extract_elements_from_pdf(path: str):
    return partition_pdf(filename=path)
