from langchain_community.document_loaders import PyPDFLoader

file_path = (
    "Apple_Vision_Pro_Privacy_Overview.pdf"
)
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()

print(pages)