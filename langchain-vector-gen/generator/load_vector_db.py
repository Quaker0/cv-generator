from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

cv_loader = TextLoader("files/cv.txt")
cover_loader = TextLoader("files/cover_letter.txt")
documents = cv_loader.load() + cover_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(docs, embedding)
vectorstore.save_local("faiss_index")
