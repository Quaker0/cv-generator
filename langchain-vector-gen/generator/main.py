from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from stream_handlers import PrintAndCaptureHandler, FileLoggingHandler

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

vectorstore = FAISS.load_local(
    "faiss_index", embeddings=embedding, allow_dangerous_deserialization=True
)

response_handler = PrintAndCaptureHandler()
llm = ChatOllama(model="llama3:8b", streaming=True, callbacks=[response_handler])

with open("files/job.txt", "r", encoding="utf-8") as f:
    job_text = f.read().strip()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an AI assistant. Don't use hyphens, semi-colons or overly complicated language. "
                "Use the user's CV and cover letter (retrieved as {context}) to tailor the answer to this job. Value recent and more senior job positions in the CV higher. "
                "STRICT GROUNDING: Only use facts present in {context}. If a detail is not in {context}, don't mention it at all. Do not infer or guess. "
                "The job description below is provided only to guide what to emphasize; do NOT introduce any new facts from it unless those facts also appear in {context}.\n\n"
                f"Job description to target:\n{job_text}"
            ),
        ),
        (
            "human",
            (
                "Use the retrieved context below to craft the answer.\n\n"
                "Retrieved context:\n{context}\n\n"
                "Question: {question}"
            ),
        ),
    ]
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

file_handler = FileLoggingHandler("context.txt")
compressor_llm = ChatOllama(model="llama3:8b", streaming=True, callbacks=[file_handler])
extractor = LLMChainExtractor.from_llm(llm=compressor_llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=extractor, base_retriever=retriever
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever,
    chain_type_kwargs={"prompt": prompt},
)
response = qa_chain.invoke(
    {
        "query": "Write a very short and direct description of why I am good for the job, max 100 words. Don't saystraight out that I want the job but lean on that I would be a good fit."
    }
)
