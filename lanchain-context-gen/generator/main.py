from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from .stream_handlers import PrintAndCaptureHandler

response_handler = PrintAndCaptureHandler()
llm = ChatOllama(model="llama3:8b", streaming=True, callbacks=[response_handler])

with open("files/job.txt", "r", encoding="utf-8") as f:
    job_text = f.read().strip()

with open("files/cv.txt", "r", encoding="utf-8") as f:
    cv_text = f.read().strip()

with open("files/cover_letter.txt", "r", encoding="utf-8") as f:
    cover_letter_text = f.read().strip()

context = "CV:\n" + cv_text + "\n\n" + "Cover Letter:\n" + cover_letter_text

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an AI assistant. Don't use hyphens, semi-colons or overly complicated language. "
                "Use the user's CV and cover letter (provided as {context}) to tailor the answer to this job. Value recent and more senior job positions in the CV higher. "
                "STRICT GROUNDING: Only use facts present in {context}. If a detail is not in {context}, don't mention it at all. Do not infer or guess. "
                "The job description below is provided only to guide what to emphasize; do NOT introduce any new facts from it unless those facts also appear in {context} and do not copy any sentence word for word from it.\n\n"
                "The job description that the text should target is provided as {job_desc}."
            ),
        ),
        (
            "human",
            (
                "Use the provided context below to craft the answer.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Job description: {job_desc}"
            ),
        ),
    ]
)

chain = prompt | llm

_ = chain.invoke(
    {
        "context": context,
        "question": "Write a short and direct description of why I am good for the job, max 100 words. Don't say that I want the job just that I would be a good fit and then mention the name of the company I am applying for. Mention my start-up experience. Use a more direct way of writing with fewer descriptive words.",
        "job_desc": job_text,
    }
)
