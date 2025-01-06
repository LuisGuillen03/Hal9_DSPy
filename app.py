import hal9 as h9
import fitz
from io import BytesIO
import requests
import dspy
from openai import OpenAI
import numpy as np

# Initialize OpenAI client with Hal9 proxy
openai_client = OpenAI(
    base_url="https://api.hal9.com/proxy/server=https://api.openai.com/v1/",
    api_key="hal9"
)

# Generate embeddings for given texts
def generate_embeddings(texts):
    embeddings = []
    for text in texts:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)

# Split text into overlapping chunks
def split_text(text, n_words=300, overlap=0):
    if overlap >= n_words:
        raise ValueError("Overlap must be smaller than the number of words per chunk.")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunk = words[start:start + n_words]
        chunks.append(" ".join(chunk))
        start += n_words - overlap

    return chunks

# Input prompt from Hal9
prompt = h9.input()

# Configure the language model
lm = dspy.LM('openai/gpt-4-turbo', api_key='hal9', api_base='https://api.hal9.com/proxy/server=https://api.openai.com/v1/')
dspy.configure(lm=lm)

# RAG module definition
class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question, search):
        context = search(question).passages
        return self.respond(context=context, question=question)

chunks = messages = h9.load("chunks", [])

if h9.is_url(prompt):
    # Download and process PDF content
    response = requests.get(prompt)
    pdf_document = fitz.open(stream=BytesIO(response.content))
    text = "".join([pdf_document[page_num].get_text() for page_num in range(len(pdf_document))])
    new_chunks = split_text(text, 3000, 300)

    # Avoid duplicating existing chunks
    combined_chunks = chunks + [chunk for chunk in new_chunks if chunk not in chunks]
    h9.save("chunks", combined_chunks, hidden=True)
    print("The PDF has been successfully processed. I am ready to answer questions about this document.")
else:
    if len(chunks) == 0:
        print("I am an agent designed to chat with PDF documents. Please upload at least one PDF to enable me to generate an answer.")
    else:
        embedder = dspy.Embedder(generate_embeddings)
        search = dspy.retrievers.Embeddings(embedder=embedder, corpus=chunks, k=5)

        rag = RAG()
        response = rag(question=prompt, search=search)

        print(response.response)