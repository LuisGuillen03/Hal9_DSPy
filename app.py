import hal9 as h9
import fitz
from io import BytesIO
import requests
import dspy
from openai import OpenAI
from custom_dspy import CustomOpenAIVectorizer, Hal9_FaissRM

# Initialize the OpenAI client with the Hal9 proxy
openai_client = OpenAI(
    base_url="https://api.hal9.com/proxy/server=https://api.openai.com/v1/",
    api_key="hal9"
)

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

# DSPy Signature
class GenerateAnswer(dspy.Signature):
    context = dspy.InputField(desc="Context passages or facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Detailed and long answer generated referenced on passages")

# RAG module definition
class RAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return prediction

# Input prompt from Hal9
prompt = h9.input()

# Load document chunks from Hal9 storage
chunks = h9.load("chunks", [])

# Create the vectorizer with the custom OpenAI client
vectorizer = CustomOpenAIVectorizer(openai_client=openai_client)

# Process the input PDF
if h9.is_url(prompt):
    # Download and process PDF content
    response = requests.get(prompt)
    pdf_document = fitz.open(stream=BytesIO(response.content))
    text = "".join([pdf_document[page_num].get_text() for page_num in range(len(pdf_document))])
    new_chunks = split_text(text, 3000, 300)

    # Add only unique chunks to the list
    combined_chunks = list(set(chunks + new_chunks))
    h9.save("chunks", combined_chunks, hidden=True)

    # Initialize FaissRM
    frm = Hal9_FaissRM(document_chunks=combined_chunks, vectorizer=vectorizer, update=True)

    print("The PDF has been successfully processed and the index updated.")
else:
    if len(chunks) == 0:
        print("I am an agent designed to chat with PDF documents. Please upload at least one PDF to enable me to generate an answer.")
    else:
        #Configure DSPy
        lm = dspy.LM('openai/gpt-4-turbo', api_key='hal9', api_base='https://api.hal9.com/proxy/server=https://api.openai.com/v1/')
        frm = Hal9_FaissRM(document_chunks=chunks, vectorizer=vectorizer)
        dspy.configure(lm=lm, rm=frm)

        rag = RAG()
        response = rag(question=prompt)
        print(response.answer)