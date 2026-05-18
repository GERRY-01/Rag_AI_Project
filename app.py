from flask import Flask, request, render_template

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from groq import Groq
from secret import api_key


app = Flask(__name__)

client = Groq(api_key=api_key)

# Load PDF document
loader = PyPDFLoader("Final Report document (2).pdf")
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector database
vector_store = FAISS.from_documents(chunks, embeddings)

# Retriever
retriever = vector_store.as_retriever()


@app.route("/", methods=["GET", "POST"])
def index():

    answer = ""

    if request.method == "POST":

        query = request.form["query"]

        # Retrieve relevant chunks
        results = retriever.invoke(query)

        context = "\n".join(
            [result.page_content for result in results]
        )

        prompt = f"""
        You are an AI assistant.
        Use the context below to answer the question.

        Context:
        {context}

        Question:
        {query}

        Answer clearly and based only on the context.
        """

        # Generate response
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_completion_tokens=1024,
            top_p=1,
            stream=True
        )

        # Store streamed answer
        for chunk in completion:

            content = chunk.choices[0].delta.content

            if content:
                answer += content

    return render_template(
        "index.html",
        answer=answer
    )


if __name__ == "__main__":
    app.run()