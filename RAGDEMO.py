
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Step 1: Prepare document
document = """
Artificial Intelligence (AI) is transforming the way we live and work.
Machine learning, a subset of AI, enables computers to learn from data.
Deep learning uses neural networks inspired by the human brain.

Natural Language Processing (NLP) helps computers understand and process
human language. Applications include translation, chatbots, and text analysis.

Computer Vision enables machines to process images and videos.
Applications include facial recognition, autonomous vehicles,
and medical image analysis.

Reinforcement Learning is a type of machine learning where agents
learn by interacting with an environment. They receive rewards for
good actions and penalties for bad ones. Applications include
game playing, robotics, and autonomous systems.
"""

print("Step 1: Document loaded")
print("Length:", len(document), "characters")

# Step 2: Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = text_splitter.split_text(document)

print("\nStep 2: Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:\n{chunk}")

# Step 3: Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\nStep 3: Embedding model loaded",embeddings.model_name)
print("This model will convert text chunks into numerical vectors")

# Step 4: Create vector store
vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Step 4: Vector store created")
print("Embedding dimension:", len(embeddings.embed_query("test")))

# Step 5: Similarity search
query = "What is reinforcement learning?"

result = vector_store.similarity_search(query, k=2)

print("\nStep 5: Similarity search results")

for i, doc in enumerate(result):
    print(f"\nResult {i + 1}:")
    print(doc.page_content)

# Step 6: Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=api_key
)

# Step 7: Create RAG pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(
        search_kwargs={"k": 2}
    )
)

# Step 8: Ask questions
questions = [
    "What is reinforcement learning and how does it work?",
    "What are the main applications of computer vision?",
    "How is NLP used in real-world applications?"
]

print("\nStep 8: Asking questions")

for question in questions:
    try:
        response = qa_chain.invoke(question)
        print("\nQuestion:", question)
        print("Answer:", response)
    except Exception as e:
        print("Error:", str(e))
