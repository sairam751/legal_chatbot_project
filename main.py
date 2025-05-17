# legal_advisor_chatbot/main.py
from langchain.agents import initialize_agent, Tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import SystemMessage
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain_core.chains import RetrievalQA, LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool


# Environment setup
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HUGGING_FACE_KEY"] = os.getenv("HUGGING_FACE_KEY")

# Callback for streaming
callback = StdOutCallbackHandler()

# Load and split document
loader = PyPDFLoader("sample_contract.pdf")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Embeddings and Vector Store
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# LLM Setup
#llm = ChatOpenAI(temperature=0, streaming=True, callbacks=[callback])
llm = ChatGroq(model="qwen-qwq-32b", temperature=0, streaming=True, callbacks=[callback])

# Prompt Template
qa_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a legal advisor AI. Use the following context to answer the legal question:
    
    Context:
    {context}

    Question:
    {question}

    Answer in a clear, concise, and formal manner.
    """
)

# Retrieval QA Chain
"""qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)"""
question_answer_chain = create_stuff_documents_chain(llm, qa_template)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define tools
@tool
def math_tool(query: str) -> str:
    """Performs math calculations from legal texts or penalty clauses."""
    return str(eval(query))

@tool
def document_query_tool(question: str) -> str:
    """Query over uploaded legal documents."""
    result = question_answer_chain.invoke({"context": docs, "question": question})
    return result

# Tools for agent
tools = [
    Tool(name="DocumentQuery", func=document_query_tool, description="Use for questions about uploaded legal documents."),
    Tool(name="MathTool", func=math_tool, description="Use for performing calculations."),
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent="chat-conversational-react-description",
    verbose=True
)

# Run agent
if __name__ == "__main__":
    print("Welcome to Legal Advisor Chatbot")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = agent.run(input=query)
        print("Bot:", response)
