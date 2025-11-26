import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ------------------ EMBEDDINGS ------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------ LLM (Qwen3-8B) ------------------
llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-8B",
    task="text-generation",
    max_new_tokens=250,
    temperature=0.2,
)

llm = ChatHuggingFace(llm=llm_endpoint)

# ------------------ PROMPT TEMPLATE ------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful and knowledgeable study abroad assistant specializing in visa requirements, 
educational opportunities, and travel information for students planning to study internationally.

Your role is to:
- Provide accurate, clear, and concise answers based ONLY on the provided context documents
- Reference previous conversation context when relevant to maintain continuity
- Be friendly, professional, and supportive
- If information is not available in the provided context, clearly state: "I don't have that information in the provided documents. Please consult official sources or contact the relevant embassy.answer the question based on your knowledge but mention that you don't have that information in the provided documents."

Guidelines:
- Use the context documents as your primary source of information
- Consider the conversation history to provide contextually relevant answers
- Keep answers focused and avoid unnecessary repetition
- If asked about something not in the context, politely indicate the limitation but answer the question based on your knowledge but mention that you don't have that information in the provided documents.

Previous Conversation:
{chat_history}

Relevant Context from Documents:
{context}"""),
    ("human", """Current Question: {question}

Please provide a helpful answer based on the context above and the conversation history."""),
])

# ------------------ LCEL RAG CHAIN ------------------
rag_chain = (
    {
        "context": RunnablePassthrough(),
        "question": RunnablePassthrough(),
        "chat_history": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
