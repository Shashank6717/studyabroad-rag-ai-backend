import os
from dotenv import load_dotenv

# 1. Supabase
from supabase import create_client

# 2. Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# 3. LCEL
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 4. HuggingFace Endpoint LLM
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

# ---------- SUPABASE ----------
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)

# ---------- EMBEDDINGS ----------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------- HUGGINGFACE LLM (Qwen) ----------
llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-8B",
    task="text-generation",
    max_new_tokens=250,
    temperature=0.2,
)

llm = ChatHuggingFace(llm=llm_endpoint)

# ---------- PROMPT ----------
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a study abroad assistant.
Use ONLY the provided context.
If the answer is not in the context, say:
"I don't know based on the provided documents."
"""),

    ("human", """
Context:
{context}

Question: {question}
""")
])

# ---------- LCEL RAG CHAIN ----------
rag_chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# ------------ TEST FUNCTION -------------
def run_rag(question, country="USA"):
    print("\n🔍 QUESTION:", question)

    # Embed question
    q_emb = embeddings.embed_query(question)

    # Supabase vector search
    res = supabase.rpc("match_documents", {
        "query_embedding": q_emb,
        "match_count": 4,
        "country_filter": country
    }).execute()

    chunks = [r["content"] for r in res.data]
    print("📄 Retrieved:", len(chunks))

    if not chunks:
        print("❌ No relevant chunks found.")
        return

    context = "\n\n".join(chunks)

    # LLM RAG answer
    answer = rag_chain.invoke({
        "context": context,
        "question": question
    })

    print("\n🧠 ANSWER:\n", answer)


# ------------ RUN TEST -------------
if __name__ == "__main__":
    run_rag("What are the visa requirements for USA?")
