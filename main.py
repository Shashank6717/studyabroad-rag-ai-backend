from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from utils.supabase_client import get_supabase
from rag_pipeline import rag_chain, embeddings
import bcrypt
from utils.jwt_utils import create_access_token,verify_token
from fastapi import Header
from utils.gemini import generate_chat_title
import os


app = FastAPI()

# -------------- CORS (Allow Next.js) ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://study-abroad-rag-ai-frontend-ek2i.vercel.app",
        "https://study-abroad-rag-ai-frontend-ek2i-fay6su70g.vercel.app",
        "http://localhost:3000",  # for local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------- Supabase Client ---------------------
supabase = get_supabase()

@app.post("/api/chat/query")
def chat_query(payload: dict = Body(...),
    authorization: str = Header(None)):
    if not authorization:
        return {"error": "Authorization header missing"}

    token = authorization.replace("Bearer ", "")
    user_data = verify_token(token)

    if not user_data:
        return {"error": "Invalid or expired token"}

    user_id = user_data["user_id"]
    question = payload.get("question")

    if not question:
        return {"answer": "Please enter a question."}

    # 1️⃣ Embed question
    q_emb = embeddings.embed_query(question)

    # 2️⃣ Vector search via RPC
    result = supabase.rpc("match_documentsssss", {
        "query_embedding": q_emb,
        "match_count": 4,
    }).execute()

    if not result.data:
        return {"answer": "No relevant content found in the documents."}

    # 3️⃣ Build context
    chunks = [r["content"] for r in result.data]
    context = "\n\n".join(chunks)

    # 4️⃣ RAG answer
    answer = rag_chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": answer,
        "sources": chunks
    }

@app.post("/api/auth/signup")
def signup_user(payload: dict = Body(...)):
    email = payload.get("email")
    password = payload.get("password")

    if not email or not password:
        return {"error": "Email and password are required"}

    # Hash password
    salt = bcrypt.gensalt()
    password_hash = bcrypt.hashpw(password.encode(), salt).decode()

    # Check if user exists
    existing = supabase.table("users").select("*").eq("email", email).execute()
    if existing.data:
        return {"error": "User already exists"}

    # Insert new user
    new_user = supabase.table("users").insert({
        "email": email,
        "password_hash": password_hash
    }).execute()

    user_id = new_user.data[0]["id"]

    # Create JWT
    token = create_access_token({"user_id": user_id, "email": email})

    return {
        "message": "Signup successful",
        "token": token,
        "user": {
            "id": user_id,
            "email": email
        }
    }


@app.post("/api/auth/login")
def login_user(payload: dict = Body(...)):
    email = payload.get("email")
    password = payload.get("password")

    if not email or not password:
        return {"error": "Email and password are required"}

    # Fetch user from DB
    result = supabase.table("users").select("*").eq("email", email).execute()

    if not result.data:
        return {"error": "Invalid credentials"}

    user = result.data[0]

    # Verify password
    if not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        return {"error": "Invalid credentials"}

    # Create JWT
    token = create_access_token({"user_id": user["id"], "email": email})

    return {
        "message": "Login successful",
        "token": token,
        "user": {
            "id": user["id"],
            "email": email
        }
    }

@app.post("/api/chat/rename/{chat_id}")
def rename_chat(payload: dict = Body(...), authorization: str = Header(None),chat_id: str = None):
    if not authorization:
        return {"error": "Authorization header missing"}

    token = authorization.replace("Bearer ", "")
    user_data = verify_token(token)
    if not user_data:
        return {"error": "Invalid or expired token"}
    
    new_title = payload.get("new_title")

    if not chat_id or not new_title:
        return {"error": "chat_id and new_title are required"}

    # Update chat title
    supabase.table("chats").update({"title": new_title}).eq("id", chat_id).execute()

    return {"message": "Chat renamed successfully"}

@app.post("/api/chat/create")
def create_chat(authorization: str = Header(None)):
    if not authorization:
        return {"error": "Authorization header missing"}

    token = authorization.replace("Bearer ", "")
    user_data = verify_token(token)
    if not user_data:
        return {"error": "Invalid or expired token"}

    user_id = user_data["user_id"]

    # Create empty chat session
    new_chat = supabase.table("chats").insert({
        "user_id": user_id,
        "title": "New Chat"
    }).execute()

    chat_id = new_chat.data[0]["id"]

    return {"chat_id": chat_id, "message": "Chat created"}


@app.get("/api/chat/list")
def chat_list(authorization: str = Header(None)):
    if not authorization:
        return {"error": "Authorization header missing"}

    token = authorization.replace("Bearer ", "")
    user_data = verify_token(token)
    if not user_data:
        return {"error": "Invalid or expired token"}

    user_id = user_data["user_id"]

    result = supabase.table("chats") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .execute()

    return {"chats": result.data}

@app.get("/api/chat/messages/{chat_id}")
def get_chat_messages(chat_id: str, authorization: str = Header(None)):
    if not authorization:
        return {"error": "Authorization header missing"}

    token = authorization.replace("Bearer ", "")
    user_data = verify_token(token)
    if not user_data:
        return {"error": "Invalid or expired token"}

    result = supabase.table("chat_messages") \
        .select("*") \
        .eq("chat_id", chat_id) \
        .order("created_at") \
        .execute()

    return {"messages": result.data}


@app.post("/api/chat/querywithrag")
def chat_query(
    payload: dict = Body(...),
    authorization: str = Header(None)
):
    if not authorization:
        return {"error": "Authorization header missing"}

    token = authorization.replace("Bearer ", "")
    user_data = verify_token(token)
    if not user_data:
        return {"error": "Invalid or expired token"}

    user_id = user_data["user_id"]

    question = payload.get("question")
    # country = payload.get("country", "USA")
    # Fetch chat title

    chat_id = payload.get("chat_id")

    if not chat_id:
        return {"error": "chat_id is required"}

    chat_info = supabase.table("chats").select("title").eq("id", chat_id).execute()
    current_title = chat_info.data[0]["title"]


    if current_title == "New Chat":
        title = generate_chat_title(question)
        supabase.table("chats").update({"title": title}).eq("id", chat_id).execute()
    if not question:
        return {"error": "Please enter a question."}

    # ------------------- FETCH CONVERSATION HISTORY ---------------------
    history_res = supabase.table("chat_messages") \
        .select("*") \
        .eq("chat_id", chat_id) \
        .order("created_at", desc=False) \
        .limit(10) \
        .execute()

    history_messages = []
    for msg in history_res.data:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_messages.append(f"{role}: {msg['content']}")

    history_text = "\n".join(history_messages)

    # ------------------- EMBEDDING ---------------------
    q_emb = embeddings.embed_query(question)

    # ------------------- VECTOR SEARCH ---------------------
    result = supabase.rpc("match_documentsssss", {
        "query_embedding": q_emb,
        "match_count": 4,
        # "country_filter": country
    }).execute()

    if not result.data:
        return {"answer": "No relevant content found in the documents."}

    chunks = [r["content"] for r in result.data]
    context = "\n\n".join(chunks)

    # ------------------- RAG + MEMORY ---------------------
    answer = rag_chain.invoke({
        "history": history_text,
        "context": context,
        "question": question
    })

    # ------------------- SAVE MESSAGES ---------------------
    # Save user's message
    supabase.table("chat_messages").insert({
        "chat_id": chat_id,
        "role": "user",
        "content": question
    }).execute()

    # Save assistant's answer
    supabase.table("chat_messages").insert({
        "chat_id": chat_id,
        "role": "assistant",
        "content": answer
    }).execute()

    return {
        "answer": answer,
        "sources": chunks
    }



@app.delete("/api/chat/delete/{chat_id}")
def delete_chat(chat_id: str, authorization: str = Header(None)):
    if not authorization:
        return {"error": "Authorization header missing"}

    token = authorization.replace("Bearer ", "")
    user_data = verify_token(token)
    if not user_data:
        return {"error": "Invalid or expired token"}
    
    user_id = user_data["user_id"]

    # Delete chat and its messages
    supabase.table("chats").delete().eq("id", chat_id).execute()
    supabase.table("chat_messages").delete().eq("chat_id", chat_id).execute()

    return {"message": "Chat deleted successfully"}


if __name__ == "__main__":
    import uvicorn

    PORT = int(os.environ.get("PORT", 4000,10000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False
    )
