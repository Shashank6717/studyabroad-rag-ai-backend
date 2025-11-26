# just for scripting purpose

# import os
# from supabase import create_client
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from dotenv import load_dotenv

# load_dotenv()

# supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))

# pdfs = {
#     "USA": "pdfs/usa.pdf",
#     "UK": "pdfs/uk.pdf",
#     "Canada": "pdfs/canada.pdf",
#     "Australia": "pdfs/australia.pdf"
# }

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# for country, path in pdfs.items():
#     print(f"📘 Processing {country}")
#     loader = PyPDFLoader(path)
#     docs = loader.load()
#     chunks = splitter.split_documents(docs)

#     for c in chunks:
#         emb = embeddings.embed_query(c.page_content)
#         supabase.table("documents").insert({
#             "country": country,
#             "content": c.page_content,
#             "embedding": emb
#         }).execute()

# print("🎉 All embeddings stored in Supabase!")
