from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

embeddings=OllamaEmbeddings(model="mxbai-embed-large:latest")
df=pd.read_csv("./realistic_restaurant_reviews.csv")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)


if add_documents:
    docs=[]
    ids=[]
    for it,row in df.iterrows():
        doc=Document(page_content=row['Title']+"  "+row['Review'], metadata={"Date":row['Date'], "Rating":row['Rating'] },id=str(it))
        docs.append(doc)
        ids.append(str(it))

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=docs, ids=ids)

retriever=vector_store.as_retriever(
    search_kwargs={"k":4}
)
