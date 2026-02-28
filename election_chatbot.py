
from ollama import Client
import json
import chromadb
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

client = chromadb.PersistentClient()
remote_client = Client(host=f"http://localhost:11434")
collection = client.get_or_create_collection(name="articles_demo")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=20, separators=["."]
)
with open("counter.txt","r") as f:
    s=f.read().strip()
    count = int(s)


print("Reading election.jsonl and generating embeddings...")
with open("election.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i < count:
            continue
        count+=1
        article = json.loads(line)
        content = article["content"]
        sentences = text_splitter.split_text(content)
        print(f"Processing article {i}: {article.get('title', 'No Title')}")

        # 1. Loop through and add individual sentences
        for j, each_sentence in enumerate(sentences):
            response = remote_client.embed(
                model="nomic-embed-text", 
                input=f"search_document: {each_sentence}"
            )
            embedding = response["embeddings"][0]
            collection.add(
                ids=[f"article_{i}_sentence_{j}"],
                embeddings=[embedding],
                documents=[each_sentence],
                metadatas=[{"title": article["title"]}],
            )

        # 2. Add the full article content
        # response = remote_client.embed(
        #     model="nomic-embed-text", 
        #     input=f"search_document: {content}"
        # )
        # embedding = response["embeddings"][0]

        # collection.add(
        #     ids=[f"article_{i}"],
        #     embeddings=[embedding],
        #     documents=[content],
        #     metadatas=[{"title": article["title"]}],
        # )

print("Database built successfully!")

# --- Query Section ---
with open ("counter.txt", 'w') as f:
    f.write(str(count))
while True:
    query=input("how mayy i help you?")
    if query=="break":
        break
    #query = "are there any predicted hindrance for upcoming election ?"
    query_embed = remote_client.embed(
        model="nomic-embed-text", 
        input=f"query: {query}"
    )["embeddings"][0]

    results = collection.query(query_embeddings=[query_embed], n_results=1)

#print(f"\nQuestion: {query}")
# Using results["metadatas"][0][0] and results["documents"][0][0] to access the first match
#print(f'\nTitle : {results["metadatas"][0][0]["title"]} \n{results["documents"][0][0]}')


    context='\n'.join(results["documents"][0])  # Combine all retrieved documents into a single context string
    prompt = f"""You are a helpful assistant. Answer the question based on the context provided. Use the information in the context to form your answer. If context does not have enough information just say "I don't know"

    Context: {context}

    Question: {query}

    Answer:"""
    response = remote_client.generate(
            model="gemma3:4b",
            prompt=prompt,
            options={
                "temperature": 0.1
            }
        )

    print(prompt)
    answer = response['response']

    print(answer)