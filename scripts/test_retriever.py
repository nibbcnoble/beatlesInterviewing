from ..app.rag.retriever import get_retriever

r = get_retriever()
results = r.retrieve("What did John think about Liverpool?", beatle="john", top_k=3)

for item in results:
    print(item["score"], item["chunk"]["chunk_id"], item["chunk"]["speaker"])
    print(item["chunk"]["text"][:300])
    print()
