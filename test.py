from pathway.xpacks.llm.embedders import OpenAIEmbedder
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.vector_store import VectorStoreClient, VectorStoreServer

PATHWAY_PORT = 8765

client = VectorStoreClient(
    host="127.0.0.1",
    port=PATHWAY_PORT,
)

# Example query to test the setup
query = "Who is Manoj?"
response = client.query(query)
print(response)
