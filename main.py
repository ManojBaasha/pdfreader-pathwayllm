import logging
import sys
import time
import getpass
import os
import pathway as pw

from pathway.xpacks.llm.embedders import OpenAIEmbedder
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.vector_store import VectorStoreClient, VectorStoreServer

PATHWAY_PORT = 8765

logging.basicConfig(stream=sys.stderr, level=logging.WARN, force=True)

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

data_sources = []
data_sources.append(
    pw.io.fs.read(
        "./sample_documents",
        format="binary",
        mode="streaming",
        with_metadata=True,
    )
)

# Choose document transformers
text_splitter = TokenCountSplitter()
embedder = OpenAIEmbedder(api_key="")

# The `PathwayVectorServer` is a wrapper over `pathway.xpacks.llm.vector_store` to accept LangChain transformers.
# Feel free to fork it to develop bespoke document processing pipelines.
vector_server = VectorStoreServer(
    *data_sources,
    embedder=embedder,
    splitter=text_splitter,
)
vector_server.run_server(host="127.0.0.1", port=PATHWAY_PORT, threaded=True, with_cache=False)
time.sleep(30)  # Workaround for Colab - messages from threads are not visible unless a cell is running
