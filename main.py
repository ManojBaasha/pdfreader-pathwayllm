import logging
import sys
import time
import getpass
import os
import pathway as pw
from pathway.xpacks.llm.llms import OpenAIChat, prompt_chat_single_qa

#  REST Connector config.
HTTP_HOST = os.environ.get("PATHWAY_REST_CONNECTOR_HOST", "127.0.0.1")
HTTP_PORT = os.environ.get("PATHWAY_REST_CONNECTOR_PORT", "8080")

#  LLM model parameters
#  For OPENAI API
API_KEY = ""
#  Specific model from OpenAI. You can also use gpt-3.5-turbo for faster responses.
MODEL_LOCATOR = "gpt-4"
# Controls the stochasticity of the openai model output.
TEMPERATURE = 0.0
# Max completion tokens
MAX_TOKENS = 50


class QueryInputSchema(pw.Schema):
    query: str
    user: str


query, response_writer = pw.io.http.rest_connector(
    host=HTTP_HOST,
    port=int(HTTP_PORT),
    schema=QueryInputSchema,
    autocommit_duration_ms=50,
)

model = OpenAIChat(
    api_key=API_KEY,
    model=MODEL_LOCATOR,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    retry_strategy=pw.udfs.FixedDelayRetryStrategy(),
    cache_strategy=pw.udfs.DefaultCache(),
)

response = query.select(
    query_id=pw.this.id, result=model(prompt_chat_single_qa(pw.this.query))
)

response_writer(response)
pw.run()

