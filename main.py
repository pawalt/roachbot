from langchain.agents import load_tools
from langchain.memory import ConversationEntityMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory, CombinedMemory
from langchain.agents import initialize_agent, Tool, ZeroShotAgent, AgentExecutor
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.tools import BaseTool
from langchain.schema import messages_to_dict, messages_from_dict
from langchain.schema import BaseMemory

from pydantic import BaseModel
from typing import Callable, List, Dict, Any
import json

import psycopg2
from psycopg2 import Error

import chromadb
from dotenv import load_dotenv

load_dotenv()

def execute_query(connection, query):
    try:
        cursor = connection.cursor()
        cursor.execute(query)

        if cursor.description is None:
            connection.commit()
            return "Query executed successfully."

        # Fetch and return results as a string
        result = ""
        rows = cursor.fetchall()
        for row in rows:
            result += str(row) + "\n"
        if not result.strip():
            return "Empty result"
        return result
    except (Exception, Error) as error:
        return f"Error while executing query: {error}"
    finally:
        if cursor:
            cursor.close()

class SQLExecutor(BaseTool):
    name = "SQL Executor"
    description = "useful for when you need to execute SQL queries"

    conn_url = "this will fail"

    def _run(self, query: str) -> str:
        """Use the tool."""
        with psycopg2.connect(self.conn_url) as conn:
            return execute_query(conn, query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SQLExecutor does not support async")

class SchemaMemory(BaseMemory, BaseModel):
    """Memory class for storing information about database schema."""
    conn_url = "this will fail"

    # Define key to pass information about entities into prompt.
    memory_key: str = "schema"
        
    def clear(self):
        return

    @property
    def memory_variables(self) -> List[str]:
        """Define the variables we are providing to the prompt."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Load the memory variables, in this case the db schema."""

        with psycopg2.connect(self.conn_url) as conn:
             tables = execute_query(conn, "SHOW CREATE ALL TABLES")
             return {
                self.memory_key: tables,
             }

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Noop since this memory only tracks schema"""
        return


PERSIST_DIR = "docs-db"
SUMMARY_FILE = "memory_summary.json"
EMBEDDING_FUNC = OpenAIEmbeddings()
# options:
# - gpt-4 (slow and expensive but smart)
# - gpt-3.5-turbo (less smart but fast and cheap. super bad at using plugins)
MODEL_NAME="gpt-4"


def get_chroma_docs_store():
    client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=PERSIST_DIR,
        anonymized_telemetry=False
    )
    return Chroma( 
        embedding_function=EMBEDDING_FUNC,
        collection_name="cockroach-docs",
        persist_directory=PERSIST_DIR,
        client_settings=client_settings,
    )


def create_db():
    docs_path = "./v22.2"
    doc_loader = DirectoryLoader(docs_path, glob="**/*.md", loader_cls=TextLoader)
    docs = doc_loader.load()
    text_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=20)
    split_docs = text_splitter.split_documents(docs)

    docs_store = get_chroma_docs_store()
    docs_store.add_documents(documents=split_docs, embeddings=EMBEDDING_FUNC)
    docs_store.persist()

def get_summarized_memory():
    try:
        with open(SUMMARY_FILE, 'r') as f:
            data = json.load(f)
            assert "msg_buf" in data, "no message buf"
            assert "moving_sum" in data, "no moving summary found"
            data["msg_buf"] = messages_from_dict(data["msg_buf"])
            return data
    except Exception as e:
        print(f"Could not read history file. Returning default valuies. {e}")
        return {
            "msg_buf": [],
            "moving_sum": "",
        }

def initialize_agent_with_memory():
    llm = ChatOpenAI(temperature=0.5, model_name=MODEL_NAME)

    docs_store = get_chroma_docs_store() 
    docs = RetrievalQA.from_chain_type(llm=llm, retriever=docs_store.as_retriever())
    conn_url = "postgresql://root@localhost:26257/defaultdb?sslmode=disable"
    sql_executor = SQLExecutor(conn_url=conn_url)
    schema_memory = SchemaMemory( 
        conn_url=conn_url,
    )

    tools = [
        Tool(
            name="CockroachDB Docs QA System",
            func=docs.run,
            description="useful when you need to get information about how to use cockroachdb. input should be a fully formed question.",
        ),
        Tool(
            name="SQL executor plugin",
            func=sql_executor.run,
            description="""useful for when you need to run SQL against a persistent CockroachDB instance.
This takes a well-formed SQL query as input and returns the result of that query."""
        ),
    ]

    prefix = """Have a conversation with a human, answering the following questions as best
you can and following their instructions. If you perform an action, respond precisely with
a description of the action you performed. If the user did not provide sufficient information
for you to perform your task and know you're doing the right thing, respond with a request
for more detailed information.
You have access to the following tools:"""
    suffix = """Begin!

Summarized history:
{history}

Current database table schema:
{schema}

Question: {input}
{agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools, 
        prefix=prefix, 
        suffix=suffix, 
        input_variables=["input", "agent_scratchpad", "history", "schema"]
    )

    old_save = ConversationSummaryBufferMemory.save_context
    ConversationSummaryBufferMemory.save_context = lambda s, i, o: save_memory_with_oldfun(s, i, o, old_save)

    stored_mem = get_summarized_memory()
    convo_memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=500,
        input_key="input",
    )
    convo_memory.moving_summary_buffer = stored_mem["moving_sum"]
    convo_memory.chat_memory.messages = stored_mem["msg_buf"]
    combined_memory = CombinedMemory(memories=[convo_memory, schema_memory])

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=combined_memory)

def save_memory_with_oldfun(self: ConversationSummaryBufferMemory, inputs: Dict[str, Any], outputs: Dict[str, str], oldfun) -> None:
    oldfun(self, inputs, outputs)

    msg_buf = messages_to_dict(self.buffer)
    moving_sum = self.moving_summary_buffer

    with open(SUMMARY_FILE, "w") as f:
        json.dump({
            "msg_buf": msg_buf,
            "moving_sum": moving_sum,
        }, f)

def run_agent_with_memory(runner: Callable[[AgentExecutor], None]):
    agent = initialize_agent_with_memory()

    runner(agent)

def play_ping_pong(agent: AgentExecutor) -> None:
    p = """
I want to store information about my ping pong games.
I want to store who played the games, who won, and what the score was.
Figure out how to store this in the database and set up the database as such.
"""
    agent.run(input=p)

    p = """
Peyton and Lasse played ping pong. Lasse won 21-15. Record this information.
"""
    agent.run(input=p)

    p = """
Peyton and Lasse played ping pong. Peyton won 21-19. Record this information.
"""
    agent.run(input=p)

    p = """
Peyton and Lasse played ping pong. Lasse won 21-20. Record this information.
"""
    agent.run(input=p)

    p = """
Who has won more ping pong games?
"""
    agent.run(input=p)

    p = """
What's the total number of points scored by all ping pong players?
"""
    agent.run(input=p)

def delete_games(agent: AgentExecutor) -> None:
    p = """
    Delete all tables from the database.
"""
    agent.run(input=p)

def repl(agent: AgentExecutor):
    try:
        while True:
            print()
            user_input = input("Human $ ").strip()

            if user_input == "done":
                break

            agent.run(user_input)
    except EOFError:
        print("done")

run_agent_with_memory(repl)
