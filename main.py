from langchain.agents import load_tools
from langchain.memory import ConversationEntityMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory
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
        conn = psycopg2.connect(self.conn_url)
        return execute_query(conn, query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SQLExecutor does not support async")


PERSIST_DIR = "docs-db"
EMBEDDING_FUNC = OpenAIEmbeddings()
# other option is gpt-3.5-turbo
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


def initialize_agent_with_memory():
    llm = ChatOpenAI(temperature=0, model_name=MODEL_NAME)

    docs_store = get_chroma_docs_store() 
    docs = RetrievalQA.from_chain_type(llm=llm, retriever=docs_store.as_retriever())
    sql_executor = SQLExecutor(conn_url="postgresql://root@localhost:26257/defaultdb?sslmode=disable")

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
you can and following their instructions. In cases where you make a change that you'll have to remember down the line,
return that change in the response. An example is SQL DDL. You should record these statements or the resulting schema so it can
be used down the line. You have access to the following tools:"""
    suffix = """Begin!"

    Summarized history:
    {history}

    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools, 
        prefix=prefix, 
        suffix=suffix, 
        input_variables=["input", "history", "agent_scratchpad"]
    )

    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=500)

    llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)


def run_agent_with_memory():
    agent = initialize_agent_with_memory()

    p = """
I want to store information about my ping pong games.
I want to store who played the games, who won, and what the score was.
Figure out how to store this in the database and set up the database as such.
"""
    agent.run(input=p)

    print(agent.memory.load_memory_variables({}))

    p = """
Peyton and Lasse played ping pong. Lasse won 21-15. Record this information.
"""
    agent.run(input=p)

    print(agent.memory.load_memory_variables({}))

def dedup_games():
    agent = initialize_agent_with_memory()

    p = """
    Delete all tables from the database.
"""
    agent.run(input=p)

    print(agent.memory.load_memory_variables({}))

run_agent_with_memory()
