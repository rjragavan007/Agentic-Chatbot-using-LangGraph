##################################IMPORT##################################
import os
import requests
import math
import sqlite3
import pandas as pd
import logging

from datetime import datetime
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict,Annotated,Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage,SystemMessage,BaseMessage,AIMessage
from langchain_core.runnables import RunnableConfig

# TOOLS #
# tools_condition => Inbuilt condition to decide whether to use tool or not
# ToolNode => Inbuilt node for having various tools
from simpleeval import simple_eval
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import interrupt,Command
# TOOLS #

# RAG #
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# RAG #
##################################IMPORT##################################

# LLM
load_dotenv()
# llm=ChatGroq(model="openai/gpt-oss-20b",temperature=0.5,groq_api_key=os.getenv('GROQ_API_KEY'))
# llm=ChatGroq(model="openai/gpt-oss-120b",temperature=0.5,groq_api_key=os.getenv('GROQ_API_KEY'))
llm=ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite',temperature=0.5,google_api_key=os.getenv("GEMINI_API_KEY"))
# llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash',temperature=0.5,google_api_key=os.getenv("GEMINI_API_KEY"))
df = pd.read_csv("valid_symbols.csv", skiprows=2)
valid_symbols = set(df.iloc[:, 0].dropna().astype(str).str.strip().str.upper())
##################################TOOLS##########################################
def isvalid(symbol):
    if not symbol or not isinstance(symbol, str):
        return False
    symbol = symbol.strip().upper()
    return symbol in valid_symbols

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.
    Args:
        symbol: Stock ticker symbol, for example AAPL.
        quantity: Number of shares to buy.
    Returns the order status (success or cancelled).
    """
    # This pauses the graph and returns control to the caller
    if quantity>0 and isvalid(symbol):
        decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")
    else:
        return {"status": "cancelled",
                "message":"Incorrect symbol or Invalid quantity mentioned",            
                "symbol": symbol,
                "quantity": quantity,}
    if isinstance(decision, str) and decision.lower()=="yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,}
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,}

search_tool=TavilySearch(max_results=3,topic="general",search_depth="advanced")

@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Available functions: sqrt, abs, round, min, max. Constants: pi, e.
    Example: 2 + 2, sqrt(16), 10 * 5, round(pi, 2)
    """
    try:
        functions = {
            "sqrt": math.sqrt,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max
        }
        names = {"pi": math.pi, "e": math.e}

        # simple_eval only evaluates what is whitelisted above. 
        result = simple_eval(expression, functions=functions, names=names)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def get_stock_price(symbol:str):
    """Uses the input symbol and determines the stock market price
    of the stock using alpha vantage's API key (for e.g AAPL)"""
    try:
        url=f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol.strip().upper()}&apikey={os.getenv('ALPHAVANTAGE_API_KEY')}"
        response=requests.get(url,timeout=10)

        # Catches HTTPerror i.e, status code based errors
        response.raise_for_status()
        # response.json()=>return a valid json if not raises value error
        data=response.json()

        if "Error Message" in data:
            return f"Error arised: {data['Error Message']}"
        if "Note" in data:
            return f"Rate limit arised: {data['Note']}"
        if "Information" in data:
            return f"Subscription not enough: {data['Information']}"
        if not data.get('Global Quote'):
            return "The Symbol passed didnt return any value."
        return data
    except requests.exceptions.RequestException:
        return "Could not fetch the stock price"
    except ValueError:
        return "Could not fetch the stock price"
    
@tool
def get_weather(city:str)->dict:
    """Uses the input location and determines the weather of the given
    location using weather stack API (for e.g New Delhi)"""
    try:
        url=f"http://api.weatherstack.com/current?access_key={os.getenv('WEATHERSTACK_API')}&query={city}"
        response=requests.get(url,timeout=10)

        # Catches HTTPerror i.e, status code based errors
        response.raise_for_status()
        # response.json()=>return a valid json if not raises value error
        data=response.json()
        if "success" in data:
            return f"Cannot fetch current weather due to :\n{data['error']['info']}"
        return response.json()
    except requests.exceptions.RequestException:
        return "Could not fetch current weather"
    except ValueError:
        return "Could not fetch current weather"

@tool
def add_notes(note:str,runtime:ToolRuntime):
    """
    Receives one argument as input note(str) and stores it in the notes.db with thread_id aswell
    and returns a confirmation that data is inserted
    The purpose of thread_id is to not confuse among the various conversational threads"""
    try:
        # 1.Create a connector which connects to specified db if exists
        # If db not exists then it is created
        con=sqlite3.connect(database="notes.db")

        # 2.Create a cursor to execute SQL queries
        cursor=con.cursor()

        #3.Create table if it not exists
        cursor.execute("""
                        CREATE TABLE IF NOT EXISTS note(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        thread_id TEXT,
                        note TEXT NOT NULL,
                        created_at TEXT NOT NULL)""")

        # Get current date and time
        now = datetime.now()
        thread_id=runtime.config.get('configurable', {}).get('thread_id')

        if thread_id:
            # 4. Insert data using safely parameterized queries (prevents SQL injection)
            user_data = (thread_id,note,f"{now.day}-{now.month}-{now.year}",)
            cursor.execute("INSERT INTO note (thread_id,note,created_at) VALUES (?,?,?)", user_data)
        else:
            return "Thread id missing..."
        
        # 5.Commit changes for write operations (CREATE, INSERT, UPDATE, DELETE)
        con.commit()
        print("Data inserted successfully.")
        return "Data inserted successfully."
    
    except sqlite3.Error as error:
        print(f"Error while connecting to sqlite: {error}")
        return f"Data insertion Unsuccessful because of the error: {error}"

    finally:
        # 6. Always close the cursor and connection when done
        if 'cursor' in locals():
            cursor.close()
        if 'con' in locals():
            con.close()
        print("\nSQLite connection is closed.")

@tool
def list_notes(runtime:ToolRuntime)->str:
    """returns all notes as single string with all ids and respective values in it. 
    If there are none, it returns a clear “no notes yet” message."""
    try:
        # 1.Create a connector which connects to specified db if exists
        # If db not exists then it is created
        con=sqlite3.connect(database="notes.db")

        # 2.Create a cursor to execute SQL queries
        cursor=con.cursor()

        #3.Create table if it not exists
        cursor.execute("""
                        CREATE TABLE IF NOT EXISTS note(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        thread_id TEXT,
                        note TEXT NOT NULL,
                        created_at TEXT NOT NULL)""")

        thread_id=runtime.config.get('configurable', {}).get('thread_id')

        if thread_id:
        #4.Retrieve all data to list
            cursor.execute("SELECT id,note,created_at FROM note WHERE thread_id=? ORDER BY id ASC",(thread_id,))
        else:
            return "No thread id found..."

        # Fetch all rows from the result of the query
        rows = cursor.fetchall()
        if rows==[]:
            return "no notes yet!"
        else:
            data=""
            for row in rows:
                r=(str(row[0]),row[1],row[2])
                data=data+("|").join(r)
                data=data+"\n"
            return data

    except sqlite3.Error as error:
        print(f"Error while connecting to sqlite: {error}")
        return f"Error arised , Can not display notes: {error}"
    
    finally:
        # 6. Always close the cursor and connection when done
        if 'cursor' in locals():
            cursor.close()
        if 'con' in locals():
            con.close()
        print("\nSQLite connection is closed.")
        
@tool
def delete_note(note_id:int,runtime:ToolRuntime)->str:
    """Use list_notes first to find the id and deletes by id.
      If the id doesn’t exist, it says so rather than pretending it worked."""

    try:
        # 1.Create a connector which connects to specified db if exists
        # If db not exists then it is created
        con=sqlite3.connect(database="notes.db")

        # 2.Create a cursor to execute SQL queries
        cursor=con.cursor()

        thread_id=runtime.config.get('configurable', {}).get('thread_id')
        if thread_id:
        #3.Delete desired rows
            cursor.execute("""DELETE FROM note WHERE id=? AND thread_id=?""",(note_id,thread_id,))
        else:
            return "Thread id not found!"

        # Commit changes in database
        con.commit()

        # 4.check rowcount to determine whether row is deleted or not
        if cursor.rowcount==1:
            return f"Row with id {note_id} is successfully deleted!"
        else:
            return f"No row with id {note_id} exists"

    except sqlite3.Error as error:
        print(f"Error while deleting: {error}")
        return f"Error arised while deleting: {error}"
    
    finally:
        # 5. Always close the cursor and connection when done
        if 'cursor' in locals():
            cursor.close()
        if 'con' in locals():
            con.close()
        print("\nSQLite connection is closed.")

@tool
def exchange_currency(from_curr:str,to_curr:str)->str:
    """
    The purpose of this tool is to convert from one currency to another currency using the
    conversion rates from ExchangeRate APIs and perform the math using the calculator tool.
    1.Find the 3 Letter code for the specific country from_curr and pass it to from_curr argument in url.Similarly for the to_curr argument also.
    For example INR for INDIA and USD for United States of America

    2.Check the returned json value and find the specific conversion rate. 

    3.Use the calculator tool and the conversion rate returned form API call to perform the math operation.
    For example, 20 dollars in Indian ruppees => 20,USA,INDIA => 20,USD,INR =>20*95.01 = 1900.2
    """
    try:
        url=f"https://v6.exchangerate-api.com/v6/{os.getenv('EXCHANGERATE_API_KEY')}/pair/{from_curr.strip().upper()}/{to_curr.strip().upper()}"
        # use timeout limit so it doesnt take so long
        response=requests.get(url,timeout=20)
        # determine the status code of response. for ex, 200,401,402
        response.raise_for_status()
        data=response.json()

        if data.get('result')!='success':
            return f"Error arised: {data.get('error-type','unknown-error')}"
        return f"1 {data['base_code']}={data['conversion_rate']} {data['target_code']}"
    
    except requests.exceptions.RequestException:
        return f"Could not reach the exchange rate service"

    except ValueError:
        return f"Unexpected response from the exchange rate service"
##################################TOOLS##########################################

##################################RAG TOOL############################################
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Streamlit helper function
def ingest_rag_documents(file_path, file_name,thread_id):
    DB_PATH = os.path.join("faiss_db",str(thread_id),os.path.splitext(file_name)[0])
    os.makedirs(DB_PATH, exist_ok=True)
    # 1.LOAD
    loader=PyPDFLoader(file_path)
    docs=loader.load()
    # 2.SPLIT
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=splitter.split_documents(docs)
    # 3.EMBEDDINGS=>GLOBAL
    # 4.VECTORSTORE
    vector_store=FAISS.from_documents(chunks,embeddings)
    vector_store.save_local(DB_PATH)

def get_retriever(file_name,thread_id:str):
    DB_PATH = os.path.join("faiss_db",thread_id,os.path.splitext(file_name)[0])
    # embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #5.RETRIEVER
    vector_store=FAISS.load_local(folder_path=DB_PATH,
                                  embeddings=embeddings,
                                  allow_dangerous_deserialization=True)
    retriever=vector_store.as_retriever(search_type='similarity',search_kwargs={"k":4})
    return retriever

def get_available_documents(thread_id:str):
    db_root = os.path.join("faiss_db",thread_id)

    if not os.path.isdir(db_root):
        return []

    return [
        name for name in os.listdir(db_root)
        if os.path.isfile(os.path.join(db_root, name, "index.faiss"))
        and os.path.isfile(os.path.join(db_root, name, "index.pkl"))
    ]

@tool
def rag_tool(query:str,file_name:str,runtime:ToolRuntime)->str:
    """
    This is a rag tool which uses retriever to retrieve relevant data from the data uploaded.
    Use this tool when the user asks for questions or facts from the uploaded document.
    
    Args: 
    1.query: The information on which retrieval needs to be performed using uploaded data 
    2.file_name: This helps to identify which file's retriever needs to be returned
    """
    thread_id=runtime.config.get('configurable', {}).get('thread_id')
    if thread_id:
        available_files=get_available_documents(thread_id)
    else:
        return "Thread id missing..."
    file_name = os.path.splitext(file_name)[0]

    if file_name not in available_files:
        return f"Document not found. Available documents: {available_files}"
    
    try:
        retriever=get_retriever(file_name,thread_id)
        documents=retriever.invoke(query)
    except FileNotFoundError:
        logging.exception("FAISS index not found")
        return "The requested document index does not exist."
    except Exception:
        logging.exception("RAG retrieval failed")
        return "Unable to retrieve information from the requested document."
    if not documents:
        return "No relevant information based on the input query"
    formatted_documents=[]
    for index,document in enumerate(documents,start=1):
        source=document.metadata.get('source','unknown source')
        page=document.metadata.get('page','unknown page')
        formatted_documents.append(
            f"Document:{index}\n"
            f"Source:{source}\n"
            f"page:{page}\n"
            f"Content:{document.page_content}\n"
        )
    return "\n\n".join(formatted_documents)

##################################RAG TOOL############################################
# Bind tools
tools=[rag_tool,search_tool,calculator,add_notes,list_notes,delete_note,get_stock_price,get_weather,exchange_currency,purchase_stock]
llm_with_tools=llm.bind_tools(tools)

# State
class ChatState(TypedDict):
    # add_messages=>Inbuilt Reducer for Message
    messages:Annotated[list[BaseMessage],add_messages]

# Node-1
def chat_node(state: ChatState,config:RunnableConfig):
    """LLM node that can answer directly or call an appropriate tool."""
    thread_id = config.get('configurable', {}).get('thread_id')
    if thread_id:
        available_files = get_available_documents(thread_id)
    else:
        return {"messages": [AIMessage(content="Thread id is missing")]}
    system_message = SystemMessage(
    content=(
        "You are a helpful Agentic Chatbot equipped with several tools. "
        "Your primary responsibility is to answer users accurately, clearly, "
        "and efficiently.\n\n"

        "GENERAL TOOL USAGE PRINCIPLES:\n"
        "- Answer directly using your existing knowledge whenever possible.\n"
        "- Do NOT call tools for every user query.\n"
        "- Use a tool only when it is genuinely necessary to fulfil the "
        "user's request accurately.\n"
        "- Do not call multiple tools unnecessarily.\n"
        "- Select only the most relevant tool for the task.\n"
        "- Never call a tool just because it is available.\n"
        "- If a tool is not required, answer directly without calling one.\n"
        "- After receiving a tool result, interpret it and provide a clear "
        "final answer to the user.\n\n"

        "AVAILABLE TOOLS AND WHEN TO USE THEM:\n"
        "- search_tool: Use when the user requests current information, "
        "recent events, or information that requires web research.\n"
        "- calculator: Use when calculations are required. "
        "Answer simple conceptual mathematical questions directly.\n"
        "- get_stock_price: Use when the user asks for current stock prices "
        "or related live market data.\n"
        "- get_weather: Use when the user asks for current weather or forecasts.\n"
        "- exchange_currency: Use when the user requests currency conversion "
        "using current exchange rates.\n"
        "- purchase_stock: Use ONLY when the user explicitly requests "
        "a stock purchase and provides or confirms the required stock symbol "
        "and quantity. Never use it for stock research, price enquiries, "
        "recommendations, or general stock discussions. "
        "Human approval is required before completing the purchase.\n"
        "- add_notes: Use when the user explicitly asks to save or add a note.\n"
        "- list_notes: Use when the user asks to view their saved notes.\n"
        "- delete_note: Use when the user explicitly asks to delete a note.\n"
        "- rag_tool: Use when the user asks questions that require information "
        "from an uploaded PDF document.\n\n"

        "PDF USAGE INSTRUCTIONS:\n"
        f"-available files: {available_files}\n"
        "- Use rag_tool when the user's question requires information "
        "from an uploaded PDF.\n"
        "- Pass the exact document name from the available document list "
        "as the file_name argument.\n"
        "- Never invent or modify a filename.\n"
        "- If multiple documents could match the user's request and "
        "the intended document is unclear, ask the user to clarify.\n"
        "- If no relevant document is available, ask the user to upload it.\n"
        "- Base document-specific answers on retrieved content. "
        "Do not invent facts or pretend that information exists in a document "
        "when it was not retrieved.\n\n"

        "IMPORTANT DECISION RULE:\n"
        "Before calling any tool, determine whether the user's request "
        "actually requires external information, an action, a calculation, "
        "or retrieval from an uploaded document. "
        "If it does not, answer directly. "
        "If it does, use only the necessary tool or tools.\n\n"

        "Maintain a natural conversational style. "
        "Ask for clarification when essential information is missing. "
        "Never perform an action that the user has not explicitly requested."))

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Node-2
tool_node=ToolNode(tools)

# Checkpointer and SQL connection
conn=sqlite3.connect(database="chatbot.db",check_same_thread=False)
checkpoint=SqliteSaver(conn)

# graph
graph=StateGraph(ChatState)

# add nodes
graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)

# add edges
graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')

chatbot=graph.compile(checkpointer=checkpoint)

# response=chatbot.invoke({"messages":HumanMessage(content="What is your name? in 5 words")},config={"configurable":{"thread_id":"thread_1"}})
# print(response)

# Streamlit helper function
def get_all_threads():
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT thread_id FROM checkpoints GROUP BY thread_id ORDER BY MAX(checkpoint_id) ASC")
        return [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        # checkpoints table doesn't exist yet (fresh chatbot.db)
        return []
    finally:
        cursor.close()
