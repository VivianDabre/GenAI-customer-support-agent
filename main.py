import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict
from langgraph.graph import StateGraph 
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import Image, display
from groq import Groq


load_dotenv() # Loads environment variables from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment variables")
client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama3-70b-8192" 

# FastAPI setup
app = FastAPI()

# Define input data structure
class CustomerQuery(BaseModel):
    name: str
    age: int
    gender: str
    email: str
    subject: str
    query: str

# LangGraph schema
class InputState(Dict):
    name: str
    age: int
    gender: str
    email: str
    subject: str
    query: str
    prepared_data: Optional[Dict]
    sentiment: Optional[str]
    ticket_type: Optional[str]
    escalated: Optional[str]
    technical_response: Optional[str]
    general_query: Optional[str]
    resolution: Optional[str]
    steps: List[str]

# Define LangGraph nodes
#Data preparation node - fetches the data from frontend
def data_prep_node(state: InputState) -> InputState:
    state["prepared_data"] = {"detail": state["query"]}
    state["steps"].append("**Fetched the data.**")
    return state

# Sentiment analysis node - LLM agent to get the query from the user and provide the sentiment and reasoning
def sentiment_analysis_node(state: InputState) -> InputState:
    print("Running sentiment_analysis_node")
    query_text = state["prepared_data"]["detail"]
    # call Groq LLM 
    prompt = (
        "You are a sentiment analysis assistant for customer support queries.\n"
        "Please provide the sentiment of the following customer query as one word: positive, neutral, or negative.\n\n"
        "Also provide a brief reason for your sentiment classification.\n\n"
        f"Query: {query_text}\n\n"
        "Respond in this exact JSON format:\n"
        '{\n  "sentiment": "positive",\n  "reason": "The query expresses gratitude."\n}'
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You analyse sentiment in customer support queries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=60
    )

    raw_output = response.choices[0].message.content.strip()
    import json
    try:
        parsed = json.loads(raw_output)
        sentiment = parsed.get("sentiment", "neutral").lower() # gets the sentiment, if not default is neutral
        reason = parsed.get("reason", "No reason provided.") #get the reason, if not default is no reason provided
    except Exception:
        # fallback if response is not JSON
        sentiment = "neutral"
        reason = raw_output

    if sentiment not in {"positive", "neutral", "negative"}:
        sentiment = "neutral"

    state["sentiment"] = sentiment
    state["steps"].append(f"**Sentiment detected:** **{sentiment}**")
    state["steps"].append(f"**Reason:** **{reason}**")

    return state

#Ticket Type node - LLM agent to get the query and subject from the user and clarify it into pre-defined categories.
def ticket_type_node(state: InputState) -> InputState:
    print("Running ticket_type_node")
    subject_text = state["subject"]
    query_text = state["prepared_data"]["detail"]

    prompt = (
        "You are a customer support ticket classifier.\n"
        "Based on the subject and customer query below, classify the ticket into one of these categories:\n"
        "Technical, Billing & accounts, Product/general Inquiry.\n\n"
        f"Subject: {subject_text}\n"
        f"Query: {query_text}\n\n"
        "Respond with only the category name."
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You classify support tickets into predefined categories."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=10
    )

    ticket_type = response.choices[0].message.content.strip()

    valid_types = {"technical", "billing & accounts", "product/general inquiry"}
    ticket_type_lower = ticket_type.lower()
    ticket_type = ticket_type_lower if ticket_type_lower in valid_types else "product/general inquiry"

    state["ticket_type"] = ticket_type
    state["steps"].append(f"**Ticket categorised as:** **{ticket_type}**")

    return state
def technical_node(state: InputState) -> InputState:
    print("Running technical_node")
    ticket_type_technical = state.get("ticket_type")
    query_text = state["prepared_data"]["detail"]
    prompt = (
        "You are a customer support ticket classifier.\n"
        "Based on the subject and customer query below, Provide a technical support response to the query in short 2 sentences\n"
        f"Query: {query_text}\n"
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Provide a technical support response to the query"},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=40
    )
    
    technical_response = response.choices[0].message.content.strip()
    state["technical_response"] = technical_response
    state["steps"].append(f"**Response:** **{technical_response}**")

def general_query_node(state: InputState) -> InputState:
    print("Running general_inquery_node")
    ticket_type_general = state.get("ticket_type")
    query_text = state["prepared_data"]["detail"]
    prompt = (
        "You are a customer support ticket classifier.\n"
        "Based on the subject and customer query below, Provide a general support response to the query in short 2 sentences\n"
        f"Query: {query_text}\n"
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Provide a general support response to the query"},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=40
    )
    
    general_query = response.choices[0].message.content.strip()
    state["general_response"] = general_query
    state["steps"].append(f"**Response:** **{gneral_response}**")

# Escalate node - Idenfies keywords from query and label it as escalated. LLM can be used for keyword analysis or intent analysis
def escalate_node(state: InputState) -> InputState:
    print("Running escalate_node")
    query_text = state["prepared_data"]["detail"].lower()
    sentiment = state.get("sentiment", "neutral")
    ticket_type = state.get("ticket_type", "product/general inquiry").lower()

    escalation_keywords = [
        "frustrated", "angry", "upset", "not happy", "complain", "bad service",
        "unacceptable", "escalate", "worst", "problem not solved", "disappointed"
    ]

    triggered_by_keywords = any(keyword in query_text for keyword in escalation_keywords)
    triggered_by_sentiment = sentiment == "negative"
    triggered_by_ticket_type = ticket_type == "billing & accounts" and sentiment != "positive"
    should_escalate = triggered_by_keywords or triggered_by_sentiment or triggered_by_ticket_type
    state["escalated"] = should_escalate

    if triggered_by_keywords:
        state["steps"].append("**Escalation triggered: keyword detected.**")
    elif triggered_by_sentiment:
        state["steps"].append("**Escalation triggered: negative sentiment.**")
    elif triggered_by_ticket_type:
        state["steps"].append("**Escalation triggered: billing issue with non-positive sentiment.**")
    else:
        state["steps"].append("**No escalation needed.**")

    return state



def resolution_node(state: InputState) -> InputState:
    print("Running resolution_node")
    if state.get("escalated", False):
        resolution = "**Escalated to human support agent for further assistance.**"
        state["steps"].append("**Resolution: Ticket escalated to human agent.**")
    else:
        resolution = "**Automated resolution provided based on query and ticket type.**"
        state["steps"].append("**Resolution: Handled automatically.**")

    state["resolution"] = resolution
    return state

def route_query(state: InputState) -> InputState:
    print("Running route_query_node")
    if state["ticket_type"] == "technical":
        return "technical_node"
    elif state["ticket_type"] == "Product/general Inquiry":
        return "general_query_node"
    else:
        return "resolution_node"

# Build LangGraph
graph = StateGraph(InputState)
graph.add_node("data_prep_node", data_prep_node)
graph.add_node("sentiment_analysis_node", sentiment_analysis_node)
graph.add_node("ticket_type_node", ticket_type_node)
graph.add_node("technical_node", technical_node)
graph.add_node("general_query_node", general_query_node)
graph.add_node("escalate_node", escalate_node)
graph.add_node("resolution_node", resolution_node)

graph.set_entry_point("data_prep_node")
graph.add_edge("data_prep_node", "sentiment_analysis_node")
graph.add_edge("sentiment_analysis_node", "ticket_type_node")

graph.add_conditional_edges(
    "ticket_type_node",
    route_query,
    {
        "technical_node", technical_node,
        "general_query_node", general_query_node
    })
graph.add_edge("technical_node", "escalate_node")
graph.add_edge("general_query_node", "escalate_node")
graph.add_edge("escalate_node", "resolution_node")
graph.set_finish_point("resolution_node")

app_graph = graph.compile()

# FastAPI endpoint
@app.post("/fetch-data")
async def fetch_data(data: CustomerQuery):
    input_data = data.dict()
    input_data["steps"] = []  # Initialize steps list

    try:
        result = app_graph.invoke(input_data)

        return {
            "message": "Processed successfully using LangGraph",
            "steps": result.get("steps", []),
            "final_resolution": result.get("resolution", "N/A")
        }

    except Exception as e:
        return {
            "error": str(e),
            "steps": input_data.get("steps", []),
            "details": "LangGraph execution failed."
        }
