from fastapi import FastAPI, WebSocket, Request, Response
import httpx
import json
import base64
import asyncio
import websockets
import logging
from typing import Dict, Optional
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define settings using Pydantic
class Settings(BaseSettings):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    n8n_webhook_url: str = os.getenv("N8N_WEBHOOK_URL", "")
    repl_public_url: str = os.getenv("REPL_PUBLIC_URL", "")
    port: int = int(os.getenv("PORT", "8000"))
    voice: str = "shimmer"

settings = Settings()

# Session store to track ongoing calls
sessions: Dict[str, Dict] = {}

# Initialize Pinecone client lazily
_pinecone_client = None
_pinecone_index = None

def get_pinecone_index():
    global _pinecone_client, _pinecone_index
    if _pinecone_client is None:
        try:
            _pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
            _pinecone_index = _pinecone_client.Index("voice-agent")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            return None
    return _pinecone_index

# Initialize FastAPI app
app = FastAPI()

# Helper function to send requests to N8N
async def send_to_n8n(route: str, number: str, payload: Optional[Dict] = None) -> Dict:
    async with httpx.AsyncClient() as client:
        data = {"route": route, "number": number}
        if payload:
            data["payload"] = payload
        response = await client.post(settings.n8n_webhook_url, json=data)
        return response.json()

# Helper function to query Pinecone
async def query_pinecone(question: str) -> str:
    index = get_pinecone_index()
    if index is None:
        logger.warning("Pinecone index not available, returning mock response")
        return f"Here's what I found about '{question}' in our knowledge base..."
    
    try:
        # TODO: Implement actual Pinecone query
        return f"Here's what I found about '{question}' in our knowledge base..."
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        return "I'm sorry, I couldn't access the knowledge base at the moment."

# Incoming call endpoint
@app.post("/incoming-call")
async def incoming_call(request: Request) -> Response:
    form_data = await request.form()
    caller = form_data.get("From")
    logger.info(f"Incoming call from {caller}")

    # Fetch chat history from N8N (route 1)
    history = await send_to_n8n("1", caller)
    first_message = history.get("firstMessage", "Hello! How can I help you today?")

    # Create a session
    session_id = f"session_{caller}"
    sessions[session_id] = {"caller": caller, "first_message": first_message}

    # Generate TwiML to connect to media stream
    twiml = f"""
    <Response>
        <Connect>
            <Stream url="wss://{settings.repl_public_url}/media-stream">
                <Parameter name="sessionId" value="{session_id}"/>
            </Stream>
        </Connect>
    </Response>
    """
    return Response(content=twiml, media_type="text/xml")

# Media stream WebSocket handler
@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        # Read initial JSON handshake to get sessionId
        handshake = await websocket.receive_json()
        session_id = handshake.get("sessionId")
        if not session_id or session_id not in sessions:
            logger.error(f"Invalid or missing sessionId: {session_id}")
            await websocket.close()
            return

        session = sessions[session_id]
        caller = session["caller"]

        # Connect to OpenAI's realtime API
        openai_ws = await websockets.connect(
            "wss://api.openai.com/v1/realtime",
            extra_headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
        )

        # Send session.update to OpenAI
        session_update = {
            "type": "session.update",
            "audio": {
                "input": {"format": "g711_ulaw", "server_vad": True},
                "output": {"voice": settings.voice}
            },
            "system": {
                "instructions": "You are a helpful voice assistant. Answer questions and help schedule meetings."
            },
            "modalities": ["text", "audio"],
            "temperature": 0.7,
            "functions": [
                {
                    "name": "question_and_answer",
                    "description": "Answer a question using the knowledge base.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "The question to answer"}
                        },
                        "required": ["question"]
                    }
                },
                {
                    "name": "schedule_meeting",
                    "description": "Schedule a meeting.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of the attendee"},
                            "email": {"type": "string", "description": "Email of the attendee"},
                            "purpose": {"type": "string", "description": "Purpose of the meeting"},
                            "datetime": {"type": "string", "description": "Date and time of the meeting"},
                            "location": {"type": "string", "description": "Location of the meeting"}
                        },
                        "required": ["name", "email", "purpose", "datetime", "location"]
                    }
                }
            ]
        }
        await openai_ws.send(json.dumps(session_update))

        # Bidirectional relay loop
        async def relay_twilio_to_openai():
            while True:
                try:
                    data = await websocket.receive_json()
                    if data.get("event") == "media":
                        media_payload = data.get("media", {}).get("payload")
                        if media_payload:
                            # Forward audio to OpenAI
                            await openai_ws.send(json.dumps({
                                "type": "input.audio",
                                "payload": media_payload
                            }))
                except Exception as e:
                    logger.error(f"Error relaying Twilio to OpenAI: {e}")
                    break

        async def relay_openai_to_twilio():
            while True:
                try:
                    data = await openai_ws.recv()
                    event = json.loads(data)
                    if "response" in event:
                        if "audio" in event["response"] and "delta" in event["response"]["audio"]:
                            # Forward audio to Twilio
                            await websocket.send_json({
                                "event": "media",
                                "media": {"payload": event["response"]["audio"]["delta"]}
                            })
                        elif "function_call_arguments" in event["response"] and event["response"]["function_call_arguments"].get("done"):
                            # Handle function call
                            func_name = event["response"]["function_call_arguments"].get("name")
                            args = event["response"]["function_call_arguments"].get("arguments", {})
                            if func_name == "question_and_answer":
                                # Use Pinecone for Q&A
                                answer = await query_pinecone(args.get("question", ""))
                                await openai_ws.send(json.dumps({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "function_call_output",
                                        "role": "system",
                                        "output": answer
                                    }
                                }))
                                await openai_ws.send(json.dumps({"type": "response.create"}))
                            elif func_name == "schedule_meeting":
                                # Use N8N for booking
                                result = await send_to_n8n("3", caller, args)
                                await openai_ws.send(json.dumps({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "function_call_output",
                                        "role": "system",
                                        "output": result.get("message", "Meeting scheduled successfully.")
                                    }
                                }))
                                await openai_ws.send(json.dumps({"type": "response.create"}))
                except Exception as e:
                    logger.error(f"Error relaying OpenAI to Twilio: {e}")
                    break

        # Run both relay tasks concurrently
        await asyncio.gather(relay_twilio_to_openai(), relay_openai_to_twilio())

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()
        if 'openai_ws' in locals():
            await openai_ws.close()

# Root endpoint for health check
@app.get("/")
async def root():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.port) 