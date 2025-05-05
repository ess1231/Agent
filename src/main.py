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
    # UltraVox settings
    ultravox_api_key: str = os.getenv("ULTRAVOX_API_KEY", "")
    ultravox_model: str = os.getenv("ULTRAVOX_MODEL", "fixie-ai/ultravox-70B")
    ultravox_voice: str = os.getenv("ULTRAVOX_VOICE", "Tanya-English")
    ultravox_sample_rate: int = int(os.getenv("ULTRAVOX_SAMPLE_RATE", "8000"))
    ultravox_buffer_size: int = int(os.getenv("ULTRAVOX_BUFFER_SIZE", "60"))
    
    # Other settings
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    n8n_webhook_url: str = os.getenv("N8N_WEBHOOK_URL", "")
    ultravox_public_url: str = os.getenv("ULTRAVOX_PUBLIC_URL", "")
    port: int = int(os.getenv("PORT", "8080"))

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
            _pinecone_index = _pinecone_client.Index("voice-agent")  # Change to your actual index name
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            return None
    return _pinecone_index

# Initialize FastAPI app
app = FastAPI()

# Helper function to format phone number
def format_phone_number(number: str) -> str:
    """Format phone number by removing the plus sign and any non-digit characters."""
    return ''.join(filter(str.isdigit, number))

# Helper function to create UltraVox call
async def create_ultravox_call(system_prompt: str, first_message: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.ultravox.ai/api/calls",
            headers={"X-API-Key": settings.ultravox_api_key},
            json={
                "systemPrompt": system_prompt,
                "model": settings.ultravox_model,
                "voice": settings.ultravox_voice,
                "temperature": 0.1,
                "initialMessages": [
                    {"role": "MESSAGE_ROLE_USER", "text": first_message}
                ],
                "medium": {
                    "serverWebSocket": {
                        "inputSampleRate": settings.ultravox_sample_rate,
                        "outputSampleRate": settings.ultravox_sample_rate,
                        "clientBufferSizeMs": settings.ultravox_buffer_size
                    }
                },
                "selectedTools": [
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
        )
        return response.json()["joinUrl"]

# Helper function to send requests to N8N
async def send_to_n8n(route: str, number: str, payload: Optional[Dict] = None) -> Dict:
    async with httpx.AsyncClient() as client:
        data = {"route": route, "number": format_phone_number(number)}
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
        # Use the latest Pinecone search API
        results = index.search(
            namespace="default",  # Change if you use a different namespace
            query={
                "top_k": 3,
                "inputs": {
                    "text": question
                }
            }
        )
        hits = results.get('result', {}).get('hits', [])
        if not hits:
            return "Sorry, I couldn't find anything relevant in the knowledge base."
        # Return the top result's text
        return hits[0]['fields']['chunk_text']
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        return "I'm sorry, I couldn't access the knowledge base at the moment."

# Helper function to summarize transcript
async def summarize_transcript(transcript: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.ultravox.ai/api/calls",
            headers={"X-API-Key": settings.ultravox_api_key},
            json={
                "model": settings.ultravox_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes phone call transcripts into concise bullet points."
                    },
                    {
                        "role": "user",
                        "content": f"Here is the full transcript of the call:\n\n{transcript}\n\nPlease give me a concise bullet-point summary."
                    }
                ],
                "temperature": 0.7
            }
        )
        return response.json()["choices"][0]["message"]["content"]

# Incoming call endpoint
@app.post("/incoming-call")
async def incoming_call(request: Request) -> Response:
    try:
        form_data = await request.form()
        # Try both Caller and From parameters
        caller = form_data.get("From") or form_data.get("Caller")
        if not caller:
            logger.error("No caller number provided")
            return Response(content="<Response><Reject/></Response>", media_type="text/xml")
            
        logger.info(f"Incoming call from {caller}")

        # Create a session
        session_id = f"session_{format_phone_number(caller)}"
        sessions[session_id] = {"caller": caller, "first_message": "Hello! How can I help you today?"}

        # Generate TwiML to connect to media stream
        twiml = f"""
        <Response>
            <Connect>
                <Stream url="wss://{settings.ultravox_public_url}/media-stream">
                    <Parameter name="sessionId" value="{session_id}"/>
                </Stream>
            </Connect>
        </Response>
        """
        return Response(content=twiml, media_type="text/xml")
    except Exception as e:
        logger.error(f"Error handling incoming call: {str(e)}")
        return Response(content="<Response><Reject/></Response>", media_type="text/xml")

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
        
        # Initialize transcript in session
        session["transcript"] = ""

        # Create UltraVox call and get join URL
        system_prompt = "You are a helpful voice assistant. Answer questions and help schedule meetings."
        join_url = await create_ultravox_call(system_prompt, session["first_message"])

        # Connect to UltraVox WebSocket
        ultravox_ws = await websockets.connect(join_url)

        # Bidirectional relay loop
        async def relay_twilio_to_ultravox():
            while True:
                try:
                    data = await websocket.receive_json()
                    if data.get("event") == "media":
                        media_payload = data.get("media", {}).get("payload")
                        if media_payload:
                            # Forward audio to UltraVox
                            await ultravox_ws.send(json.dumps({
                                "type": "input.audio",
                                "payload": media_payload
                            }))
                    elif data.get("event") == "stop":
                        # Handle stop event
                        transcript = session.get("transcript", "")
                        if transcript:
                            summary = await summarize_transcript(transcript)
                            await send_to_n8n("2", caller, {
                                "transcript": transcript,
                                "summary": summary
                            })
                        break
                except Exception as e:
                    logger.error(f"Error relaying Twilio to UltraVox: {e}")
                    break

        async def relay_ultravox_to_twilio():
            while True:
                try:
                    data = await ultravox_ws.recv()
                    event = json.loads(data)
                    if "response" in event:
                        if "audio" in event["response"] and "delta" in event["response"]["audio"]:
                            # Forward audio to Twilio
                            await websocket.send_json({
                                "event": "media",
                                "media": {"payload": event["response"]["audio"]["delta"]}
                            })
                        elif "text" in event["response"] and "delta" in event["response"]["text"]:
                            # Accumulate transcript
                            session["transcript"] += event["response"]["text"]["delta"]
                        elif "function_call_arguments" in event["response"] and event["response"]["function_call_arguments"].get("done"):
                            # Handle function call
                            func_name = event["response"]["function_call_arguments"].get("name")
                            args = event["response"]["function_call_arguments"].get("arguments", {})
                            if func_name == "question_and_answer":
                                # Use Pinecone for Q&A
                                answer = await query_pinecone(args.get("question", ""))
                                await ultravox_ws.send(json.dumps({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "function_call_output",
                                        "role": "system",
                                        "output": answer
                                    }
                                }))
                                await ultravox_ws.send(json.dumps({"type": "response.create"}))
                            elif func_name == "schedule_meeting":
                                # Use N8N for booking
                                result = await send_to_n8n("3", caller, args)
                                await ultravox_ws.send(json.dumps({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "function_call_output",
                                        "role": "system",
                                        "output": result.get("message", "Meeting scheduled successfully.")
                                    }
                                }))
                                await ultravox_ws.send(json.dumps({"type": "response.create"}))
                except Exception as e:
                    logger.error(f"Error relaying UltraVox to Twilio: {e}")
                    break

        # Run both relay tasks concurrently
        await asyncio.gather(relay_twilio_to_ultravox(), relay_ultravox_to_twilio())

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()
        if 'ultravox_ws' in locals():
            await ultravox_ws.close()

# Root endpoint for health check
@app.get("/")
async def root():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.port)