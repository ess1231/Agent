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
from openai import OpenAI
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define settings using Pydantic
class Settings(BaseSettings):
    # OpenAI settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
    openai_voice: str = os.getenv("OPENAI_VOICE", "alloy")
    
    # Pinecone settings
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "")
    
    # Other settings
    n8n_webhook_url: str = os.getenv("N8N_WEBHOOK_URL", "")
    port: int = int(os.getenv("PORT", "8080"))
    domain: str = os.getenv("RAILWAY_STATIC_URL", "localhost:8080")

settings = Settings()

# Initialize OpenAI client
try:
    openai_client = OpenAI(
        api_key=settings.openai_api_key,
        timeout=30.0,  # Add timeout
        max_retries=3  # Add retries
    )
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise

# Initialize Pinecone client
try:
    pinecone = Pinecone(
        api_key=settings.pinecone_api_key,
        environment=settings.pinecone_environment
    )
    pinecone_index = pinecone.Index(name="voice-agent")
    logger.info(f"Successfully connected to Pinecone index: voice-agent")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone client: {e}")
    raise

# Session store to track ongoing calls
sessions: Dict[str, Dict] = {}

# Initialize FastAPI app
app = FastAPI()

# Helper function to format phone number
def format_phone_number(number: str) -> str:
    """Format phone number by removing the plus sign and any non-digit characters."""
    return ''.join(filter(str.isdigit, number))

# Helper function to query knowledge base
async def query_knowledge_base(question: str) -> str:
    """Query the Pinecone knowledge base for relevant information."""
    try:
        # Generate embedding for the question
        embedding = openai_client.embeddings.create(
            input=question,
            model="text-embedding-3-small"
        ).data[0].embedding

        # Query Pinecone
        results = pinecone_index.query(
            vector=embedding,
            top_k=3,
            include_metadata=True
        )

        # Process results
        if not results.matches:
            return "I couldn't find any relevant information in the knowledge base."

        # Format the response
        context = "\n".join([match.metadata.get('text', '') for match in results.matches])
        
        # Use OpenAI to generate a response based on the context
        response = openai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the question. If the context doesn't contain relevant information, say so."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
        return "I'm sorry, I encountered an error while searching the knowledge base."

# Helper function to schedule meeting
async def schedule_meeting(meeting_details: Dict) -> Dict:
    """Schedule a meeting using N8N."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                settings.n8n_webhook_url,
                json={
                    "route": "schedule_meeting",
                    "details": meeting_details
                }
            )
            return response.json()
    except Exception as e:
        logger.error(f"Error scheduling meeting: {e}")
        return {"error": "Failed to schedule meeting"}

# Helper function to fetch chat history from N8N
async def fetch_chat_history(caller: str) -> Dict:
    """Fetch chat history from N8N and return the first message."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                settings.n8n_webhook_url,
                json={
                    "route": "fetch_chat_history",
                    "caller": caller
                }
            )
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        return {"firstMessage": "Hello! How can I help you today?"}

# Helper function to save transcript and summary to N8N
async def save_transcript_summary(session_id: str, transcript: str) -> Dict:
    """Save transcript and generate a summary using OpenAI, then send to N8N."""
    try:
        # Generate summary using OpenAI
        summary_response = openai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "Summarize the following conversation in a few sentences."},
                {"role": "user", "content": transcript}
            ],
            temperature=0.7,
            max_tokens=100
        )
        summary = summary_response.choices[0].message.content

        # Send transcript and summary to N8N
        async with httpx.AsyncClient() as client:
            response = await client.post(
                settings.n8n_webhook_url,
                json={
                    "route": "save_transcript_summary",
                    "session_id": session_id,
                    "transcript": transcript,
                    "summary": summary
                }
            )
            return response.json()
    except Exception as e:
        logger.error(f"Error saving transcript and summary: {e}")
        return {"error": "Failed to save transcript and summary"}

# Incoming call endpoint
@app.post("/incoming-call")
async def incoming_call(request: Request) -> Response:
    try:
        form_data = await request.form()
        caller = form_data.get("From") or form_data.get("Caller")
        if not caller:
            logger.error("No caller number provided")
            return Response(content="<Response><Reject/></Response>", media_type="text/xml")
            
        logger.info(f"Incoming call from {caller}")

        # Fetch chat history and get first message
        chat_history = await fetch_chat_history(caller)
        first_message = chat_history.get("firstMessage", "Hello! How can I help you today?")

        # Create a session
        session_id = f"session_{format_phone_number(caller)}"
        sessions[session_id] = {
            "caller": caller,
            "transcript": "",
            "first_message": first_message,
            "created_at": datetime.now().isoformat()
        }

        # Generate TwiML to connect to media stream using domain from settings
        twiml = f"""
        <Response>
            <Connect>
                <Stream url="wss://{settings.domain}/media-stream">
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
    logger.info("WebSocket connection attempt received")
    await websocket.accept()
    
    try:
        # Read initial JSON handshake to get sessionId
        handshake = await websocket.receive_json()
        logger.info(f"WebSocket handshake received: {handshake}")
        session_id = handshake.get("sessionId")
        
        if not session_id or session_id not in sessions:
            logger.error(f"Invalid or missing sessionId: {session_id}")
            await websocket.close()
            return

        session = sessions[session_id]
        
        # Initialize OpenAI conversation
        messages = [
            {
                "role": "system",
                "content": """You are a friendly and professional voice AI assistant. Your name is Alex. 
                You are speaking to users over the phone, so keep your responses natural and conversational.
                
                Guidelines:
                1. Be friendly but professional
                2. Keep responses concise but informative
                3. If you don't know something, say so
                4. For scheduling meetings, ask for: name, email, purpose, date/time, and location
                5. Use natural pauses and conversational markers like "um", "let me see", etc.
                6. If the user is unclear, ask clarifying questions
                7. End conversations naturally when appropriate
                
                Remember you're having a real-time conversation, so:
                - Don't use markdown or special formatting
                - Don't list items with numbers unless speaking them out
                - Use natural speech patterns
                - Keep responses brief but complete"""
            },
            {
                "role": "assistant",
                "content": session["first_message"]
            }
        ]

        # Main conversation loop
        while True:
            try:
                # Receive audio data from Twilio
                data = await websocket.receive_json()
                if data.get("event") == "media":
                    audio_data = data.get("media", {}).get("payload")
                    
                    if not audio_data:
                        logger.warning("Received media event without payload")
                        continue
                        
                    try:
                        # Convert audio to text using OpenAI
                        audio_text = openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_data
                        ).text
                    except Exception as e:
                        logger.error(f"Error transcribing audio: {e}")
                        continue
                        
                    # Add to transcript
                    session["transcript"] += f"\nUser: {audio_text}"
                    
                    # Add to conversation
                    messages.append({"role": "user", "content": audio_text})
                    
                    # Check for conversation end
                    if any(phrase in audio_text.lower() for phrase in ["goodbye", "bye", "thanks", "thank you"]):
                        messages.append({
                            "role": "assistant",
                            "content": "You're welcome! Have a great day!"
                        })
                        # Save transcript and summary to N8N
                        await save_transcript_summary(session_id, session["transcript"])
                        break
                    
                    # Check if user wants to schedule a meeting
                    if "schedule" in audio_text.lower() or "meeting" in audio_text.lower():
                        # Extract meeting details using OpenAI
                        meeting_details = await extract_meeting_details(audio_text)
                        if meeting_details:
                            result = await schedule_meeting(meeting_details)
                            messages.append({
                                "role": "assistant",
                                "content": result.get("message", "I've scheduled the meeting for you. Is there anything else I can help with?")
                            })
                        else:
                            messages.append({
                                "role": "assistant",
                                "content": "I'd be happy to schedule a meeting for you. Could you please provide your name, email, the purpose of the meeting, and your preferred date and time?"
                            })
                    else:
                        # Query knowledge base for answers
                        answer = await query_knowledge_base(audio_text)
                        messages.append({"role": "assistant", "content": answer})
                    
                    try:
                        # Generate response using OpenAI
                        response = openai_client.chat.completions.create(
                            model=settings.openai_model,
                            messages=messages,
                            temperature=0.7,
                            max_tokens=150  # Keep responses concise
                        )
                        
                        assistant_message = response.choices[0].message.content
                        
                        # Convert text to speech with appropriate voice settings
                        speech_response = openai_client.audio.speech.create(
                            model="tts-1",
                            voice=settings.openai_voice,
                            input=assistant_message,
                            response_format="mp3",
                            speed=1.0  # Natural speaking speed
                        )
                        
                        # Send audio back to Twilio
                        await websocket.send_json({
                            "event": "media",
                            "media": {
                                "payload": base64.b64encode(speech_response.content).decode()
                            }
                        })
                    except Exception as e:
                        logger.error(f"Error generating or sending response: {e}")
                        continue
                        
                elif data.get("event") == "stop":
                    # Save transcript and summary to N8N
                    await save_transcript_summary(session_id, session["transcript"])
                    break
                    
            except Exception as e:
                logger.error(f"Error in conversation loop: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Helper function to extract meeting details
async def extract_meeting_details(text: str) -> Optional[Dict]:
    """Extract meeting details from user's speech using OpenAI."""
    try:
        response = openai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "Extract meeting details from the text. Return only the details in JSON format."},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        
        details = json.loads(response.choices[0].message.content)
        return details if all(k in details for k in ["name", "email", "purpose", "datetime", "location"]) else None
        
    except Exception as e:
        logger.error(f"Error extracting meeting details: {e}")
        return None

# Root endpoint for health check
@app.get("/")
async def root():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.port)