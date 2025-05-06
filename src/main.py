from fastapi import FastAPI, WebSocket, Request, Response, WebSocketDisconnect
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
import io
import wave
import audioop
import tempfile

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
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
    
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")
    if not pinecone_env:
        raise RuntimeError("PINECONE_ENVIRONMENT is not set")
        
    logger.info(f"Initializing Pinecone with environment: {pinecone_env}")
    pinecone = Pinecone(
        api_key=pinecone_api_key,
        environment=pinecone_env
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

# Helper function to fetch chat history from N8N
async def fetch_chat_history(caller: str) -> Dict:
    """Fetch chat history from N8N and return the first message."""
    try:
        if not settings.n8n_webhook_url:
            logger.warning("N8N_WEBHOOK_URL is not set, using default welcome message")
            return {"firstMessage": "Hello! How can I help you today?"}
            
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    settings.n8n_webhook_url,
                    json={
                        "route": 1,  # Route 1 for fetch_chat_history
                        "caller": caller
                    },
                    timeout=5.0  # Add a timeout
                )
                response.raise_for_status()  # Raise an exception for 4XX/5XX responses
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error while fetching chat history: {e.response.status_code} - {e.response.text}")
                return {"firstMessage": "Hello! How can I help you today?"}
            except httpx.RequestError as e:
                logger.error(f"Request error while fetching chat history: {e}")
                return {"firstMessage": "Hello! How can I help you today?"}
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        return {"firstMessage": "Hello! How can I help you today?"}

# Helper function to schedule meeting
async def schedule_meeting(meeting_details: Dict) -> Dict:
    """Schedule a meeting using N8N."""
    try:
        if not settings.n8n_webhook_url:
            logger.warning("N8N_WEBHOOK_URL is not set, cannot schedule meeting")
            return {"message": "I'm sorry, meeting scheduling is not available at the moment."}
            
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    settings.n8n_webhook_url,
                    json={
                        "route": 3,  # Route 3 for schedule_meeting
                        "details": meeting_details
                    },
                    timeout=5.0  # Add a timeout
                )
                response.raise_for_status()  # Raise an exception for 4XX/5XX responses
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error while scheduling meeting: {e.response.status_code} - {e.response.text}")
                return {"message": "I'm sorry, I encountered an error while trying to schedule your meeting. Please try again later."}
            except httpx.RequestError as e:
                logger.error(f"Request error while scheduling meeting: {e}")
                return {"message": "I'm sorry, I couldn't connect to the scheduling service. Please try again later."}
    except Exception as e:
        logger.error(f"Error scheduling meeting: {e}")
        return {"message": "I'm sorry, I encountered an unexpected error while scheduling your meeting."}

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

        # Skip N8N if webhook URL is not set
        if not settings.n8n_webhook_url:
            logger.warning("N8N_WEBHOOK_URL is not set, skipping saving transcript")
            return {"status": "skipped"}
            
        # Send transcript and summary to N8N
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    settings.n8n_webhook_url,
                    json={
                        "route": 2,  # Route 2 for save_transcript_summary
                        "session_id": session_id,
                        "transcript": transcript,
                        "summary": summary
                    },
                    timeout=5.0  # Add a timeout
                )
                response.raise_for_status()  # Raise an exception for 4XX/5XX responses
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error while saving transcript: {e.response.status_code} - {e.response.text}")
                return {"error": "Failed to save transcript due to HTTP error"}
            except httpx.RequestError as e:
                logger.error(f"Request error while saving transcript: {e}")
                return {"error": "Failed to save transcript due to connection error"}
    except Exception as e:
        logger.error(f"Error saving transcript and summary: {e}")
        return {"error": "Failed to save transcript and summary"}

# Helper function to convert mulaw audio to WAV for OpenAI
def convert_mulaw_to_wav(mulaw_data, sample_rate=8000, channels=1):
    """Convert Twilio's mulaw audio to WAV format that OpenAI can process."""
    try:
        # Check if there's enough audio data (at least 0.1 seconds at 8000 samples/sec)
        if len(mulaw_data) < 800:  # 8000 samples/sec * 0.1 sec = 800 samples
            logger.warning(f"Audio data too short: {len(mulaw_data)} bytes, need at least 800 bytes")
            return None
            
        # Convert mulaw to PCM
        pcm_data = audioop.ulaw2lin(mulaw_data, 2)  # 2 bytes per sample
        
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            with wave.open(temp_wav.name, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)  # 2 bytes per sample
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_data)
            
            # Read the WAV file back
            with open(temp_wav.name, 'rb') as f:
                wav_data = f.read()
                
        return wav_data
    except Exception as e:
        logger.error(f"Error converting mulaw to WAV: {e}")
        return None

# Buffer to accumulate audio data
class AudioBuffer:
    def __init__(self, min_length=16000):  # Increase to 2 seconds at 8kHz
        self.buffer = bytearray()
        self.min_length = min_length  # Minimum length for processing (2 seconds)
        
    def add_data(self, data):
        """Add data to the buffer, return True if buffer exceeds minimum length"""
        if data:
            self.buffer.extend(data)
        return len(self.buffer) >= self.min_length
        
    def get_data(self):
        """Get the buffered data and clear the buffer"""
        data = bytes(self.buffer)
        self.buffer.clear()
        return data

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

        # Try to fetch chat history, but provide default if n8n fails
        try:
            chat_history = await fetch_chat_history(caller)
            first_message = chat_history.get("firstMessage", "Hello! How can I help you today?")
        except Exception as e:
            logger.warning(f"Failed to fetch chat history from n8n, using default greeting: {e}")
            first_message = "Hello! I'm your AI assistant. How can I help you today?"

        # Create a session
        session_id = f"session_{format_phone_number(caller)}"
        sessions[session_id] = {
            "caller": caller,
            "transcript": "",
            "first_message": first_message,
            "created_at": datetime.now().isoformat()
        }

        # Generate TwiML to connect to media stream using domain from settings
        # Add track attribute to make sure audio is bidirectional
        twiml = f"""
        <Response>
            <Connect>
                <Stream url="wss://{settings.domain}/media-stream" track="both">
                    <Parameter name="sessionId" value="{session_id}"/>
                </Stream>
            </Connect>
            <Say>If you cannot hear the assistant, please wait a moment.</Say>
        </Response>
        """
        logger.info(f"Returning TwiML: {twiml}")
        return Response(content=twiml, media_type="text/xml")
    except Exception as e:
        logger.error(f"Error handling incoming call: {str(e)}")
        return Response(content="<Response><Reject/></Response>", media_type="text/xml")

# Media stream WebSocket handler
@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    logger.info("WebSocket connection attempt received")
    await websocket.accept()
    session_id = None
    connection_open = True
    
    try:
        # 1. Wait for 'connected' event from Twilio
        msg = await websocket.receive_json()
        logger.info(f"Twilio event received: {msg}")
        if msg.get("event") != "connected":
            logger.warning(f"Expected 'connected' event but got: {msg.get('event')}")
            # Continue anyway - some clients might have different handshakes
        
        # 2. Wait for 'start' event, extract sessionId from multiple possible locations
        msg = await websocket.receive_json()
        logger.info(f"Twilio event received: {json.dumps(msg)}")  # Log entire message for debugging
        
        # Extract sessionId from various possible locations in the event
        if msg.get("event") == "start":
            # Try multiple paths where sessionId might be found
            start_data = msg.get("start", {})
            custom_parameters = start_data.get("customParameters", {})
            regular_parameters = start_data.get("parameters", {})
            
            # Log all potential locations for debugging
            logger.info(f"Start data: {json.dumps(start_data)}")
            logger.info(f"Custom parameters: {json.dumps(custom_parameters)}")
            logger.info(f"Regular parameters: {json.dumps(regular_parameters)}")
            
            # Try multiple locations for sessionId, with specific logging
            session_id = custom_parameters.get("sessionId")
            if session_id:
                logger.info(f"Found sessionId in customParameters: {session_id}")
            else:
                session_id = regular_parameters.get("sessionId")
                if session_id:
                    logger.info(f"Found sessionId in parameters: {session_id}")
                else:
                    # Try one more fallback - caller ID / phone number
                    account_sid = start_data.get("accountSid")
                    stream_sid = start_data.get("streamSid")
                    logger.info(f"No sessionId found. AccountSid: {account_sid}, StreamSid: {stream_sid}")
                    
                    # Look for call info in the event that might contain a phone number
                    caller = None
                    for key in start_data:
                        if isinstance(start_data[key], dict) and "from" in start_data[key]:
                            caller = start_data[key].get("from")
                            break
                    
                    if caller:
                        logger.info(f"Found caller in start event: {caller}")
                        session_id = f"session_{format_phone_number(caller)}"
        else:
            logger.warning(f"Expected 'start' event but got: {msg.get('event')}")
        
        logger.info(f"Final extracted sessionId: {session_id}")
        
        # Show all available sessions for debugging
        logger.info(f"Available sessions: {list(sessions.keys())}")
        
        # Check if session exists - with fallback options
        if not session_id:
            logger.error("Could not extract a sessionId from the Twilio event")
            await websocket.close()
            return
            
        if session_id not in sessions:
            # If the exact session is not found, try normalizing the phone number
            logger.warning(f"Session {session_id} not found in active sessions")
            
            # Try a secondary lookup based on phone numbers if it looks like a phone-based session
            if session_id.startswith("session_"):
                phone_part = session_id[8:]  # Extract just the phone number part
                # Look for sessions with similar phone numbers
                for existing_id, session_data in sessions.items():
                    if existing_id.startswith("session_"):
                        existing_phone = existing_id[8:]
                        if phone_part in existing_phone or existing_phone in phone_part:
                            logger.info(f"Found similar session: {existing_id}")
                            session_id = existing_id
                            break
        
        if session_id not in sessions:
            logger.error(f"Invalid or missing sessionId: {session_id}")
            await websocket.close()
            return
        
        logger.info(f"Successfully found session: {session_id}")
        session = sessions[session_id]
        
        # Initialize OpenAI conversation
        messages = [
            {
                "role": "system",
                "content": """You are a friendly and professional voice AI assistant. Your name is Alex. \
                You are speaking to users over the phone, so keep your responses natural and conversational.

                Guidelines:
                1. Be friendly but professional
                2. Keep responses concise but informative
                3. If you don't know something, say so
                4. For scheduling meetings, ask for: name, email, purpose, date/time, and location
                5. Use natural pauses and conversational markers like 'um', 'let me see', etc.
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
        
        # Initialize audio buffer for accumulating short audio chunks
        audio_buffer = AudioBuffer(min_length=16000)  # 2 seconds at 8kHz
        
        # IMPORTANT: Immediately send the welcome message
        try:
            logger.info(f"Sending welcome message: {session['first_message']}")
            
            # Generate welcome message audio
            speech_response = openai_client.audio.speech.create(
                model="tts-1",
                voice=settings.openai_voice,
                input=session["first_message"],
                response_format="mp3",
                speed=1.0
            )
            
            logger.info(f"Generated audio of size: {len(speech_response.content)} bytes")
            
            # Send welcome audio to Twilio
            try:
                # Check if connection is still open
                if connection_open:
                    payload = base64.b64encode(speech_response.content).decode()
                    logger.info(f"Encoded audio to base64, size: {len(payload)}")
                    
                    await websocket.send_json({
                        "event": "media",
                        "media": {
                            "payload": payload,
                            "track": "outbound"  # Add track parameter to specify direction
                        }
                    })
                    logger.info("Welcome message sent successfully")
                else:
                    logger.warning("Cannot send welcome message - connection not open")
                    return
            except Exception as e:
                logger.error(f"WebSocket error sending welcome message: {e}")
                if "WebSocket is not connected" in str(e) or "Cannot call \"send\"" in str(e):
                    logger.warning("WebSocket disconnected")
                    connection_open = False
                    return
        except Exception as e:
            logger.error(f"Error generating welcome message: {e}")
        
        # Main conversation loop
        while connection_open:
            try:
                # Set timeout for receiving data
                try:
                    data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning("WebSocket receive timeout - no data received for 30 seconds")
                    # Send a ping to check connection
                    try:
                        if connection_open:
                            await websocket.send_json({"event": "ping"})
                            logger.info("Ping sent to keep connection alive")
                            continue
                        else:
                            logger.warning("Connection no longer open")
                            break
                    except Exception as ping_error:
                        logger.error(f"Failed to send ping, connection likely lost: {ping_error}")
                        connection_open = False
                        break
                
                if data.get("event") == "media":
                    audio_data = data.get("media", {}).get("payload")
                    if not audio_data:
                        logger.warning("Received media event without payload")
                        continue
                    
                    audio_text = None
                    try:
                        # Decode base64 audio
                        audio_bytes = base64.b64decode(audio_data)
                        
                        # Add to buffer
                        if not audio_buffer.add_data(audio_bytes):
                            logger.info("Buffering audio data, waiting for more...")
                            continue
                            
                        # Get accumulated audio data
                        buffered_audio = audio_buffer.get_data()
                        logger.info(f"Processing {len(buffered_audio)} bytes of audio data")
                        
                        # Convert mulaw to WAV format for OpenAI
                        wav_bytes = convert_mulaw_to_wav(buffered_audio)
                        if not wav_bytes:
                            logger.warning("Failed to convert audio or audio too short, waiting for more")
                            continue
                            
                        audio_file = io.BytesIO(wav_bytes)
                        audio_file.name = "audio.wav"
                        
                        try:
                            # Transcribe audio
                            audio_text = openai_client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file
                            ).text
                            
                            logger.info(f"Transcribed text: {audio_text}")
                        except Exception as e:
                            if "audio_too_short" in str(e):
                                logger.warning("Audio too short for transcription, skipping this chunk")
                                continue  # Skip this audio chunk and wait for more audio
                            else:
                                # For other errors, log and continue with fallback
                                logger.error(f"Error transcribing audio: {e}")
                                audio_text = "[Transcription failed. Please try again.]"
                    except Exception as e:
                        logger.error(f"Error processing audio data: {e}")
                        # Set a fallback message if transcription fails
                        audio_text = "[Transcription failed. Please try again.]"
                    
                    # Skip empty transcriptions
                    if not audio_text or audio_text.strip() == "":
                        logger.warning("Empty transcription, skipping")
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
                        try:
                            # Send goodbye message
                            speech_response = openai_client.audio.speech.create(
                                model="tts-1",
                                voice=settings.openai_voice,
                                input="You're welcome! Have a great day!",
                                response_format="mp3",
                                speed=1.0
                            )
                            
                            # Check if connection is still open
                            if connection_open:
                                payload = base64.b64encode(speech_response.content).decode()
                                await websocket.send_json({
                                    "event": "media",
                                    "media": {
                                        "payload": payload,
                                        "track": "outbound"  # Add track parameter to specify direction
                                    }
                                })
                            else:
                                logger.warning("Cannot send goodbye - connection closed")
                        except Exception as e:
                            logger.error(f"Error sending goodbye message: {e}")
                            connection_open = False
                            
                        # Save transcript - but don't let n8n failure stop us
                        try:
                            await save_transcript_summary(session_id, session["transcript"])
                        except Exception as e:
                            logger.error(f"Error saving transcript: {e}")
                            
                        break
                    
                    try:
                        # Process message and generate response
                        answer = None
                        # Skip n8n calls if needed - respond directly from Pinecone
                        if "schedule" in audio_text.lower() or "meeting" in audio_text.lower():
                            try:
                                meeting_details = await extract_meeting_details(audio_text)
                                if meeting_details:
                                    result = await schedule_meeting(meeting_details)
                                    answer = result.get("message", "I've scheduled the meeting for you. Is there anything else I can help with?")
                                else:
                                    answer = "I'd be happy to schedule a meeting for you. Could you please provide your name, email, the purpose of the meeting, and your preferred date and time?"
                            except Exception as e:
                                logger.error(f"Error scheduling meeting: {e}")
                                answer = "I'm sorry, I'm having trouble with the scheduling service right now. Could we try again in a moment?"
                        else:
                            # Use Pinecone for Q&A
                            try:
                                answer = await query_knowledge_base(audio_text)
                            except Exception as e:
                                logger.error(f"Error querying knowledge base: {e}")
                                answer = "I'm sorry, I'm having trouble accessing my knowledge base right now."
                        
                        # Add assistant's response to messages
                        messages.append({"role": "assistant", "content": answer})
                        
                        # Use Chat API as fallback if needed
                        if not answer or answer.startswith("I couldn't find"):
                            try:
                                response = openai_client.chat.completions.create(
                                    model=settings.openai_model,
                                    messages=messages,
                                    temperature=0.7,
                                    max_tokens=150
                                )
                                assistant_message = response.choices[0].message.content
                            except Exception as e:
                                logger.error(f"Error with chat completion: {e}")
                                assistant_message = "I'm sorry, I'm having trouble processing that right now. Could you please try again?"
                        else:
                            assistant_message = answer
                        
                        # Convert to speech and send
                        speech_response = openai_client.audio.speech.create(
                            model="tts-1",
                            voice=settings.openai_voice,
                            input=assistant_message,
                            response_format="mp3",
                            speed=1.0
                        )
                        
                        logger.info(f"Generated response audio of size: {len(speech_response.content)} bytes for message: {assistant_message[:30]}...")
                        
                        try:
                            # Check if connection is still open
                            if connection_open:
                                payload = base64.b64encode(speech_response.content).decode()
                                logger.info(f"Encoded response audio to base64, sending {len(payload)} bytes")
                                
                                await websocket.send_json({
                                    "event": "media",
                                    "media": {
                                        "payload": payload,
                                        "track": "outbound"  # Add track parameter to specify direction
                                    }
                                })
                                logger.info("Response audio sent successfully")
                            else:
                                logger.warning("Cannot send response - connection closed")
                                break
                        except Exception as ws_error:
                            logger.error(f"Error sending audio response: {ws_error}")
                            if "WebSocket is not connected" in str(ws_error) or "Cannot call \"send\"" in str(ws_error):
                                logger.warning("WebSocket connection lost, breaking conversation loop")
                                connection_open = False
                                break
                    except Exception as e:
                        logger.error(f"Error generating or sending response: {e}")
                        # Send fallback message if there's an error
                        try:
                            if connection_open:
                                fallback_message = "I'm sorry, I'm having trouble processing that right now. Could you please try again?"
                                speech_response = openai_client.audio.speech.create(
                                    model="tts-1",
                                    voice=settings.openai_voice,
                                    input=fallback_message,
                                    response_format="mp3",
                                    speed=1.0
                                )
                                
                                await websocket.send_json({
                                    "event": "media",
                                    "media": {
                                        "payload": base64.b64encode(speech_response.content).decode(),
                                        "track": "outbound"  # Add track parameter to specify direction
                                    }
                                })
                            else:
                                logger.warning("Cannot send fallback - connection closed")
                                break
                        except Exception as e2:
                            logger.error(f"Error sending fallback message: {e2}")
                            connection_open = False
                        
                elif data.get("event") == "stop":
                    logger.info("Received stop event, ending call")
                    try:
                        await save_transcript_summary(session_id, session["transcript"])
                    except Exception as e:
                        logger.error(f"Error saving transcript on stop: {e}")
                    break
                elif data.get("event") == "mark":
                    # Handle mark events (related to recording)
                    logger.info(f"Received mark event: {data}")
                elif data.get("event") == "pong":
                    # Handle pong response
                    logger.info("Received pong from client")
                else:
                    logger.info(f"Received unknown event: {data.get('event')}")
            except WebSocketDisconnect:
                logger.warning("WebSocket disconnected")
                connection_open = False
                break
            except Exception as e:
                logger.error(f"Error in conversation loop: {e}")
                connection_open = False
                break
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_open = False
    finally:
        try:
            # Only try to close if not already closed
            if connection_open:
                await websocket.close()
                logger.info("WebSocket closed cleanly")
        except Exception as e:
            logger.error(f"Error closing websocket: {e}")

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