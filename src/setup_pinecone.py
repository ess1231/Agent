import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import json

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Sample knowledge base data
knowledge_base = [
    {
        "text": "Our company offers 24/7 customer support through our voice AI assistant. The assistant can help with product information, troubleshooting, and scheduling meetings.",
        "metadata": {"source": "company_info", "category": "support"}
    },
    {
        "text": "To schedule a meeting, you need to provide: your name, email address, purpose of the meeting, preferred date and time, and location (virtual or physical).",
        "metadata": {"source": "meeting_info", "category": "scheduling"}
    },
    {
        "text": "Our business hours are Monday to Friday, 9 AM to 5 PM EST. For urgent matters outside these hours, please leave a message and we'll get back to you as soon as possible.",
        "metadata": {"source": "company_info", "category": "hours"}
    }
]

def create_embeddings(texts):
    """Create embeddings for the given texts using OpenAI."""
    embeddings = []
    for text in texts:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

def setup_pinecone():
    """Set up Pinecone index and populate it with knowledge base data."""
    try:
        # Create index if it doesn't exist
        index_name = os.getenv("PINECONE_INDEX_NAME", "voice-agent")
        if index_name not in pinecone.list_indexes().names():
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # OpenAI text-embedding-3-small dimension
                metric="cosine"
            )
            print(f"Created new index: {index_name}")
        
        # Get the index
        index = pinecone.Index(index_name)
        
        # Create embeddings for the knowledge base
        texts = [item["text"] for item in knowledge_base]
        embeddings = create_embeddings(texts)
        
        # Prepare vectors for upsert
        vectors = []
        for i, (embedding, item) in enumerate(zip(embeddings, knowledge_base)):
            vectors.append({
                "id": f"vec_{i}",
                "values": embedding,
                "metadata": item["metadata"]
            })
        
        # Upsert vectors to Pinecone
        index.upsert(vectors=vectors)
        print(f"Successfully upserted {len(vectors)} vectors to {index_name}")
        
        # Test the index
        test_query = "What are your business hours?"
        test_embedding = create_embeddings([test_query])[0]
        results = index.query(
            vector=test_embedding,
            top_k=1,
            include_metadata=True
        )
        
        print("\nTest query results:")
        for match in results.matches:
            print(f"Score: {match.score}")
            print(f"Text: {match.metadata}")
        
    except Exception as e:
        print(f"Error setting up Pinecone: {e}")

if __name__ == "__main__":
    setup_pinecone() 