import os
import time
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
from psycopg2.extensions import register_adapter, AsIs
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

# Initialize embeddings
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embedding_model()

# Database connection settings
CONNECTION_STRING = "postgresql://postgres:MALEK0192k%40@localhost:5433/postgres"
COLLECTION_NAME = "document_chunks_health"

# Initialize Vector Store
# Initialize Vector Store
@st.cache_resource
def init_vector_store():
    return PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        distance_strategy="cosine"  # Lowercase required
    )

vector_store = init_vector_store()

# Initialize database schema
def init_db():
    conn = psycopg2.connect(CONNECTION_STRING)
    with conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE TABLE IF NOT EXISTS document_chunks_health (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding vector(384)
            );
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                chat_id VARCHAR(255),
                question TEXT,
                question_embedding vector(384),
                response_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
    conn.close()

init_db()

# Initialize session state for chat history
if 'chats' not in st.session_state:
    st.session_state.chats = {}
    st.session_state.active_chat = None

# Medical document chunks
documents = [
    "Maintain a balanced diet rich in fruits, vegetables, and lean proteins.",
    "Stay well-hydrated by drinking plenty of water throughout the day.",
    "If you're experiencing nausea, try eating small, frequent meals.",
    "Avoid raw or undercooked foods to minimize the risk of infection.",
    "If you are having trouble eating, consult a dietician for meal replacement options.",
    "Take all medications as prescribed and report any side effects immediately.",
    "Do not stop or change medication without consulting your doctor.",
    "Attend all scheduled appointments for treatments and follow-up care.",
    "Contact us immediately for unusual symptoms like fever, chills, or severe pain.",
    "Inform all doctors about current medications, including OTC drugs and supplements."
]

# Initialize knowledge base
existing_docs = vector_store.similarity_search(query="health", k=1)
if not existing_docs:
    docs = [Document(page_content=doc) for doc in documents]
    vector_store.add_documents(docs)

# Streamlit UI
st.title("Cancer Care Q&A System")

# Sidebar for chat history
with st.sidebar:
    st.header("Chat History")
    
    if st.button("+ New Chat"):
        new_chat_id = str(time.time())
        st.session_state.chats[new_chat_id] = {
            'title': 'New Chat', 
            'messages': []
        }
        st.session_state.active_chat = new_chat_id
    
    for chat_id in list(st.session_state.chats.keys()):
        chat = st.session_state.chats[chat_id]
        title = chat['title']
        if st.button(title, key=chat_id):
            st.session_state.active_chat = chat_id

def get_relevant_chunks(question: str, top_n: int = 3) -> list:
    docs = vector_store.similarity_search(question, k=top_n)
    return [doc.page_content for doc in docs]

if st.session_state.active_chat:
    current_chat = st.session_state.chats[st.session_state.active_chat]
    st.subheader(current_chat['title'])

    for message in current_chat['messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["type"] == "response":
                st.markdown("**Relevant Medical Guidance:**")
                for i, chunk in enumerate(message["chunks"], 1):
                    st.markdown(f"{i}. {chunk}")
                st.info("**Summary of Recommendations:**\n" + message["summary"])

    question = st.chat_input("Ask your cancer care-related question:")

    if question:
        # Generate question embedding
        question_embedding = embeddings.embed_query(question)
        
        # Store question in database
        conn = psycopg2.connect(CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO chat_history 
            (chat_id, question, question_embedding) 
            VALUES (%s, %s, %s) RETURNING id""",
            (st.session_state.active_chat, question, question_embedding)
        )
        question_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()

        if len(current_chat['messages']) == 0:
            current_chat['title'] = f"Chat: {question[:30]}..." if len(question) > 30 else question
        
        current_chat['messages'].append({
            "role": "user",
            "type": "question",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner('Analyzing your question...'):
            try:
                relevant_chunks = get_relevant_chunks(question)
                summary = "\n".join([f"- {chunk.rstrip('.')}." for chunk in relevant_chunks])
                
                # Update database with response
                conn = psycopg2.connect(CONNECTION_STRING)
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE chat_history SET response_summary = %s WHERE id = %s",
                    (summary, question_id)
                )
                conn.commit()
                cursor.close()
                conn.close()

                current_chat['messages'].append({
                    "role": "assistant",
                    "type": "response",
                    "content": "Here's what I found:",
                    "chunks": relevant_chunks,
                    "summary": summary
                })

                with st.chat_message("assistant"):
                    st.markdown("Here's what I found:")
                    st.markdown("**Relevant Medical Guidance:**")
                    for i, chunk in enumerate(relevant_chunks, 1):
                        st.markdown(f"{i}. {chunk}")
                    st.info("**Summary of Recommendations:**\n" + summary)
                    
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
else:
    st.info("Click '+ New Chat' in the sidebar to start a conversation")