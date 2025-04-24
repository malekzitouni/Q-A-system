import os
import time
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
from psycopg2.extensions import register_adapter, AsIs
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load environment variables
load_dotenv()

# Caching the embedding model to avoid reloading it on every run
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Cache the generation model using Hugging Face pipeline.
# Here we use an Arabic generation model. Feel free to change to a model you prefer.
@st.cache_resource
def load_generation_model():
    return pipeline("text-generation", model="aubmindlab/aragpt2-base")

embedding_model = load_embedding_model()
generator = load_generation_model()

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5433,
        database="postgres",
        user="postgres",
        password="MALEK0192k@"
    )

# Initialize database schema
def init_db():
    conn = get_db_connection()
    with conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
            DROP TABLE IF EXISTS document_chunks_health;
            CREATE TABLE document_chunks_health (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding vector(768)
            );
            DROP TABLE IF EXISTS chat_history;
            CREATE TABLE chat_history (
                id SERIAL PRIMARY KEY,
                chat_id VARCHAR(255),
                question TEXT,
                question_embedding vector(768),
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
    "السرطان هو مرض يتسبب في نمو غير طبيعي للخلايا. يمكن أن يحدث في أي جزء من الجسم و يمكن أن ينتشر إلى أجزاء أخرى.",
    "الأعراض تختلف حسب نوع السرطان. ممكن تشمل فقدان الوزن، تعب، ألم، أو تغييرات في الجلد.",
    "الشيميو يدمر الخلايا السرطانية لكن ممكن يسبب تعب، تساقط الشعر، وغثيان. المدة تفرق من شخص لاخر، غالبا جلسات على شهور.",
    "تقدر تخدم إذا الحالة تسمح، لكن خذّ راحتك و ما ترهقش راسك. الإرهاق يزيد التعافي صعوبة",
    "العلاج الإشعاعي يستعمل أشعة عالية الطاقة لتدمير الخلايا السرطانية. ممكن يسبب تعب و مشاكل جلدية.",
    "الجراحة تهدف لإزالة الورم. ممكن تكون جراحة بسيطة أو معقدة حسب الحالة.",
    "العلاج المناعي يقوي جهاز المناعة لمحاربة السرطان. ممكن يسبب أعراض جانبية مثل الحمى و التعب.",
    "العلاج الهرموني يستعمل لتقليل تأثير الهرمونات على نمو السرطان. ممكن يسبب أعراض مثل الهبات الساخنة.",
    "العلاج المستهدف يستهدف خلايا سرطانية معينة. ممكن يسبب أعراض جانبية مثل الإسهال و الطفح الجلدي.",
    "السرطان في مراحله الأولى يكون أسهل في العلاج. كلما تقدم، العلاج يصبح أصعب.",
    "السرطان في مراحله المتقدمة يمكن أن ينتشر إلى أجزاء أخرى من الجسم. العلاج يكون أكثر تعقيد.",
    "السرطان يمكن أن يؤثر على أي شخص، لكن بعض الأنواع أكثر شيوعًا في فئات عمرية معين",
    "كل أكل صحي: خضار، فواكه، بروتين. تجنب أكل غير نظيف",
    "الوزن الزائد يزيد خطر الإصابة بالسرطان. حاول تحافظ على وزن صحي.",
    "الكحول يزيد خطر الإصابة بأنواع معينة من السرطان. حاول تحد من استهلاكه.",
    "التدخين هو أحد أكبر عوامل الخطر للإصابة بالسرطان. حاول تتجنب التدخين و الأماكن المليئة بالدخان.",
    "التعرض لأشعة الشمس يزيد خطر الإصابة بسرطان الجلد. حاول تحمي نفسك من الشمس و استعمل واقي شمس.",
    "التعرض للمواد الكيميائية السامة يزيد خطر الإصابة بالسرطان. حاول تتجنب التعرض لها.",
    "الفحص المبكر يساعد في الكشف عن السرطان في مراحله الأولى. استشر طبيبك عن الفحوصات المناسبة لك.",
    "الفحص الذاتي للثدي يساعد في الكشف عن سرطان الثدي مبكرًا. تعلم كيف تقوم به.",
    "الفحص الذاتي للخصيتين يساعد في الكشف عن سرطان الخصية مبكرًا. تعلم كيف تقوم به.",
    "الفحص المبكر لسرطان القولون يساعد في الكشف عنه مبكرًا. استشر طبيبك عن الفحوصات المناسبة لك.",
    "الفحص المبكر لسرطان الرئة يساعد في الكشف عنه مبكرًا. استشر طبيبك عن الفحوصات المناسبة لك.",
    "الفحص المبكر لسرطان البروستاتا يساعد في الكشف عنه مبكرًا. استشر طبيبك عن الفحوصات المناسبة لك.",
    "الفحص المبكر لسرطان عنق الرحم يساعد في الكشف عنه مبكرًا. استشر طبيبك عن الفحوصات المناسبة لك.",
    "الفحص المبكر لسرطان الجلد يساعد في الكشف عنه مبكرًا. استشر طبيبك عن الفحوصات المناسبة لك.",
    "في حالات نادرة، لكن غالبا ماشي وراثي. إذا العائلة عندها تاريخ، نقدر نعملوا فحوصات وقائية",
    "الوقاية تشمل نمط حياة صحي، تجنب التدخين، و ممارسة الرياضة. الفحص المبكر يساعد في الكشف عن السرطان في مراحله الأولى.",
    "العلاج يعتمد على نوع السرطان و المرحلة. ممكن نستعملوا الشيميو، الإشعاع، الجراحة، أو العلاج الما يهاجم خلايا معيّنة. كل حالة خاصة بيها",
]

# Populate the knowledge base if empty
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM document_chunks_health")
if cursor.fetchone()[0] == 0:
    for doc in documents:
        embedding = embedding_model.encode(doc).tolist()
        cursor.execute(
            "INSERT INTO document_chunks_health (content, embedding) VALUES (%s, %s)",
            (doc, embedding)
        )
    conn.commit()
cursor.close()
conn.close()

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
    
    for chat_id, chat in st.session_state.chats.items():
        title = chat['title']
        if st.button(title, key=chat_id):
            st.session_state.active_chat = chat_id

def get_relevant_chunks(embedding: list, top_n: int = 3) -> list:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT content 
        FROM document_chunks_health
        ORDER BY embedding <=> %s::vector(768)
        LIMIT %s
    """, (embedding, top_n))
    result = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return result

if st.session_state.active_chat:
    current_chat = st.session_state.chats[st.session_state.active_chat]
    st.subheader(current_chat['title'])

    for message in current_chat['messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("type") == "response":
                st.markdown("**Relevant Medical Guidance:**")
                for i, chunk in enumerate(message["chunks"], 1):
                    st.markdown(f"{i}. {chunk}")
                st.info("**Summary of Recommendations:**\n" + message["summary"])

    question = st.chat_input("Ask your cancer care-related question:")

    if question:
        # Generate question embedding
        question_embedding = embedding_model.encode(question).tolist()
        
        # Store the question in the database
        conn = get_db_connection()
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

        # Set or update chat title based on the first question
        if len(current_chat['messages']) == 0:
            current_chat['title'] = f"Chat: {question[:30]}..." if len(question) > 30 else question
        
        # Add the user question to the chat history
        current_chat['messages'].append({
            "role": "user",
            "type": "question",
            "content": question
        })
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner('Analyzing your question...'):
            try:
                # Retrieve the most relevant document chunks
                relevant_chunks = get_relevant_chunks(question_embedding)
                
                # Create a summary from the retrieved chunks
                summary = "\n".join([f"- {chunk.rstrip('.')}." for chunk in relevant_chunks])
                
                # Update the chat history in the database with this summary
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE chat_history SET response_summary = %s WHERE id = %s",
                    (summary, question_id)
                )
                conn.commit()
                cursor.close()
                conn.close()
                
                # Build a prompt for the generation model
                prompt = (
                    "أنت مساعد طبي متخصص في رعاية مرضى السرطان. "
                    "استناداً إلى السؤال التالي والمعلومات المستخلصة من الوثائق:\n\n"
                    f"السؤال: {question}\n\n"
                    "المعلومات الطبية:\n"
                    f"{summary}\n\n"
                    "رجاءً قم بتوليد إجابة شاملة ومفيدة تدمج هذه المعلومات وتوضح التوصيات الطبية بشكل واضح."
                )
                
                # Generate the final answer using the generation model
                generation_output = generator(prompt, max_length=200, do_sample=True, num_return_sequences=1)
                final_answer = generation_output[0]['generated_text']

                # Append the response to chat history
                current_chat['messages'].append({
                    "role": "assistant",
                    "type": "response",
                    "content": "إليك ما وجدته بناءً على المعلومات المتوفرة:",
                    "chunks": relevant_chunks,
                    "summary": summary,
                    "final_answer": final_answer
                })
                
                with st.chat_message("assistant"):
                    st.markdown("هذه المعلومات التي تم استنتاجها:")
                    st.markdown("**التوصيات الطبية المستخلصة:**")
                    for i, chunk in enumerate(relevant_chunks, 1):
                        st.markdown(f"{i}. {chunk}")
                    st.info("**ملخص التوصيات:**\n" + summary)
                    st.markdown("**الإجابة النهائية:**")
                    st.write(final_answer)
                    
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
else:
    st.info("Click '+ New Chat' in the sidebar to start a conversation")
