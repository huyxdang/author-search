# user interface
# app.py
import streamlit as st
from qdrant_client import QdrantClient
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
qdrant_client = QdrantClient(path="./qdrant_data")

def embed_query(query):
    """Generate embedding for search query"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    return response.data[0].embedding

def search_authors(query, limit=10):
    """Search for authors using vector similarity"""
    query_embedding = embed_query(query)
    
    results = qdrant_client.search(
        collection_name="authors",
        query_vector=query_embedding,
        limit=limit
    )
    
    return results

# Streamlit UI
st.set_page_config(page_title="Author Search", page_icon="🔍", layout="wide")

st.title("🔍 Researcher Search")
st.caption("Search for researchers by their work and interests")

# Search input
query = st.text_input(
    "Search for researchers",
    placeholder="e.g., GUI agents, multimodal learning, Vietnamese researchers...",
    help="Describe the researchers you're looking for"
)

num_results = st.slider("Number of results", 5, 50, 10)

if query:
    with st.spinner("Searching..."):
        results = search_authors(query, limit=num_results)
    
    st.success(f"Found {len(results)} researchers")
    
    for i, result in enumerate(results, 1):
        author = result.payload
        score = result.score
        
        with st.expander(f"**{i}. {author['name']}** (similarity: {score:.2f})", expanded=(i <= 3)):
            col1, col2, col3 = st.columns(3)
            col1.metric("Papers", author['paper_count'])
            col2.metric("First pub", author['first_year'])
            col3.metric("Last pub", author['last_year'])
            
            st.markdown("**Research Profile:**")
            # Show first few sentences of profile
            profile_preview = author['profile_text'].split('\n')[:5]
            st.write('\n'.join(profile_preview))
            
            st.markdown("**Recent Papers:**")
            for paper in author['papers']:
                st.markdown(f"- *{paper['title']}* ({paper['published'][:4]})")
else:
    st.info("👆 Enter a search query to find researchers")