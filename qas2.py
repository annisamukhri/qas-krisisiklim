import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import pandas as pd

@st.cache_resource
def load_model():
    return SentenceTransformer('finetuned-indoSBERT')

model = load_model()

@st.cache_data
def load_embeddings(embeddings_path):
    with open(embeddings_path, 'rb') as fIn:
        cached_data = pickle.load(fIn)
        corpus_questions = cached_data['questions']
        corpus_answers = cached_data['answers']
        question_embeddings = cached_data['question_embeddings']
        answer_embeddings = cached_data['answer_embeddings']
    return corpus_questions, question_embeddings, corpus_answers, answer_embeddings

@st.cache_data
def load_url_mapping(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df['question'].tolist(), df['url'].tolist()
    except Exception as e:
        st.error(f"Error loading URL mapping: {e}")
        return [], []

st.title("Question Answering System untuk Topik Krisis Iklim")

embeddings_path = 'embeddings_indoSBERT-e5b16.pkl'
csv_path = 'scraping/dataset1.csv'  
corpus_questions, question_embeddings, corpus_answers, answer_embeddings = load_embeddings(embeddings_path)
csv_questions, csv_urls = load_url_mapping(csv_path)
url_mapping = dict(zip(csv_questions, csv_urls))  


query = st.text_input("Masukan Pertanyaan:", placeholder="Type your question here...")
top_k = 5

if query:
    st.subheader("Search Results:")
    try:
        query_embedding = model.encode(query, convert_to_tensor=True)

        scores = util.cos_sim(query_embedding, answer_embeddings)[0]
        top_results = torch.topk(scores, k=top_k)

        for score, idx in zip(top_results.values, top_results.indices):
            question = corpus_questions[idx]
            answer = corpus_answers[idx]
            url = url_mapping.get(question, "URL not available") 

            max_length = 100  
            preview = answer[:max_length] + "..." if len(answer) > max_length else answer

            st.write(f"**Q:** {question}")
            expander_title = f"{preview}"

            with st.expander(expander_title, expanded=False):
                st.write(f"{answer}")
                st.write(f"**URL:** {url}")
    except Exception as e:
        st.error(f"Terjadi error saat memproses query: {e}")
