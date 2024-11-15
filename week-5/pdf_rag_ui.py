import streamlit as st
import spacy
from util import pdf_utils
from util.embedings_utils import embed_chunks, save_embeddings, embeddings_to_tensor
from util.nlp_utils import sentencize, chunk, chunks_to_text_elems
import pandas as pd

from util.vector_search_utils import retrieve_relevant_resources

# Requires !pip install sentence-transformers
from sentence_transformers import SentenceTransformer

min_token_length = 30
#query = "Who is Trey Grainger?"
query = "What is RAG?"

st.write("Initializing models")

# shared state
nlp = spacy.load("en_core_web_sm") #English()

# Add a sentencizer pipeline, see https://spacy.io/api/sentencizer/
nlp.add_pipe("sentencizer")

embedding_model_cpu = SentenceTransformer(model_name_or_path="models/models--sentence-transformers--all-mpnet-base-v2/snapshots/84f2bcc00d77236f9e89c8a360a00fb1139bf47d",
                                      device="cpu") # choose the device to load the model to (note: GPU will often be *much* faster than CPU)


st.write("Done")

st.title('PDF RAG (Retrieval Augmented Generation) Demo')

uploaded_file = st.file_uploader(
    label="Upload a pdf",
    help="Upload a pdf file to chat to it with RAG",
    type='pdf'
)

print(f"uploaded file: {uploaded_file}")

if uploaded_file is not None:
    print(f"Uploaded file: {uploaded_file}")
    with st.expander("Preprocessing"):
        st.write("Reading pdf")
        pages_and_texts = pdf_utils.open_and_read_pdf(uploaded_file)
        st.write("Done")
        # print(pages_and_texts[:2])
        # extract sentences
        st.write("Extracting sentences")
        sentencize(pages_and_texts, nlp)
        st.write("Done")
        # chunk
        st.write("Chunking")
        chunk(pages_and_texts)
        # chunks to text elems
        pages_and_chunks = chunks_to_text_elems(pages_and_texts)
        st.write("Loading to a DataFrame")
        df = pd.DataFrame(pages_and_chunks)
        # Let's filter our DataFrame/list of dictionaries to only include chunks with over 30 tokens in length
        pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
        st.write("Embedding")
        embed_chunks(pages_and_chunks_over_min_token_len, embedding_model_cpu)
        filename = save_embeddings(pages_and_chunks_over_min_token_len)
        st.write("Loading embeddings to tensor")
        tensor = embeddings_to_tensor(filename)

    st.write("vector search")
    scores, indices = retrieve_relevant_resources(query, tensor, embedding_model_cpu, st)
    st.write(f"Query: {query}")
    with st.expander("Results"):
        # Loop through zipped together scores and indicies
        for score, index in zip(scores, indices):
            st.write(f"Score: {score:.4f}")
            # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
            st.write(pages_and_chunks[index]["sentence_chunk"])
            # Print the page number too so we can reference the textbook further and check the results
            st.write(f"Page number: {pages_and_chunks[index]['page_number']}")
