from chromadb_helper import DatabaseAccessor 
from emb_fu_helper import FastTextEmbeddingFunction
import fasttext
import spacy
import streamlit as st
from prompt_helper import BNS_LLAMA_CustomModel, BNSPrompter

def load_model_and_db(cname):
    model = fasttext.load_model("models/cc.en.300.bin")
    emb_fn = FastTextEmbeddingFunction(model)
    DBA = DatabaseAccessor(cname, emb_fn)
    collection = DBA.connect()
    return collection

def query_collection(collection, events):
    with st.spinner('Processing...'):
        results = collection.query(
            query_texts=events,
            n_results=10
        )
        sections = ' '.join(results)

        # call the llama model with the text and results collected

        bns_prompt = BNSPrompter().get_prompt(events,sections)

        cm = BNS_LLAMA_CustomModel(model="llama2")
        result_final = cm.call(bns_prompt)

    return result_final

def preprocess(txt):
    txt = txt.lower().strip()
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(txt)
    clean_list = [token.text for token in doc if not token.is_stop]
    txt = ' '.join(clean_list)
    return txt

# Load model and database
collection_name = 'bns_collection_boss'
collection = load_model_and_db(collection_name)

# Streamlit app UI
text = st.text_area("Enter the incident")

# processing the text

if len(text) > 0:

    queries = [preprocess(text)]
    
    results = query_collection(collection, queries)

    if results["done"]:
        st.write(results["response"])
    else:
        st.write("No results found.")
