import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import Anthropic
from langchain.chains import RetrievalQA
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import google.generativeai as genai
import glob
import time

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PDF_DIRECTORY = "split_sections"
DB_DIRECTORY = "vectorstore"
FAISS_INDEX_FILE = os.path.join(DB_DIRECTORY, "faiss_index.pkl")

# Initialize embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

embeddings = get_embeddings()


@st.cache_data
def load_documents(_pdf_directory):
    documents = []
    for pdf_file in glob.glob(f"{_pdf_directory}/*.pdf"):
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            documents.extend(docs)
            #st.write(f"Loaded {len(docs)} pages from {pdf_file}")
        except Exception as e:
            st.error(f"Error loading {pdf_file}: {str(e)}")
    st.write(f"Total documents loaded: {len(documents)}")
    return documents

@st.cache_data
def chunk_documents(_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for doc in _documents:
        doc_chunks = text_splitter.split_documents([doc])
        chunks.extend(doc_chunks)
    st.write(f"Total chunks created: {len(chunks)}")
    return chunks
@st.cache_resource
def create_or_load_vectorstore(_chunks):
    if os.path.exists(FAISS_INDEX_FILE):
        try:
            vectorstore = FAISS.load_local(DB_DIRECTORY, embeddings, "faiss_index")
            # Verify the dimension of the loaded index
            sample_embedding = embeddings.embed_query("Sample text")
            if vectorstore.index.d != len(sample_embedding):
                st.error(f"Dimension mismatch: Index dimension {vectorstore.index.d}, Current embedding dimension {len(sample_embedding)}")
                return None, 0
            return vectorstore, len(_chunks)
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return None, 0
    else:
        try:
            vectorstore = FAISS.from_documents(_chunks, embeddings)
            os.makedirs(DB_DIRECTORY, exist_ok=True)
            vectorstore.save_local(DB_DIRECTORY, "faiss_index")
            return vectorstore, len(_chunks)
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None, 0

def get_llm(model_name, temperature):
    try:
        if model_name == "OpenAI":
            if not OPENAI_API_KEY:
                st.error("OpenAI API key is missing. Please check your .env file.")
                return None
            return ChatOpenAI(temperature=temperature)
        elif model_name == "Claude":
            if not ANTHROPIC_API_KEY:
                st.error("Anthropic API key is missing. Please check your .env file.")
                return None
            return Anthropic(temperature=temperature)
        elif model_name == "Gemini":
            if not GOOGLE_API_KEY:
                st.error("Google API key is missing. Please check your .env file.")
                return None
            genai.configure(api_key=GOOGLE_API_KEY)
            return GooglePalm(temperature=temperature)
        else:
            st.error(f"Unknown model: {model_name}")
            return None
    except Exception as e:
        st.error(f"Error initializing {model_name} model: {str(e)}")
        return None

def get_prompt_template():
    template = """You are a legal assistant analyzing court judgments. 
    Given the following extracted parts of a legal document and a question, create a comprehensive answer.
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.

    Context:
    {context}

    Question: {question}

    Please provide a detailed analysis based on the given context and question:
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])

def main():
    st.title("Legal Document Analysis System")

    if st.button("Rebuild Vector Store"):
        if os.path.exists(DB_DIRECTORY):
            import shutil
            shutil.rmtree(DB_DIRECTORY)
        st.session_state.pop('vectorstore', None)
        st.success("Vector store deleted. It will be rebuilt on the next query.")
        
    # Check if vectorstore is properly initialized
    if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None:
        st.warning("Vector store is not initialized. Initializing now...")
        documents = load_documents(PDF_DIRECTORY)
        if not documents:
            st.error("No documents were loaded. Please check your PDF directory.")
            return
        chunks = chunk_documents(documents)
        st.session_state.vectorstore, total_chunks = create_or_load_vectorstore(chunks)
        if not st.session_state.vectorstore:
            st.error("Failed to create or load vector store.")
            return
        st.success(f"Vector store initialized with {total_chunks} chunks.")

    # Debug information
    st.write(f"Vector store type: {type(st.session_state.vectorstore)}")
    st.write(f"Number of documents in vector store: {len(st.session_state.vectorstore.docstore._dict)}")

    # Model selection
    model_name = st.selectbox("Select Language Model", ["OpenAI", "Claude", "Gemini"])

    # Advanced options
    with st.expander("Advanced Options"):
        top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=4)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    # User query input
    query = st.text_input("Enter your legal query:")

    if query:
        llm = get_llm(model_name, temperature)
        if llm:
            with st.spinner("Analyzing legal documents..."):
                try:
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
                    
                    # Debug information
                    query_embedding = embeddings.embed_query(query)
                    st.write(f"Query embedding dimension: {len(query_embedding)}")
                    st.write(f"Index dimension: {st.session_state.vectorstore.index.d}")
                    
                    retrieved_docs = retriever.get_relevant_documents(query)
                    st.write(f"Number of retrieved documents: {len(retrieved_docs)}")
                    
                    prompt = get_prompt_template()
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": prompt}
                    )
                    
                    # Use StreamlitCallbackHandler for streaming output
                    st_callback = StreamlitCallbackHandler(st.empty())
                    
                    start_time = time.time()
                    response = qa_chain({"query": query}, callbacks=[st_callback])
                    end_time = time.time()

                    if response['result']:
                        st.write("Legal Analysis:", response['result'])
                        st.write(f"Time taken: {end_time - start_time:.2f} seconds")

                        # Display source documents
                        if response['source_documents']:
                            st.subheader("Relevant Legal Sources:")
                            for i, doc in enumerate(response['source_documents']):
                                st.write(f"Source {i+1}:")
                                st.write(doc.page_content)
                                st.write("Metadata:", doc.metadata)
                                st.write("---")
                        else:
                            st.warning("No relevant legal sources were found for this query.")
                    else:
                        st.warning("The model couldn't generate an analysis based on the available legal information.")

                except AssertionError as e:
                    st.error("Dimension mismatch between query and index. Please recreate the vector store.")
                    st.error("You may need to delete the existing vector store files and rerun the application.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    st.error("Please check the console for more detailed error information.")
                    import traceback
                    traceback.print_exc()  # This will print the full traceback to the console


if __name__ == "__main__":
    main()