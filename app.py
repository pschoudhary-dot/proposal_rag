import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import time
import tempfile
from pathlib import Path
from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from uuid import uuid4

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please add the OPENAI_API_KEY environment variable.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.chat_history = [] 
    st.session_state.parsed_docs = {} 

# Function to sign out
def sign_out():
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.parsed_docs = {}
    supabase.auth.sign_out()
    st.rerun()

# Sidebar for authentication
with st.sidebar:
    if not st.session_state.authenticated:
        st.title("User Authentication")
        auth_option = st.radio("Choose an option", ("Login", "Sign Up"))

        # Sign Up Section
        if auth_option == "Sign Up":
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Sign Up"):
                if not email or not password:
                    st.error("Please provide both email and password.")
                else:
                    try:
                        response = supabase.auth.sign_up({
                            'email': email,
                            'password': password
                        })
                        if response.user:
                            st.success("Sign Up successful! Please check your email for verification.")
                        else:
                            st.error(f"Error: {response.error_message}")
                    except Exception as e:
                        st.error(f"An error occurred during sign-up: {e}")

        # Login Section
        elif auth_option == "Login":
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if not email or not password:
                    st.error("Please provide both email and password.")
                else:
                    try:
                        response = supabase.auth.sign_in_with_password({
                            'email': email,
                            'password': password
                        })
                        if response.user:
                            user_id = response.user.id
                            user_email = response.user.email
                            
                            # Check if the user exists in public.users
                            user_check = supabase.table("users").select("id").eq("id", user_id).execute()
                            if not user_check.data:
                                # Insert the user into public.users
                                insert_response = supabase.table("users").insert({
                                    "id": user_id,
                                    "email": user_email,
                                }).execute()
                                
                                if not insert_response.data:
                                    st.error("Failed to create user profile in database.")
                                    st.stop()
                            
                            st.session_state.authenticated = True
                            st.session_state.user = response.user

                            # Fetch pre-existing parsed documents for the user
                            try:
                                parsed_response = supabase.table("users_documents").select(
                                    "name, processed_file"
                                ).eq("user_id", user_id).eq("processing_status", "completed").execute()
                                parsed_files = parsed_response.data
                                if parsed_files:
                                    for file in parsed_files:
                                        st.session_state.parsed_docs[file["name"]] = file["processed_file"]
                            except Exception as e:
                                st.error(f"Error fetching pre-existing parsed documents: {e}")

                            st.rerun()  # Refresh the app to update the UI
                        else:
                            st.error(f"Error: {response.error_message}")
                    except Exception as e:
                        st.error(f"An error occurred during login: {e}")
    else:
        st.success(f"Logged in as {st.session_state.user.email}")
        if st.button("Sign Out"):
            sign_out()

# Main content
if st.session_state.authenticated:
    user_id = st.session_state.user.id  # Get the current user's ID
    
    # Create two tabs: Chat and Previews
    tab1, tab2 = st.tabs(["Chat", "Previews"])
    
    with tab1:  # Chat Tab
        st.title("Chat with Your Documents")
        
        # Initialize OpenAI model
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Create a prompt template for the chat
        prompt_template = ChatPromptTemplate([
            ("system", "You are a helpful assistant that answers questions based on the provided document context."),
            ("human", "{question}"),
            MessagesPlaceholder(variable_name="context")  # Placeholder for document context
        ])
        
        # Chain the prompt template with the model
        chain = LLMChain(llm=llm, prompt=prompt_template)
        
        # Create a scrollable container for chat history
        chat_container = st.container(height=400)  # Height in pixels, adjust as needed
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # User input for chat (automatically positioned at the bottom)
        query = st.chat_input("Ask a question about your documents:")
        
        if query:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Perform similarity search to retrieve relevant document chunks 
            try:
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                query_embedding = embeddings.embed_query(query)
                
                # Query the database for similar chunks
                similarity_response = supabase.rpc("match_documents", {"query_embedding": query_embedding}).execute()
                similar_chunks = similarity_response.data
                
                # Combine retrieved chunks into a single context string
                context = "\n".join([chunk["content"] for chunk in similar_chunks])
                
                # Invoke the chain with the query and context
                response = chain.invoke({"question": query, "context": [{"role": "system", "content": context}]})
                
                # Add AI response to chat history
                ai_response = response['text']
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                
                # Display the AI's response
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
            except Exception as e:
                # Improved error handling with user-friendly message
                st.error("An error occurred while processing your query. Please try again or ensure your documents are processed correctly.")
    
    with tab2:  # Previews Tab
        st.title("Preview Tab")
        
        # Check if there are any parsed documents before creating tabs
        if st.session_state.parsed_docs:
            # Use tabs instead of nested expanders
            tabs = st.tabs([name for name in st.session_state.parsed_docs.keys()])
            for tab, (name, content) in zip(tabs, st.session_state.parsed_docs.items()):
                with tab:
                    st.subheader(f"Preview: {name}")
                    st.markdown(content)
                    
                    # Download button for the parsed Markdown file with unique key
                    st.download_button(
                        label="Download Parsed Markdown",
                        data=content,
                        file_name=f"parsed_{name}.md",
                        mime="text/markdown",
                        key=f"download_{name}"  # Unique key based on file name
                    )
        else:
            st.info("No documents have been parsed yet. Please upload and process a document first.")

    with st.sidebar:
        st.subheader("Your Projects")
        
        # Fetch projects associated with the current user
        try:
            # Query the `projects` table for the current user's projects
            projects_response = supabase.table("projects").select("id, name").eq("user_id", user_id).execute()
            projects = projects_response.data
            
            if projects:
                project_options = {project["name"]: project["id"] for project in projects}
                selected_project_name = st.selectbox("Select a Project", list(project_options.keys()))
                selected_project_id = project_options[selected_project_name]
            else:
                selected_project_id = None
                st.write("No projects found. Please create a new project.")
        except Exception as e:
            st.error(f"Error fetching projects: {e}")

        # Create a new project
        new_project_name = st.text_input("Create a New Project")
        if st.button("Create Project"):
            if not new_project_name:
                st.error("Please provide a project name.")
            else:
                try:
                    # Ensure the user is authenticated and has a valid user_id
                    if not st.session_state.authenticated or not st.session_state.user:
                        st.error("You must be logged in to create a project.")
                        st.stop()

                    user_id = st.session_state.user.id  # Get the current user's ID

                    # Verify that the user_id exists in the `users` table
                    user_check_response = supabase.table("users").select("id").eq("id", user_id).execute()
                    if not user_check_response.data:
                        st.error("User ID does not exist in the database. Please log out and log back in.")
                        st.stop()

                    # Insert the new project into the `projects` table
                    project_response = supabase.table("projects").insert({
                        "user_id": user_id,
                        "name": new_project_name
                    }).execute()

                    if project_response:
                        st.success(f"Project '{new_project_name}' created successfully.")
                        st.rerun()  # Refresh to update the project list
                    else:
                        st.error("Failed to create project.")
                except Exception as e:
                    st.error(f"Error creating project: {e}")
        
        # Upload new document
        allowed_file_types = ["xlsx", "csv", "pptx", "pdf", "docx", "txt", "md", "html"]  # Allowed file types
        uploaded_file = st.file_uploader(
            "Select a document to upload",
            type=allowed_file_types,
            help="Allowed file types: Excel (.xlsx), CSV (.csv), PowerPoint (.pptx), PDF (.pdf), Word (.docx), Text (.txt), Markdown (.md), HTML (.html)"
        )
        
        # Store the uploaded file in session state temporarily
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.success(f"File '{uploaded_file.name}' selected. Click 'Upload' to proceed.")
        
        # Upload button
        if "uploaded_file" in st.session_state and st.button("Upload"):
            uploaded_file = st.session_state.uploaded_file
            del st.session_state.uploaded_file  # Clear the temporary file from session state
            
            # Generate a unique file name to avoid duplicates
            file_name = f"{int(time.time())}_{uploaded_file.name}"
            file_data = uploaded_file.getvalue()

            # Upload the document to Supabase Storage
            try:
                storage = supabase.storage.from_("documents")  # Specify the bucket
                upload_response = storage.upload(file_name, file_data)
                
                if upload_response:
                    # Save metadata in the `users_documents` table
                    metadata_response = supabase.table("users_documents").insert({
                        "user_id": user_id,
                        "project_id": selected_project_id,  # Link document to the selected project
                        "name": uploaded_file.name,  # Original file name
                        "file_path": file_name,      # Unique file name in storage
                        "file_size": len(file_data), # File size in bytes
                        "file_type": uploaded_file.type,  # MIME type of the file
                        "processing_status": "pending",   # Initial processing status
                        "processed_file": None           # Placeholder for processed file
                    }).execute()
                    
                    if metadata_response:
                        st.success(f"Document '{uploaded_file.name}' uploaded successfully.")
                    else:
                        st.error("Failed to save document metadata.")
                else:
                    st.error("Failed to upload document.")
            except Exception as e:
                st.error(f"Error uploading document: {e}")

        # Fetch documents associated with the selected project
        if selected_project_id:
            try:
                # Query the `users_documents` table for the selected project's documents
                response = supabase.table("users_documents").select("id, name, file_path, file_size, file_type, processing_status, processed_file").eq("project_id", selected_project_id).execute()
                files = response.data
                
                if files:
                    selected_files = []
                    
                    for file in files:
                        # Add a checkbox for each document
                        is_selected = st.checkbox(f"{file['name']} ({file['file_type']}, {file['file_size']} bytes)", key=f"doc_{file['id']}")
                        if is_selected:
                            selected_files.append(file)
                    
                    # Process button for selected documents
                    if st.button("Process Selected Documents"):
                        if not selected_files:
                            st.warning("No documents selected. Please select at least one document to process.")
                        else:
                            st.info(f"Parsing {len(selected_files)} document(s)...")

                            # Progress bar initialization
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for idx, file in enumerate(selected_files):
                                try:
                                    # Download the file from Supabase storage
                                    storage = supabase.storage.from_("documents")
                                    file_path = file["file_path"]
                                    downloaded_file = storage.download(file_path)
                                    
                                    # Save the file locally
                                    temp_dir = tempfile.mkdtemp()
                                    local_file_path = Path(temp_dir) / file["name"]
                                    with open(local_file_path, "wb") as f:
                                        f.write(downloaded_file)
                                    
                                    # Parse the document based on its file type
                                    markdown_content = ""
                                    if file["file_type"] == "text/plain" or file["file_type"] == "text/markdown":  # Handle .txt and .md files as-is
                                        with open(local_file_path, "r", encoding="utf-8") as f:
                                            markdown_content = f.read()
                                        status_text.text(f"Processed '{file['name']}' successfully (plain text/Markdown).")
                                    else:
                                        # Use Docling's convert() method to parse other file types into Markdown  
                                        converter = DocumentConverter()
                                        conversion_result = converter.convert(local_file_path)
                                        
                                        # Extract Markdown content using export_to_markdown() method
                                        if hasattr(conversion_result, "document") and hasattr(conversion_result.document, "export_to_markdown"):
                                            markdown_content = conversion_result.document.export_to_markdown()
                                        else:
                                            raise ValueError(f"Unsupported ConversionResult format for '{file['name']}'.")

                                        status_text.text(f"Parsed '{file['name']}' successfully.")
                                    
                                    # Update the processing status and save the processed file in the database
                                    update_response = supabase.table("users_documents").update({
                                        "processing_status": "completed",
                                        "processed_file": markdown_content
                                    }).eq("id", file["id"]).execute()
                                    
                                    if update_response:
                                        status_text.text(f"Document '{file['name']}' processed successfully.")
                                    else:
                                        status_text.text(f"Failed to update processing status for '{file['name']}'.")

                                    # Split the content into chunks
                                    text_splitter = RecursiveCharacterTextSplitter(
                                        chunk_size=1000,  # Adjust chunk size as needed  
                                        chunk_overlap=200
                                    )
                                    chunks = text_splitter.split_text(markdown_content)
                                    status_text.text(f"Created {len(chunks)} chunks for '{file['name']}'.")

                                    # Generate embeddings for each chunk
                                    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)  # Pass the API key explicitly  
                                    for i, chunk in enumerate(chunks):
                                        embedding_vector = embeddings.embed_query(chunk)
                                        
                                        # Save the chunk and its embedding in the `document_chunks` table
                                        chunk_id = str(uuid4())  # Generate a unique ID for the chunk
                                        insert_response = supabase.table("document_chunks").insert({
                                            "id": chunk_id,
                                            "document_id": file["id"],
                                            "content": chunk,
                                            "embedding": embedding_vector,
                                            "created_at": "now()"
                                        }).execute()
                                        
                                        if insert_response:
                                            status_text.text(f"Chunk {i+1} saved successfully for '{file['name']}'.")
                                        else:
                                            status_text.text(f"Failed to save chunk {i+1} for '{file['name']}'.")

                                    # Store the parsed content in session state for preview/download
                                    st.session_state.parsed_docs[file["name"]] = markdown_content

                                    # Update progress bar
                                    progress_bar.progress((idx + 1) / len(selected_files))

                                except Exception as e:
                                    st.error(f"Error processing document '{file['name']}': {e}")

                    # Delete button for selected documents
                    if st.button("Delete Selected Documents"):
                        if not selected_files:
                            st.warning("No documents selected. Please select at least one document to delete.")
                        else:
                            # Logic to delete selected documents (implement as needed)
                            pass
                else:
                    st.write("No documents found.")
            except Exception as e:
                st.error(f"Error fetching documents: {e}")
else:
    st.sidebar.info("Please log in to access the app.")