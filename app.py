###########################
# HITL
###########################
from backend import chatbot,get_all_threads,ingest_rag_documents
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage,BaseMessage
from langgraph.types import Command,interrupt
import streamlit as st
import uuid
import os
import tempfile
import sqlite3
import logging
#######################################Function Definitions###############################
# generate unique thread id
def generate_thread_id():
    return str(uuid.uuid4())

# add new thread id to list if not present
def add_thread(thread_id):
    # add thread id only if it is not already in list
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def reset_chat():
    # generate new thread id for new chat
    st.session_state['thread_id']=generate_thread_id()

    # create new session state to store message history
    st.session_state['message_history']=[]

    # Clear any pending human approval request
    st.session_state["pending_hitl"] = None

    # add thread id to chat_threads in session state
    add_thread(st.session_state['thread_id'])

def load_conversation(thread_id):
    state=chatbot.get_state(config={"configurable":{"thread_id":thread_id}})
    # print(state)
    return state.values.get("messages", [])

#######################################Function Definitions###############################

# ========================= HITL helper functions =========================

def get_pending_interrupt(thread_id):
    """
    Return the first unresolved LangGraph interrupt for a thread.
    Returns:
        The pending Interrupt object, or None.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        # Read the current checkpoint state
        state_snapshot = chatbot.get_state(config)
        # Some LangGraph versions expose interrupts directly
        direct_interrupts = getattr(state_snapshot,"interrupts",()) or ()

        if direct_interrupts:
            return direct_interrupts[0]

        # Other LangGraph versions store interrupts inside tasks
        tasks = getattr(state_snapshot,"tasks",()) or ()

        for task in tasks:
            task_interrupts = getattr(task,"interrupts",()) or ()
            if task_interrupts:
                return task_interrupts[0]

    except sqlite3.Error as error:
        logging.exception("SQLite database error")
        raise RuntimeError("Database error") from error

    except ValueError as error:
        logging.exception("Invalid configuration")
        raise RuntimeError("Invalid configuration") from error

    except Exception as error:
        logging.exception("Unexpected error")
        raise RuntimeError("Unexpected error") from error

    return None

def save_pending_interrupt(thread_id, interrupt_object):
    """
    Save the pending interrupt information inside Streamlit state.
    """
    st.session_state["pending_hitl"] = {"thread_id": thread_id,"prompt": str(interrupt_object.value)}


def sync_pending_interrupt(thread_id):
    """
    Synchronize Streamlit HITL state with the LangGraph checkpoint.
    This allows a pending approval request to reappear after:
    - a Streamlit rerun
    - a browser refresh
    - switching between conversations
    """
    try:
        pending_interrupt = get_pending_interrupt(thread_id)
        if pending_interrupt is not None:
            save_pending_interrupt(thread_id,pending_interrupt)
        else:
            current_pending = st.session_state.get("pending_hitl")
            if (current_pending is not None and current_pending.get("thread_id") == thread_id):
                st.session_state["pending_hitl"] = None
    except RuntimeError as error:
        logging.exception("Failed to synchronize pending interrupt")
        st.error(str(error))

def resume_hitl_execution(decision):
    """
    Resume an interrupted LangGraph execution.
    Args:
        decision:
            "yes" approves the stock purchase.
            "no" rejects the stock purchase.
    """
    pending_hitl = st.session_state.get("pending_hitl")
    if not pending_hitl:
        st.warning("There is no pending action to approve or reject.")
        return
    # Get the thread that originally triggered the interrupt
    interrupted_thread_id = pending_hitl["thread_id"]

    # The same thread ID must be used when resuming
    resume_config = {"configurable": {"thread_id": interrupted_thread_id},
        "metadata": {"thread_id": interrupted_thread_id},
        "run_name": "hitl_resume_trace",}

    try:
        # Display the resumed response
        with st.chat_message("assistant"):
            status_holder = {"box": st.status("🔄 Resuming the requested action...",expanded=True)}

            def resumed_ai_only_stream():
                # Resume the graph with the human decision
                for message_chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):

                    # Update tool execution status
                    if isinstance(message_chunk,ToolMessage):
                        tool_name = getattr(message_chunk,"name","tool")

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
                        status_holder["box"].write(f"**{tool_name}** returned:\n\n {message_chunk.content}")

                    # Stream only assistant-generated text
                    if isinstance(message_chunk,AIMessage):
                        if message_chunk.text:
                            yield message_chunk.text

            # Display the streamed final answer
            resumed_ai_message = st.write_stream(resumed_ai_only_stream())

            # Check whether another interrupt occurred
            next_interrupt = get_pending_interrupt(interrupted_thread_id)

            if next_interrupt is not None:

                save_pending_interrupt(interrupted_thread_id,next_interrupt)

                status_holder["box"].update(
                    label="⚠️ Another approval is required",
                    state="complete",
                    expanded=False
                )

            else:

                # No more pending approval
                st.session_state["pending_hitl"] = None

                status_holder["box"].update(
                    label="ℹ️No more pending interrupts!",
                    state="complete",
                    expanded=False
                )

        # Save the assistant response in Streamlit UI history
        if resumed_ai_message:

            st.session_state["message_history"].append({
                "role": "assistant",
                "content": resumed_ai_message
            })

    except Exception as error:
        st.error(f"Could not resume the requested action: {error}")
        return

    # Rerun so the response appears in normal chat order
    st.rerun()

#############################Session State Initialization#################################
st.title('🤖Agentic Chatbot with Langgraph')

# create message_history in session state
if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]

# create a list of chat threads in session state
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads']=get_all_threads()

# create a thread id in session state
if 'thread_id' not in st.session_state:
    st.session_state['thread_id']=generate_thread_id()

add_thread(st.session_state['thread_id'])

# Store the currently pending human approval request
if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

# Recover pending approval after page refresh or rerun
sync_pending_interrupt(st.session_state["thread_id"])
#############################Session State Initialization#################################

############################SIDE BAR########################################
st.sidebar.title("My Conversations")

if st.sidebar.button("New chat"):

    # Reset current chat and create a new thread
    reset_chat()

    # Rerun the streamlit interface
    st.rerun()

# Loads and makes the threads in the sidebar visible
# Make latest chat visible on top
for thread_id in st.session_state['chat_threads'][::-1]:
    # display threads stored in chat_threads

    # creates one button for each conversation
    # on selecting one of the various threads:
    if st.sidebar.button(str(thread_id),key=thread_id):

        # thread id changes to selected button's thread id
        st.session_state['thread_id']=thread_id

        # load the conversations in that specific thread
        messages=load_conversation(thread_id)

        # store the selected thread id messages in current session state 'message_history'
        temp_messages=[]
        for message in messages:
            if isinstance(message,HumanMessage):
                role='user'
            elif isinstance(message,AIMessage):
                # When Toolcalling happens AIMessage stores an empty string, so skip such content
                if not message.text.strip():
                    continue
                role='assistant'
            elif isinstance(message, ToolMessage):
                temp_messages.append({"role": "tool","tool_name": getattr(message, "name", "tool"),"content": message.content})
                continue
            else:
                continue
            temp_messages.append({"role":role,"content":message.text})
        st.session_state['message_history']=temp_messages
        # rerun
        st.rerun()

############################SIDE BAR########################################

############################DISPLAY CHAT########################################

# Loads and makes the conversation in the chat visible
for message in st.session_state["message_history"]:
    if message["role"] == "tool":
        with st.status(f"🔧 `{message['tool_name']}` returned",expanded=False) as tool_box:
            tool_box.write(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# ========================= HITL approval interface =========================

# Get the currently pending approval request
pending_hitl = st.session_state.get("pending_hitl")

# Check whether the pending approval belongs to
# the currently selected conversation
current_thread_has_pending_hitl = (pending_hitl is not None and pending_hitl.get("thread_id")== st.session_state["thread_id"])

# Display approval controls
if current_thread_has_pending_hitl:
    st.warning("🧑 Human approval required\n\n"
               f"{pending_hitl['prompt']}")
    approve_column, reject_column = st.columns(2)
    # Approve button
    with approve_column:
        if st.button(
            "✅ Approve Purchase",
            key=f"approve_{st.session_state['thread_id']}",
            type="primary",use_container_width=True):
            # Send "yes" back to interrupt()
            resume_hitl_execution("yes")

    # Reject button
    with reject_column:
        if st.button("❌ Reject Purchase",key=f"reject_{st.session_state['thread_id']}",use_container_width=True):
            # Send "no" back to interrupt()
            resume_hitl_execution("no")

# Keep st.chat_input directly in the main body.
# This keeps it fixed at the bottom of the screen.

# accept_file=True adds the attachment button inside the chat input.
# file_type=["pdf"] allows PDF files only.
submission = st.chat_input("Type here",accept_file="multiple",file_type=["pdf"],
                           # Disable input while waiting for human approval
                           disabled=current_thread_has_pending_hitl)
user_input = None
# Process the submitted text and PDF
if submission:
    # Get the text entered by the user
    user_input = submission.text

    # Get the uploaded files
    # This is always a list when accept_file is enabled
    uploaded_files = submission.files

    # Process the uploaded PDF if one was attached
    if uploaded_files:
        # Store the temporary file path
        try:
            tfp_list=[]
            # Save the uploaded PDF as a temporary local file
            for file in uploaded_files:
                temporary_file_path = None
                with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as temporary_file:
                    temporary_file.write(file.getvalue())
                    temporary_file_path = temporary_file.name
                    tfp_list.append(temporary_file_path)

                # Call the existing backend RAG ingestion function
                with st.spinner(f"Processing {file.name}..."):
                    ingest_rag_documents(temporary_file_path,file.name,st.session_state['thread_id'])

                # Display PDF processing confirmation
                st.toast(f"{file.name} processed successfully.",icon="✅")

                if not user_input:
                    user_input=f"Uploaded document: \n {file.name}"
                else:
                    user_input+=f"\n\n Uploaded document: \n {file.name}"

        except Exception as error:
            user_input=f"Couldn't upload the file"
            # Display PDF processing error
            st.error(f"PDF processing failed: {error}")

        finally:
            # Delete the temporary PDF after indexing
            for tfp in tfp_list:
                if (tfp and os.path.exists(tfp)):
                    os.remove(tfp)

# user_input=st.chat_input('Type here')
if user_input:
    # add it to session state
    st.session_state['message_history'].append({'role':'user','content':user_input})

    # display user_input
    with st.chat_message('user'):
        st.write(user_input)

    # passes the current thread id for accessing old conversations in that thread
    # Langgraph uses this to save and retrieve conversation
    config={'configurable':{"thread_id":st.session_state['thread_id']},
            'metadata':{"thread_id":st.session_state['thread_id']},
            'run_name':'chat_trace'}

    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
                    # status_holder["box"].write(f"**{tool_name}** returned:")
                    # status_holder["box"].write(message_chunk.content)
                    status_holder["box"].write(f"**{tool_name}** returned:\n\n {message_chunk.content}")

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    if message_chunk.text:
                        yield message_chunk.text

            # interrupt() pauses the graph without returning a completed ToolMessage.
            # So inspect the saved checkpoint after streaming ends.
            pending_interrupt = get_pending_interrupt(st.session_state["thread_id"])

            if pending_interrupt is not None:
                # Save the interrupt so the approval buttons can be displayed
                save_pending_interrupt(st.session_state["thread_id"],pending_interrupt)

                yield ("\n\n⚠️ This stock purchase requires your approval. "
                       "Use the Approve Purchase or Reject Purchase button below.")

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            # Check whether execution is waiting for approval
            if get_pending_interrupt(st.session_state["thread_id"]) is not None:
                status_holder["box"].update(
                    label="⏸️ Waiting for human approval", state="complete", expanded=False
                )
            else:
                status_holder["box"].update(
                    label="✅ Tool finished", state="complete", expanded=False
                )

    # Save the complete assistant response in Streamlit session state
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })

############################DISPLAY CHAT########################################

# ========================= HITL ADDED =========================

    # Approval controls are rendered earlier in the script.
    # Rerun so they appear immediately after interrupt().
    if (st.session_state.get("pending_hitl") is not None
    and st.session_state["pending_hitl"].get("thread_id")== st.session_state["thread_id"]):
        st.rerun()
# =============================================================