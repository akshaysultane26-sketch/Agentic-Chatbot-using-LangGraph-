from backend import chatbot, get_all_threads, ingest_rag_document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.types import Command
import streamlit as st
import uuid
import os


def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    st.session_state["pending_interrupt"] = None
    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


st.title("Agentic Chatbot with Human-in-the-Loop")


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

# Holds the interrupt message text while we wait for the user's
# approve/reject decision. None means nothing is pending.
if "pending_interrupt" not in st.session_state:
    st.session_state["pending_interrupt"] = None

add_thread(st.session_state["thread_id"])


st.sidebar.title("My Conversations")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id), key=thread_id):
        st.session_state["thread_id"] = thread_id
        st.session_state["pending_interrupt"] = None
        messages = load_conversation(thread_id)

        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue

            temp_messages.append({"role": role, "content": message.content})

        st.session_state["message_history"] = temp_messages
        st.rerun()


for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])


CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}


def run_graph(payload):
    """Invoke the graph and handle the result — either a normal answer
    or a pause waiting for human approval."""
    result = chatbot.invoke(payload, config=CONFIG)

    if "__interrupt__" in result:
        # The graph paused — show the approval question instead of
        # a normal AI answer
        interrupt_obj = result["__interrupt__"][0]
        st.session_state["pending_interrupt"] = interrupt_obj.value
        st.rerun()
    else:
        ai_message = result["messages"][-1].content
        st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
        st.rerun()


# ==================== Pending approval UI ====================
if st.session_state["pending_interrupt"] is not None:
    with st.chat_message("assistant"):
        st.warning(f"⏸️ **Approval needed:** {st.session_state['pending_interrupt']}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Approve", use_container_width=True):
                st.session_state["message_history"].append(
                    {"role": "user", "content": "(approved)"}
                )
                st.session_state["pending_interrupt"] = None
                run_graph(Command(resume="yes"))

        with col2:
            if st.button("❌ Reject", use_container_width=True):
                st.session_state["message_history"].append(
                    {"role": "user", "content": "(rejected)"}
                )
                st.session_state["pending_interrupt"] = None
                run_graph(Command(resume="no"))


# ==================== Normal chat input ====================
else:
    chat_value = st.chat_input(
        "Type here",
        accept_file=True,
        file_type=["pdf"],
    )

    user_input = None
    uploaded_file = None

    if chat_value:
        user_input = chat_value.text
        if chat_value.files:
            uploaded_file = chat_value.files[0]

    if uploaded_file is not None:
        save_path = os.path.join(os.getcwd(), uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(f"Reading and indexing {uploaded_file.name} ..."):
            ingest_rag_document(save_path)

        st.success(f"📄 {uploaded_file.name} indexed! Ask me anything about it.")

        if not user_input:
            user_input = f"I've uploaded {uploaded_file.name}. Please summarize it."

    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.text(user_input)

        with st.spinner("Thinking..."):
            run_graph({"messages": [HumanMessage(content=user_input)]})
