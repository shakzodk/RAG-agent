#!/usr/bin/env python

# RagAiAgent - (c) Eric Dodémont, 2024.

"""
This function runs the frontend web interface.
"""

import streamlit as st
from langchain_core.messages.human import HumanMessage
import uuid
import asyncio

from modules.assistant_backend import instanciate_ai_assistant_graph_agent
from config.config import *


# Function defined in two files: should be moved in a module
def reset_conversation():
    """
    Reset the conversation: clear the chat history and clear the screen.
    """

    st.session_state.messages = []
    st.session_state.threadId = {"configurable": {"thread_id": uuid.uuid4()}}


def assistant_frontend():
    """
    Everything related to Streamlit for the main page (about & chat windows) and connection with the Langchain backend.
    """

    st.set_page_config(page_title=ASSISTANT_NAME, page_icon=ASSISTANT_ICON)
    
    # Initialize chat history (messages) for Streamlit and Langgraph (thread_id)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.threadId = {"configurable": {"thread_id": uuid.uuid4()}}

    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL

    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMPERATURE

    if "password_ok" not in st.session_state:
        st.session_state.password_ok = False

    if "input_password" not in st.session_state:
        st.session_state.input_password = ""

    # Retrieve and generate

    ai_assistant_graph_agent = instanciate_ai_assistant_graph_agent(st.session_state.model, st.session_state.temperature)

    # Write the mermaid graph in the graph.txt file (to be displayed in https://mermaid.live/)
    with open("graph.txt", "w") as f:
        f.write(ai_assistant_graph_agent.get_graph().draw_mermaid())    
    f.close()

    # # # # # # # #
    # Main window #
    # # # # # # # #

    st.image(LOGO_PATH, use_column_width=True)

    st.markdown(f"## {ASSISTANT_NAME}")
    st.caption("💬 A chatbot powered by Langchain, Langgraph and Streamlit")

    # # # # # # # # # # # # # #
    # Side bar window (About) #
    # # # # # # # # # # # # # #

    with st.sidebar:

        st.write(f"Model: {st.session_state.model} ({st.session_state.temperature})")
        st.write(ABOUT_TEXT)
        st.write(SIDEBAR_FOOTER)

    # # # # # # # # # # # #
    # Chat message window #
    # # # # # # # # # # # #

    with st.chat_message("assistant"):
        st.write(HELLO_MESSAGE)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if question := st.chat_input(USER_PROMPT):
        # Display user message in chat message container
        st.chat_message("user").markdown(question)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        try:

            # Call the agent

            if st.session_state.model == ANTHROPIC_MENU:

                # Without streaming the answer (exception because there is a bug in anthropic async)

                # Anthropic: if streaming (async / events), the answer is a list of 
                # dictionaries (NOK), in place of a string (OK).

                answer_container = st.empty()        
                response = ai_assistant_graph_agent.invoke({"messages": [HumanMessage(content=question)]}, config=st.session_state.threadId)
                answer = response["messages"][-1].content
                answer_container.write(answer)

            else:

                # With streaming the answer
                
                async def agent_answer(question):
                    # invoke (sync) --> stream (sync stream invoke) --> astream_events (async stream invoke)
                    answer = ""
                    answer_container = st.empty()
                    async for event in ai_assistant_graph_agent.astream_events({"messages": [HumanMessage(content=question)]}, config=st.session_state.threadId, version="v2"):
                        kind = event["event"]
                        if kind == "on_chat_model_stream":
                            answer_token = event["data"]["chunk"].content
                            if answer_token:
                                answer = answer + answer_token
                                answer_container.write(answer)
                    return(answer)

                async def call_agent_answer(question):
                    answer = await agent_answer(question)
                    return(answer)

                answer = asyncio.run(call_agent_answer(question))

        except Exception as e:
            st.write("Error: Cannot invoke/stream the agent!")
            st.write(f"Error: {e}")

        # Add Answer to chat history for Streamlit (messages)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Clear the conversation
        st.button(NEW_CHAT_MESSAGE, on_click=reset_conversation)
