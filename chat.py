import streamlit as st


from dotenv import load_dotenv


from llm import get_ai_response

st.set_page_config(page_title="Stock UPT", page_icon="📈")

st.title("📈 Stock UPT")
st.caption("원하시는 주식 종목(키워드)을 입력해주세요! 예: 삼성전자 / 하이브 / 2차전지")

load_dotenv()

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])



if user_question := st.chat_input(placeholder="예) 삼성전자 전망 알려줘 / 하이브 분석"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        ai_response = get_ai_response(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})
