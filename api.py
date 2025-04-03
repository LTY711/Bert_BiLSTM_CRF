import streamlit as st
import json
import requests
import time

# TODO 对话框
def main():
    st.title("欢迎访问NER智能识别系统")
    # 初始化会话状态，没有这新建
    if 'history' not in st.session_state:
        st.session_state.history = []

    # 显示对话历史
    for chat in st.session_state.history:
        # if chat['role'] == "user":
        #     with st.chat_message(chat["user"]):
        #         st.markdown(chat['content'])
        # else:
        #     with st.chat_message(chat["assistant"]):
        #         st.markdown(chat['content'])
        with st.chat_message(chat["role"]):  # 使用 'role' 键
            st.markdown(chat["content"])

    # 接收用户输入
    if user_input := st.chat_input("Chat with 小羿：请输入您要识别的句子"):
        # 显示用户输入
        with st.chat_message("user"):
            st.markdown(user_input)

        # 将用户的输入加入历史
        st.session_state.history.append({"role": "user", "content": user_input})

        # 调用 NER 模型并获取回复
        with st.spinner("小羿正在思考..."):
            response = ner_predict(user_input)
            if response is None or response == "":
                response = "抱歉，我暂时无法理解您的问题。"
        # 在页面上显示模型生成的回复
        with st.chat_message("assistant"):
            st.write_stream(stream_response(response))
            # st.markdown(response)

        # 将模型的输出加入到历史信息中
        st.session_state.history.append({"role": "assistant", "content": response})

        # 只保留十轮对话，这个可根据自己的情况设定
        if len(st.session_state.history) > 20:
            st.session_state.history = st.session_state.history[-20:]

def stream_response(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.5)

def ner_predict(text):
    url = 'http://192.168.33.120:7009/service/api/bert_bilstm_crf'
    data = {"text": text}
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    r = requests.post(url=url, data=json.dumps(data), headers=headers)
    if r.status_code != 200:
        print('模型调用有误')
        return None
    result = json.loads(r.text)
    return "您好，我的识别结果是：" + result['result']


if __name__ == '__main__':
    main()