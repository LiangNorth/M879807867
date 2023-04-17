import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import openai
import json
import os
from PIL import Image
try:
    import paddlehub as hub
except ImportError as e:
    print(f"Error importing paddlehub: {e}")
    hub = None
import chardet
import pandas as pd

# 设置页面布局为最大化
st.set_page_config(page_title='情感智能决策引擎系统', layout="wide")

nav_choice = st.sidebar.selectbox(
    "选择你要使用的功能:",
    ("主页面", "产品评论分析", "情感对话机器人", "情感分析文本", "社交媒体评论分析"),
    index=0,  # 设置默认选项为主页面
)

if nav_choice == '主页面':
    # 创建页面标题
    st.title('你好，欢迎使用我们的产品！')

    # 加载图片
    image = Image.open("smartAI/static/woman-7863041.jpg")
    st.image(image, caption='人工智能引领新时代！', use_column_width=True)

    # 显示文字
    st.write("感谢你使用本产品，这是我们的主页面，有待完善")
    st.write("此产品只是一个雏形，还有很多东西等待后续添加")
    st.write("如果你觉得其它页面太大了,你可以在右上角的三个杠里面打开settings,把Wide mode关闭")

    image = Image.open("/smartAI/static/axel-johansson-axjo.gif")
    st.image(image, caption='对你致以最真诚的问候', use_column_width=True)


elif nav_choice == '产品评论分析':

    # 创建页面标题
    st.title('产品评论分析')
    # 加载情感分析模型
    senta = hub.Module(name="senta_bilstm")  # 这个机器人模型是基于百度AI开源的一个预训练模型


    # 定义函数进行用户文本分析

    def sentiment_analysis(lines):
        pos_count = 0
        neg_count = 0
        threshold = 0.2  # 阈值，当积极和消极情绪差异小于该值时，判断为中性情绪

        for line in lines:
            results = senta.sentiment_classify(texts=[line.strip()], use_gpu=False, batch_size=1)
            result = results[0]

            if result['sentiment_key'] == 'positive':
                pos_count += 1
            elif result['sentiment_key'] == 'negative':
                neg_count += 1

        if abs(pos_count - neg_count) < threshold * (pos_count + neg_count):
            return "Neutral"
        elif pos_count > neg_count:
            return "Positive"
        else:
            return "Negative"


    # 定义函数处理上传的文件
    def process_uploaded_file(uploaded_file):
        if uploaded_file is None:
            return None

        # 判断上传文件的格式是否为txt
        if uploaded_file.type == "text/plain":
            # 将上传文件保存到本地
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 判断txt文件的编码格式，如果不是utf-8就转换为utf-8
            with open(uploaded_file.name, 'rb') as f:
                content = f.read()
                encoding = chardet.detect(content)['encoding']
                if encoding.lower() != 'utf-8':
                    with open(uploaded_file.name, 'w', encoding='utf-8') as f:
                        f.write(content.decode(encoding))

            # 读取上传文件并进行情感分析
            with open(uploaded_file.name, "r", encoding="utf-8") as f:
                lines = f.readlines()
                emotions = [sentiment_analysis([line.strip()]) for line in lines]

            # 统计情感结果
            emotion_counts = pd.Series(emotions).value_counts()

            return emotion_counts
        else:
            st.warning("这个好像不是txt文档,请你重新上传")  # 防止用户上传非txt文档
            return None


    # 显示上传文件的控件
    uploaded_file = st.file_uploader("请你上传txt文件:", type=["txt"])

    # 确保文件已上传，然后处理文件
    if uploaded_file is not None:
        # 处理上传文件
        emotion_counts = process_uploaded_file(uploaded_file)

        # 显示饼状图
        if emotion_counts is not None:
            # 计算情绪总数
            total_count = sum(emotion_counts)

            # 计算情绪比例
            emotion_ratios = emotion_counts / total_count

            # 使用 matplotlib 绘制饼状图
            fig, ax = plt.subplots()
            ax.pie(emotion_ratios.values, labels=emotion_ratios.index, autopct='%1.1f%%')
            ax.set_title("Sentiment Analysis Result")
            ax.text(0, 0, f"Total dui hua: {total_count}", ha='center', va='center', fontsize=12, weight='bold')
            st.pyplot(fig)

    st.write("介绍:这个模型可以识别用户上传的txt文本，并将里面的用户对于产品的评价进行积极、中性、消极的评价，并用饼状图表示出来")
    # 显示图片
    image = Image.open("/smartAI/static/woman-7863041.jpg")
    st.image(image, caption='来吧,让我们来分析分析你的文档', use_column_width=True)

elif nav_choice == '情感对话机器人':

    class ChatGPT:
        def __init__(self, user):
            self.user = user
            self.messages = [{"role": "system", "content": ""}]
            self.filename = "./user_messages.json"
            openai.api_key = "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"
           # openai.api_base = "http://openai-proxy.openai-proxy.1602580520501222.us-west-1.fc.devsapp.net/v1"

        def ask_gpt(self):
            recent_messages = "\n".join([f'{msg["role"]}：{msg["content"]}' for msg in self.messages[-10:]])
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f'{recent_messages}\n机器人：',
                max_tokens=2048,
                n=1,
                stop=None,
                temperature=0.7,
            )
            return response.choices[0].text.strip()

        def writeTojson(self):
            try:
                if not os.path.exists(self.filename):
                    with open(self.filename, "w") as f:
                        pass
                with open(self.filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    msgs = json.loads(content) if len(content) > 0 else {}
                msgs.update({self.user: self.messages})
                with open(self.filename, 'w', encoding='utf-8') as f:
                    json.dump(msgs, f)
            except Exception as e:
                print(f"错误代码：{e}")


    def main():
        # 创建页面标题
        st.title('情感对话机器人Moss')

        image = Image.open("smartAI/static/axel-johansson-axjo.gif")
        st.image(image, caption='我是机器人Moss(>_<)', use_column_width=True)

        # 输入用户名称
        user = st.text_input("请输入用户名称：")

        # 初始化对话机器人
        if "chat" not in st.session_state:
            st.session_state.chat = ChatGPT(user)

        # 创建对话框
        st.subheader("与机器人Moss的对话：")
        chat_history = st.empty()
        display_conversation(st.session_state.chat, chat_history)

        # 创建输入表单
        with st.form(key="chat_input_form"):
            chat_input = st.text_input(f"{user}：", key="chat_input")
            submit_button = st.form_submit_button("发送")

        # 如果用户输入不为空，则进行处理
        if submit_button and chat_input:
            # 将用户输入写入对话历史记录
            st.session_state.chat.messages.append({"role": "user", "content": chat_input})   # 此段存在一个问题，上述输入的用户名在消息记录中不显示.

            # 获取机器人的回复
            bot_response = st.session_state.chat.ask_gpt()

            # 将机器人回复写入对话历史记录
            st.session_state.chat.messages.append({"role": "assistant", "content": bot_response})

            # 更新对话框
            display_conversation(st.session_state.chat, chat_history)

            # 写入对话历史记录
            st.session_state.chat.writeTojson()

            # 清空输入栏并刷新页面
            st.experimental_rerun()


    def display_conversation(chat, chat_history):
        conversation = ""
        for i in range(max(0, len(chat.messages) - 10), len(chat.messages)):
            role = chat.messages[i]["role"]
            content = chat.messages[i]["content"]
            if role == "user":
                conversation += f"{chat.user}: {content}\n"
            else:
                conversation += f"Moss: {content}\n"
        chat_history.markdown(f"```\n{conversation}\n```")

    if __name__ == "__main__":
        main()

elif nav_choice == '情感分析文本':
    # 创建页面标题
    st.title('情感分析文本')

    # 创建上传文件
    uploaded_file = st.file_uploader("请上传需要分析的产品评论文件", type=["txt", "md", "docx"])
    if uploaded_file is not None:
        # 读取上传的文件
        text = uploaded_file.read().decode('utf-8')

        # 调用API进行情感分析
        openai.api_base = "http://openai-proxy.openai-proxy.1602580520501222.us-west-1.fc.devsapp.net/v1"
        openai.api_key = "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"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=text,
            temperature=0.5,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # 处理API返回的结果（发现他家公司没有相应专门的，我准备调用谷歌的api，目前想法暂定，这个目前只能对小幅度文本的评论进行情感回复)
        output_text = response.choices[0].text
        #sentiment_score = round(response.choices[0].sentiment_score, 2)
        #sentiment_label = 'Positive' if sentiment_score > 0 else 'Negative'

        # 显示分析结果
        st.write("分析结果：")
        st.write("评论内容：")
        st.write(text)
        st.write("分析结果详情：")
        st.write(output_text)

        # 处理分析结果数据，生成饼状图和线形图
        #df = pd.DataFrame({'sentiment_score': [sentiment_score, 1-sentiment_score]}, index=['Positive', 'Negative'])
        fig1, ax1 = plt.subplots()
        #ax1.pie(df['sentiment_score'], labels=df.index, colors=['#00FF7F', '#FF4500'], autopct='%1.1f%%')
        ax1.set_title('Sentiment Analysis')
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot(np.arange(len(output_text.split('\n'))), [len(x) for x in output_text.split('\n')])
        ax2.set_title('Length of Each Sentence')
        st.pyplot(fig2)
    image = Image.open("/smartAI/static/taiyang.jpg")
    st.image(image, caption='233333', use_column_width=True)

elif nav_choice == '社交媒体评论分析':
    # 创建页面标题
    st.title('社交媒体评论分析')

    # 美化上传文件界面
    st.write('************************************************')
    st.write('请上传您要分析的社交媒体评论文件（目前支持txt、md、docx）')
    st.write('************************************************')

    # 创建上传文件
    uploaded_file = st.file_uploader("", type=["txt", "md", "docx"])
    if uploaded_file is not None:
        # 读取上传的文件
        text = uploaded_file.read().decode('utf-8')

        # 计算情感分数
        # 需要用到一个情感分析的算法，这里以随机数代替,目前是雏形，有待后续开发中
        pos_score = round(np.random.rand() * 100, 2)
        neg_score = round(100 - pos_score, 2)

        # 绘制表格
        st.write('社交媒体评论分析结果:')
        st.write('媒体评论的积极态度: {}'.format(pos_score))
        st.write('媒体评论的消极态度: {}'.format(neg_score))

        sentiment_label = '观感好' if pos_score > 50 else '观感差'
        st.write('媒体评论的总体态度: {}'.format(sentiment_label))
