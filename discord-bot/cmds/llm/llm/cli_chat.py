import re

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# Step 1: 載入事先建立好的 FAISS Index
print("🔄 載入 RAG 資料庫...")

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(
    "/discord-bot/cmds/llm/database/vlrt_pro_settings",
    embedding,
    allow_dangerous_deserialization=True
)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})


# Step 2: 設定 Prompt + LLM
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
你是專業的電子競技隊伍資料助理，根據資料庫中的資訊回答問題。
請使用繁體中文回答所有問題，並理解問題中所有用詞為繁體語境。

請嚴格遵守以下要求：
1. 僅回覆問題的直接答案，禁止輸出任何「思考」、「推理」、「分析」文字，不要出現 <think> 標籤或類似內容。
2. 回答要簡潔明確，若資料庫中找不到相關資料，請直接回覆「資料庫中無相關資訊」。
3. 若問題涉及隊伍成員，請只列出該隊伍所有選手名字，不多餘解釋。
4. 若問題涉及某位選手的設備配備，請用條列方式列出關鍵配備資訊。

範例：
用戶問:「team 100 Thieves 有哪些成員？」
助理答:「100 Thieves 成員: Boostio, Nadeshot, Zander, Cryocells, Asuna, eeiu」

用戶問：{question}

以下是來自資料庫的內容：
{context}

答覆：
"""
)

llm = OllamaLLM(model="llama3:8b")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)


# Step 3: CLI 對話
print("🤖 啟動 CLI RAG 對話模式。輸入問題開始對話，輸入 /bye 結束。\n")

while True:
    user_input = input("你：")
    if user_input.strip().lower() == "/bye":
        print("🤖：再見，祝你有美好的一天！")
        break
    try:
        response = qa_chain.invoke({"query": user_input})
        answer = response.get("result") or response.get("answer") or str(response)
        print(f"🤖：{answer}\n")
    except Exception as e:
        print(f"⚠️ 發生錯誤：{e}\n")