import discord
from discord import app_commands
from discord.ext import commands
import traceback

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class VLRTPro(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        print("🔄 載入 RAG 資料庫...")

        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(
            "/discord-bot/cmds/llm/database/vlrt_pro_settings",
            embedding,
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
你是專業的電子競技隊伍資料助理，根據資料庫中的資訊回答問題。
請使用繁體中文回答所有問題，並理解問題中所有用詞為繁體語境。

請嚴格遵守以下要求：
1. 回答要簡潔明確，若資料庫中找不到相關資料，請直接回覆「資料庫中無相關資訊」。
2. 若問題涉及隊伍成員，請只列出該隊伍所有選手名字，不多餘解釋。
3. 若問題涉及某位選手的設備配備，請用條列方式列出關鍵配備資訊。

用戶問：{question}

以下是來自資料庫的內容：
{context}

答覆：
"""
        )

        llm = OllamaLLM(model="llama3:8b")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )

    @app_commands.command(name="vlrt_pro", description="使用 Ollama3 查詢 Valorant Pro Settings")
    @app_commands.describe(prompt="請輸入你想詢問的問題")
    async def vlrt_pro(self, interaction: discord.Interaction, prompt: str):
        await interaction.response.defer(thinking=True)
        try:
            result = self.qa_chain.invoke({"query": prompt})
            output = result.get("result") or result.get("answer") or str(result)
        except Exception as e:
            print(traceback.format_exc())
            output = f"⚠️ 發生錯誤：{e}"

        # 格式化訊息，包含使用者輸入和 AI 輸出
        formatted_response = (
            f"**你**：\n```\n{prompt}\n```"
            f"**🫧**：\n```\n{output}\n```"
        )

        await interaction.followup.send(formatted_response)

# Bot setup
async def setup(bot: commands.Bot):
    await bot.add_cog(VLRTPro(bot))