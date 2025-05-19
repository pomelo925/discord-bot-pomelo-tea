import discord
from discord import app_commands
from discord.ext import commands 

sea_message = "大江大海江大海✓✓✓\n\
側楞著身子麼轉著回還☻☻☻\n\
♪♪♫一陣強力の間奏♫♫♪\n\
不會打歌麼學打歌!\n\
阿哥咋擺你咋擺▂▃▄"

# Slash command Prompt
@app_commands.command(name="pong")
async def pong(interaction: discord.Interaction):
    await interaction.response.send_message(sea_message)

# Bot setup
async def setup(bot: commands.Bot):
    bot.tree.add_command(pong)