import asyncio
import discord
import os

from discord.ext import commands

# Environment variables
intents = discord.Intents.default()
intents.message_content = True

# Instantiation
bot = commands.Bot(command_prefix="!", intents=intents)

# Login
@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")
    await bot.tree.sync()

async def main():
    # Load extension
    await bot.load_extension("admin.sync_cmd")
    await bot.load_extension("cmds.game.whoami")
    await bot.load_extension("cmds.game.pong")
    await bot.load_extension("cmds.voice.tts")
    await bot.load_extension("cmds.llm.vlrt_pro")
    # Launch bot
    token = os.getenv("DISCORD_TOKEN")
    if token:
        await bot.start(token)
    else:
        print("❌ DISCORD_TOKEN not set")

if __name__ == "__main__":
    asyncio.run(main())