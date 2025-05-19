import discord
from discord import app_commands
from discord.ext import commands

WRONG_MESSAGE = "草你根本不了解我。"
CORRECT_BIRTHDAY = "9/25"
CORRECT_AGE = "25歲"

# Restart
class Restart(discord.ui.View):
    def __init__(self):
        super().__init__()
        self.add_item(RestartButton())

class RestartButton(discord.ui.Button):
    def __init__(self):
        super().__init__(label="我想再認識一次!", style=discord.ButtonStyle.blurple)

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.edit_message(content="我們重新開始吧^_^！我的生日是幾號呢？", view=Birthday())

# Stage 2 - Age
class Age(discord.ui.View):
    def __init__(self):
        super().__init__()
        ages = ["5歲", "15歲", "20歲", "25歲", "30歲"]
        for age in ages:
            self.add_item(self.AgeButton(label=age))

    class AgeButton(discord.ui.Button):
        def __init__(self, label):
            super().__init__(label=label, style=discord.ButtonStyle.secondary)

        async def callback(self, interaction: discord.Interaction):
            if self.label == CORRECT_AGE:
                await interaction.response.edit_message(content="你好了解我 >///< !!", view=None)
            else:
                await interaction.response.edit_message(content=WRONG_MESSAGE, view=Restart())

# Stage 1 - Birthday
class Birthday(discord.ui.View):
    def __init__(self):
        super().__init__()
        birthdays = ["2/31", "4/6", "9/25", "13/1"]
        for date in birthdays:
            self.add_item(self.BirthdayButton(label=date))

    class BirthdayButton(discord.ui.Button):
        def __init__(self, label):
            super().__init__(label=label, style=discord.ButtonStyle.secondary)

        async def callback(self, interaction: discord.Interaction):
            if self.label == CORRECT_BIRTHDAY:
                await interaction.response.edit_message(
                    content="哇！你竟然記得我的生日🥹 那你知道我幾歲嗎？", view=Age()
                )
            else:
                await interaction.response.edit_message(content=WRONG_MESSAGE, view=Restart())

# 將 whoami 封裝成 Cog 類別
class WhoAmI(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="whoami", description="柚子茶妹妹的真實身份?!")
    async def whoami(self, interaction: discord.Interaction):
        await interaction.response.send_message("我的生日是幾號呢？", view=Birthday())

# Bot setup
async def setup(bot: commands.Bot):
    await bot.add_cog(WhoAmI(bot))