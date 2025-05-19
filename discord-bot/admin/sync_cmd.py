from discord.ext import commands

class Admin(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name="sync")
    @commands.has_permissions(administrator=True)
    async def sync_cmd(self, ctx):
        """同步 slash 指令"""
        await ctx.send("[ADMIN] [INFO] Synchronize slash commands ...")
        await self.bot.tree.sync()
        await ctx.send("[ADMIN] [INFO] Synchronize successfully.")


async def setup(bot: commands.Bot):
    await bot.add_cog(Admin(bot))