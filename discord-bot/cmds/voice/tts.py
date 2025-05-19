import asyncio
import discord
import os
import traceback
from discord import app_commands
from discord.ext import commands
from gtts import gTTS

class TTS(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.active_connections = {}

        # 強制載入 Opus
        discord.opus.load_opus('/usr/lib/x86_64-linux-gnu/libopus.so.0')

    @app_commands.command(name="tts", description="語音播報系統")
    @app_commands.describe(text="我會播放你輸入的文字><")
    async def tts(self, interaction: discord.Interaction, text: str):
        voice_state = interaction.user.voice

        if not voice_state or not voice_state.channel:
            await interaction.response.send_message("你必須先在語音頻道裡面才能使用這個指令喔～", ephemeral=True)
            return

        await interaction.response.send_message(f"🫧：{text}", ephemeral=False)

        mp3_filename = f"tts_{interaction.user.id}.mp3"

        try:
            # 產生語音
            tts = gTTS(text=text, lang="zh-tw")
            tts.save(mp3_filename)

            if not os.path.exists(mp3_filename):
                await interaction.followup.send("語音檔生成失敗，請稍後再試。", ephemeral=True)
                return

            print(f"成功生成 TTS 檔案: {mp3_filename}, 大小: {os.path.getsize(mp3_filename)} bytes")

            user_channel = voice_state.channel
            channel_id = user_channel.id

            # 檢查 bot 是否已連線
            vc = discord.utils.get(self.bot.voice_clients, guild=interaction.guild)

            if vc:
                if vc.channel.id != channel_id:
                    await vc.disconnect()
                    print("離開原本的語音頻道")
                    vc = await user_channel.connect()
                    print("重新連線至使用者語音頻道")
                else:
                    print("Bot 已在相同語音頻道")
            else:
                vc = await user_channel.connect()
                print("Bot 初次連線至語音頻道")

            self.active_connections[channel_id] = vc

            ffmpeg_options = {
                'options': '-hide_banner -loglevel error',
                'executable': "/usr/bin/ffmpeg"
            }

            audio_source = discord.FFmpegPCMAudio(mp3_filename, **ffmpeg_options)
            audio = discord.PCMVolumeTransformer(audio_source)

            def after_playing(error):
                if error:
                    print(f"播放錯誤: {error}")
                # 播放完後等待並保持在頻道
                asyncio.run_coroutine_threadsafe(self._cleanup_file(mp3_filename, vc), self.bot.loop)

            vc.play(audio, after=after_playing)
            print(f"開始播放音檔: {mp3_filename}")

        except Exception as e:
            print(f"TTS錯誤: {str(e)}")
            print(traceback.format_exc())
            await interaction.followup.send("發生錯誤，請稍後再試。", ephemeral=True)
            try:
                if os.path.exists(mp3_filename):
                    os.remove(mp3_filename)
            except:
                pass

    async def _cleanup_file(self, filename: str, vc: discord.VoiceClient):
        await asyncio.sleep(0)  # 確保協程正確切換
        try:
            # 等待播放完成後再清理音檔
            if os.path.exists(filename):
                os.remove(filename)
                print(f"音檔 {filename} 播放完畢後刪除成功")
            
            # 當音訊播放結束後 bot 保持在頻道
            if vc.is_connected():
                print("Bot 保持在語音頻道")
                await asyncio.sleep(5)  # 延遲保留時間（可調整）
                
            else:
                print("Bot 斷開了語音頻道")
        except Exception as e:
            print(f"無法刪除音檔 {filename}: {e}")

# Bot setup
async def setup(bot: commands.Bot):
    await bot.add_cog(TTS(bot))
