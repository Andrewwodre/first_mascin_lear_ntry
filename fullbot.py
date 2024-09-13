from partg import startgpt, gen
from gtts import gTTS
import os.path
import discord
import logging
from discord.utils import get

tokenizer, model = startgpt()
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)
TOKEN = "MTA4NzQxMjE5MzAxMzgwNTEwOA.GJ9Joi.fkbbdwDsdK0VSSshF9vvalZEtVRQoeBih9VTEY"
class YLBotClient(discord.Client):
    async def on_ready(self):
        logger.info(f'{self.user} has connected to Discord!')
        for guild in self.guilds:
            logger.info(
                f'{self.user} подключились к чату:\n'
                f'{guild.name}(id: {guild.id})')

    async def on_member_join(self, member):
        await member.create_dm()
        await member.dm_channel.send(
            f'Привет, {member.name}!'
        )

    async def play(self, ctx, h=0, b="привет пупсики"):
        a = os.path.isfile("need.mp3")
        try:
            if a:
                os.remove("need.mp3")
        except PermissionError:
            print("non")
        if h == 1:
            tts = gTTS('привет пупсики', lang='ru')

            tts.save('need.mp3')
        else:
            tts = gTTS(b, lang='ru')

            tts.save('need.mp3')
        voice = get(self.voice_clients, guild=ctx.guild)
        channel = ctx.author.voice.channel
        if voice and voice.is_connected:
            await voice.move_to(channel)
        else:
            voice = await channel.connect()
        voice.play(discord.FFmpegPCMAudio("need.mp3"), after=lambda e: print('done', e))
        voice.is_playing()
        """
        n = discord.VoiceClient(client, channel)
        n.play("need.mp3", after=lambda e: print(e))
        n.is_playing()"""

    async def join(self, ctx, h=0):
        global voice
        channel = ctx.author.voice.channel
        voice = get(client.voice_clients)

        if voice and voice.is_connected():
            await voice.move_to(channel)
        else:
            voice = await channel.connect()
        if h:
            await ctx.channel.send("Приветик, пупсик")
        b = gen(ctx.content, tokenizer, model)
        await ctx.channel.send(b)
        await self.play(ctx, h=h, b=b)

    async def on_message(self, message):
        if message.content.lower()[0] == ".":
            if message.author == self.user:
                return
            if "привет" in message.content.lower():
                await self.join(message, 1)
            elif "ты кто" in message.content.lower():
                await message.channel.send("ээээм... это я")
            elif ".kish" == message.content.lower() or ".киш" == message.content.lower():
                await message.channel.send("+play король и шут 50")
            else:
                await self.join(message)
        else:
            return


intents = discord.Intents.default()
intents.members = True
intents.message_content = True
client = YLBotClient(intents=intents)
client.run(TOKEN)

tts = gTTS('hello my name kevin', lang='ru')

tts.save('hello.mp3')