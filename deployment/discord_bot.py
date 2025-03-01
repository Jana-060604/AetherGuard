'''import discord
import requests
import os
import json
from discord.ext import commands
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
API_URL = os.getenv("API_URL")

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"{bot.user} is now running!")

@bot.command(name="predict")
async def predict(ctx, *features):
    try:
        # Convert to floats after stripping whitespaces
        features = [float(f.strip()) for f in features]
        
        if len(features) != 47:
            await ctx.send("❌ Please provide exactly 47 features.")
            return
        
        # Send API request with the correct JSON structure
        payload = json.dumps({"features": features})
        headers = {"Content-Type": "application/json"}

        
        response = requests.post(API_URL, data=payload, headers=headers)
        print(payload)
        result = response.json()

        print(f"API Response: {result}")  # Debug in console

        if "prediction" in result:
            prediction = "Malicious 🔥" if result["prediction"] == 1 else "Not Malicious ✅"
            await ctx.send(f"The transaction is **{prediction}**")
        else:
            await ctx.send("❌ Error in API response format.")
            
    except ValueError:
        await ctx.send("❌ Invalid input! Please provide only numerical values.")
        
    except Exception as e:
        await ctx.send(f"⚠️ Error: {str(e)}")

bot.run(TOKEN)
'''

import discord
import requests
import os
import json
import openai
from discord.ext import commands
from discord.ui import Button, View
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
API_URL = os.getenv("API_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"{bot.user} is now running!")

@bot.command(name="predict")
async def predict(ctx, *features):
    try:
        # Convert to floats after stripping whitespaces
        features = [float(f.strip()) for f in features]
        
        if len(features) != 47:
            await ctx.send("❌ Please provide exactly 47 features.")
            return
        
        # Send API request with the correct JSON structure
        payload = json.dumps({"features": features})
        headers = {"Content-Type": "application/json"}

        response = requests.post(API_URL, data=payload, headers=headers)
        print(payload)
        result = response.json()

        print(f"API Response: {result}")  # Debug in console

        if "prediction" in result:
            prediction = "Malicious 🔥" if result["prediction"] == 1 else "Not Malicious ✅"
            await ctx.send(f"The transaction is **{prediction}**")
            
            # After prediction, ask for the user to feel free to ask anything
            await ctx.send("Feel free to ask anything related to Web3, AI, or Machine Learning!")

            def check(msg):
                return msg.author == ctx.author  # Only respond to the original user

            try:
                msg = await bot.wait_for("message", timeout=30.0, check=check)

                # Prepare the prompt for any Web3, AI, or ML related question
                explanation_prompt = f"Provide a detailed and clear explanation based on the user's question. The user asked: {msg.content}. Ensure the response is related to Web3, AI, or Machine Learning concepts. Answer concisely and accurately with relevant information."

                # Send the user's query to OpenAI for a relevant answer
                response = openai.ChatCompletion.create(
                    model="gpt-4",  # Or you can use "gpt-3.5-turbo" for lower cost
                    messages=[
                        {"role": "system", "content": "You are a knowledgeable assistant in the fields of Web3, AI, and Machine Learning."},
                        {"role": "user", "content": explanation_prompt}
                    ],
                    max_tokens=200
                )

                answer = response['choices'][0]['message']['content'].strip()
                await ctx.send(answer)

            except TimeoutError:
                await ctx.send("❌ You took too long to respond.")
        else:
            await ctx.send("❌ Error in API response format.")
            
    except ValueError:
        await ctx.send("❌ Invalid input! Please provide only numerical values.")
        
    except Exception as e:
        await ctx.send(f"⚠️ Error: {str(e)}")

bot.run(TOKEN)
