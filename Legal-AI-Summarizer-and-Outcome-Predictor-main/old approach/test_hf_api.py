import os
from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from huggingface_hub.errors import HfHubHTTPError

print("--- Starting API Token Test ---")

# 1. Load the .env file
if load_dotenv():
    print("Successfully loaded the .env file.")
else:
    print("Error: Could not find or load the .env file.")
    exit()

# 2. Check if the token was loaded as an environment variable
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if api_token:
    print("API token found in environment variables.")
else:
    print("Error: HUGGINGFACEHUB_API_TOKEN not found in environment variables.")
    exit()

# 3. Try to connect to the Hugging Face Hub
try:
    print("Attempting to initialize the LLM...")
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={"temperature": 0.1, "max_new_tokens": 50}
    )
    print("\n✅ SUCCESS! Your API token is working correctly.")

except HfHubHTTPError as e:
    print(f"\n❌ FAILURE! The connection failed.")
    print("--- Error Details ---")
    print(e)
    print("\nThis means your token is invalid or you haven't accepted the model's terms.")

except Exception as e:
    print(f"\n❌ FAILURE! An unexpected error occurred: {e}")