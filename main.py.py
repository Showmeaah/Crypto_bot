from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
import requests
import requests_cache
from langchain_ollama import OllamaLLM
# Load environment variables
load_dotenv()
# Set up an in-memory cache that lasts for the session
requests_cache.install_cache(backend='memory', expire_after=60)
ollama_model = OllamaLLM(model="deepseek-r1:latest")


class Llm_Setup:
    def __init__(self):
        self.llm = None
        self.chat_history = []  # Store conversation history
        self.setup_llm()

    def setup_llm(self):
        """Initialize the Azure OpenAI LLM"""
        try:
            self.llm = AzureChatOpenAI(
                deployment_name=os.getenv("DEPLOYMENT_NAME"),
                api_key=os.getenv("API_KEY"),
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_version=os.getenv("API_VERSION")
            )
            if not self.llm:
                raise ValueError("Failed to initialize AzureChatOpenAI.")
        except Exception as e:
            print(f"Error during LLM setup: {e}")

    def get_llm(self):
        return self.llm

    def crypto_name_extractor(self, query):
        """Extracts cryptocurrency name from user query using LLM"""
        prompt = f"""Analyze the following user query: "{query}"
        Task:
        - Determine if the query references any cryptocurrency (e.g., by ticker symbol or common abbreviation).
        - If a cryptocurrency is mentioned, return its full name in lowercase letters (e.g., 'bitcoin' for 'btc').
        - If no cryptocurrency is mentioned, return the phrase "no name".

        Respond strictly with the specified output format and nothing else.

        Examples:
        1. Query: "What's the price of eth?"
        Response: ethereum
        2. Query: "Tell me the latest news."
        Response: no name"""
        return prompt

    def get_simple_price(self, crypto, currency="inr"):
        """Fetches real-time crypto price using CoinGecko API"""
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": crypto,
            "vs_currencies": currency,
            "include_last_updated_at": True
        }
        headers = {
            "accept": "application/json",
            "X-Cg-demo-api-key": os.getenv("API_KEY_GECKO")
        }

        response = requests.get(url, params=params, headers=headers)

        # Check if the response is from cache
        if response.from_cache:
            print("Using cached data for:", crypto)

        if response.status_code == 200:
            return response.json()
        else:
            print("Error:", response.status_code)
            return None

    def generate_response(self, user_query):
        """Generates chatbot response while maintaining context"""
        # Append user query to chat history
        self.chat_history.append({"role": "user", "content": user_query})

        # Extract cryptocurrency name
        crypto_name_prompt = self.crypto_name_extractor(user_query)
        extracted_crypto_name = self.llm.invoke(crypto_name_prompt)

        if "no name" in extracted_crypto_name.content:
            # If no cryptocurrency is mentioned, use LLM for a general response
            chat_prompt = [
                {"role": "system", "content": "You are an AI crypto chatbot, answering questions intelligently."}
            ] + self.chat_history  # Pass history

            generic_search = self.llm.invoke(chat_prompt)
            response_text = generic_search.content
        else:
            # Fetch cryptocurrency price
            extracted_crypto_price = self.get_simple_price(
                extracted_crypto_name.content)

            if extracted_crypto_price:
                # Create contextual response
                crypto_query_prompt = f"""You are provided with the following price data for cryptocurrencies:
                {extracted_crypto_price}
                Now, please answer the following query using the above data:
                "{user_query}"
                Your answer should be clear, concise, and directly reference the price of crypto from the given data.
                """

                crypto_query = self.llm.invoke(crypto_query_prompt)
                response_text = crypto_query.content
            else:
                response_text = "Sorry, I couldn't fetch the latest prices. Try again later."

        # Append bot response to chat history
        self.chat_history.append(
            {"role": "assistant", "content": response_text})

        return response_text


def chat():
    """Interactive chatbot loop with context tracking"""
    llm_setup = Llm_Setup()

    print("\n💬 Crypto Chatbot is ready! Type 'exit' to quit.\n")

    while True:
        user_query = input("You: ").strip()

        if user_query.lower() in ["exit", "quit"]:
            print("👋 Exiting chat. Have a great day!")
            break

        response = llm_setup.generate_response(user_query)
        print(f"🤖 CryptoBot: {response}")


if __name__ == "__main__":
    chat()
