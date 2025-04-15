#config file

import os
from dotenv import load_dotenv

load_dotenv()
api_key=os.getenv("ALPACA_API_KEY")
secret_key=os.getenv("SECRET_KEY")
BASE_URL=os.getenv("BASE_URL")