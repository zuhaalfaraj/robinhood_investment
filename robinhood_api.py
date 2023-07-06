#from robin_stocks import *
import robin_stocks.robinhood as r
from dotenv import load_dotenv

load_dotenv()

login = r.login(os.getenv('USERNAME'),os.getenv('PASSWORD'))