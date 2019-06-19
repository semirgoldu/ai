from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer 
from chatterbot.trainers import ListTrainer

import training_data as td
# Uncomment the following lines to enable verbose logging
# import logging
# logging.basicConfig(level=logging.INFO)

# Create a new ChatBot instance
bot = ChatBot(
    'Terminal',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[ 
    
    
    
    
    {'import_path':'chatterbot.logic.MathematicalEvaluation'},
    {'import_path':'chatterbot.logic.BestMatch'},
    
    
                                   
                     
                     ],
    
    #input_adapter='chatterbot.input.TerminalAdapter',
    #output_adapter='chatterbot.outputTerminalAdapter',
    #database='my',
    database_uri='sqlite:////storage/emulated/0/chatbot233.sqlite'
    
)
bot.set_trainer(ListTrainer) 


#bot.set_trainer(ChatterBotCorpusTrainer) 
#bot.train("chatterbot.corpus.english")
bot.train(td.train_data)
def resp(text):
	bot_input = bot.get_response(text)