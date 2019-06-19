import os
import sys
import json
from androidhelper import Android
import os
import sys
import glob
import replace_vars as rv
folder="/sdcard/extras/org.qpython.qpy/files/lib/python2.7/site-packages/*.egg"
for file in glob.glob(folder):
    sys.path.append(file)
import nltk 
import ssl
from nltk.corpus import wordnet
try: 
    _create_unverified_https_context = ssl._create_unverified_context 
except AttributeError: 
    pass
else: 
    ssl._create_default_https_context = _create_unverified_https_context 
nltk.data.path.append("/sdcard/nltkdata")
#Initialize Android
import chatterbot_api as chatbot
droid = Android()
os.environ['PATH'] = os.environ['PATH']+':/data/data/com.termux/files/usr/bin/'
os.environ['SHELL'] = '/data/data/com.termux/files/usr/bin/sh'
os.environ['MKSH'] = '/data/data/com.termux/files/usr/bin/sh'
os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']+":"+'/data/data/com.termux/files/usr/lib'
import subprocess

def chat(text,layout):
    #layout.views.result.text="Processing ..."

    try:
        res = chatbot.bot.get_response(text).text
        if "{{" in res:
            res=rv.replace_vars(res)
        return res
    # Press ctrl-c or ctrl-d on the keyboard to exit
    except Exception as e:
        return e