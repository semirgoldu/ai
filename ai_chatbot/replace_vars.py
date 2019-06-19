#-*-coding:utf8;-*-
#qpy:2
#qpy:console
#import nltk
import time
import json
import os
from androidhelper import Android
droid=Android()
def get_data(item,property,query):
	data_dir=os.getcwd()
	js=data_dir+"/scripts/ai_chatbot/data/"+query+".json"
	js_data=[]
	
		
	with open(js,"r") as f:
		content=f.read()
		j_data=json.loads(content)
		t=j_data[item]
		q=False
		for p in property:
			q=p
			
		for x in t:
			js_data.append(x[property[0]][property[1]].encode("UTF-8").strip().replace("["," ").replace("]"," ").replace('"',' '))
		#print str("\n". join(js_data))
			
	return str("\n". join(js_data))
def get_array_data(item,property,query):
	data_dir=os.getcwd()
	js=data_dir+"/scripts/ai_chatbot/data/"+query+".json"
	js_data=[]
	
		
	with open(js,"r") as f:
		content=f.read()
		j_data=json.loads(content)
		t=j_data[item]
		q=False
		for p in property:
			q=p
			
		for x in t:
			#print x.encode()
			js_data.append(x.encode("utf-8").strip().replace("["," ").replace("]"," ").replace('"',' ').replace("'"," "))
		#print str("\n". join(js_data))
			
	return str("\n". join(js_data))
VARS = {
    "{{day}}":time.strftime("%A"),
    "{{date}}":time.strftime("%_e"),
    "{{month}}":time.strftime("%B"),
    "{{year}}":time.strftime("%Y"),
    "{{time}}":time.strftime("%A")+" "+time.strftime("%B")+" "+time.strftime("%_e")+" "+time.strftime("%Y"),
    "{{sim_no}}": droid.getNeighboringCellInfo(),
    "{{us_presidents}}":get_data("objects",["person","name"],"humans/us_presidents"),
    "{{celebrities}}":get_array_data("celebrities",[],"humans/celebrities")

}


	
	
def replace_vars(query):
    try:
    	#get word synsets
    
        #res = VARS[query]
        res=query
        for x in VARS:
        	if "{{" in res:
        	    res=res.replace(x,str(VARS[x]))
        	
        
        #loop over sunsets and get definition
        
        return res
    except Exception as e:
        return e
    

