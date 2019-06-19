'''
Created on Jul 24, 2012

@author: Admin
'''


from view.fullscreenwrapper2 import *
from androidhelper import Android
import init_resource as ir

import datetime
import os
import view.pathhelpers as pathhelpers
import sys
import time

droid = Android()


# Main Screen Class
class MainScreen(Layout):
    def __init__(self):
        #initialize your class data attributes
        
        #load & set your xml
        super(MainScreen,self).__init__(pathhelpers.read_layout_xml("main.xml"),"Ocr")

    def on_show(self):
    	#self.views.tt.add_event(click_EventHandler(self.views.tt,self.get_options))
    	#self.views.lists.set_listitems(["semir","worku","semir","worku","semir","worku"])
        #initialize your layout views on screen_show
        self.views.logo.src = pathhelpers.get_drawable_pathname("logo.png")
        self.views.close_app.add_event(click_EventHandler(self.views.close_app,self.cls_app))
        self.views.search.add_event(click_EventHandler(self.views.search,self.search_term))
    def search_term(self,view,event ):
    	#print view
        term=self.views.search_box.text
        self.views.result.text= "\nMe: "+term +" \n" +self.views.result.text
    	res= ir.chat(term,self)
    	self.views.result.text="\nSansa: " + res +"\n" + self.views.result.text
    	self.views.search_box.text=""
        #FullScreenWrapper2App.close_layout()
    def cls_app(self,view,event):
        FullScreenWrapper2App.close_layout()
    def get_options(self,view,event):
    	title = 'GUI Test?'
    	choices=['Continue', 'Skip', 'baz']
    	droid.dialogCreateAlert(title)
    	droid.dialogSetSingleChoiceItems(choices)
    	droid.dialogSetPositiveButtonText('Yay!')
    	droid.dialogShow()
    	response = droid.dialogGetResponse().result
		
    	selected_choices = droid.dialogGetSelectedItems().result
    	print selected_choices
    	self.views.tt.text=choices[selected_choices[0]]
        return True
    def on_close(self):
        pass

if __name__ == '__main__':
    FullScreenWrapper2App.initialize(droid)
    FullScreenWrapper2App.show_layout(MainScreen())
    FullScreenWrapper2App.eventloop()
    
