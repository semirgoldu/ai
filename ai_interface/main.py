from view.fullscreenwrapper2 import *
from  androidhelper.sl4a import Android
import datetime
import os
import view.pathhelpers as pathhelpers
import sys
import time
from query_classifier import query_classifier as qc
droid = Android()

class MainScreen(Layout):
    
    def __init__(self):
    	
        #initialize your class data attributes
        
        #load & set your xml
        super(MainScreen,self).__init__(pathhelpers.read_layout_xml("main.xml"),"SL4A")

    def on_show(self):
        #initialize your layout views on screen_show
        self.views.logo.src = pathhelpers.get_drawable_pathname("logo.png")
        
        #self.views.videoView.src="file:///sdcard/video.mp4"
        #self.views.videoView.request_focus()
        #rr= self.views.videoView
        #print FullScreenWrapper2App.get_android_instance()
        self.views.classify.add_event(click_EventHandler(self.views.classify,self.classify_query))
        self.views.close_app.add_event(click_EventHandler(self.views.close_app, self.close_out))
    def close_out(self,view,event ):
        #self.views.history.set_listitems(self.items)
        FullScreenWrapper2App.close_layout()
    def classify_query(self,view,event ):
    	#print view
        term=self.views.search_box.text
        result= qc.predict(term)
        self.views.result.text=result
      
    def on_close(self):
        pass

if __name__ == '__main__':
    FullScreenWrapper2App.initialize(droid)
    FullScreenWrapper2App.show_layout(MainScreen())
    FullScreenWrapper2App.eventloop()
    
