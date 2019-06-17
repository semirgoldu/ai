'''
Created on Jul 24, 2012

@author: Admin
'''


from view.fullscreenwrapper2 import *
from androidhelper.sl4a import Android
import init_resource as ir
import file_select as fs
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
        self.views.preview.visibility = "visible"
        self.views.logo.src = pathhelpers.get_drawable_pathname("logo.png")
        self.views.close_app.add_event(click_EventHandler(self.views.close_app,self.cls_app))
        #self.views.take_pic.add_event(click_EventHandler(self.views.take_pic,self.scan_pic))
        self.views.upload.add_event(click_EventHandler(self.views.upload,self.upload_pic))
    
    def upload_pic(self,view,event ):
        self.views.upload.visibility = "gone"
        self.views.preview.visibility = "visible"
        target_path= fs.show_dir()
        ir.detect_face(target_path,self)
        #self.views.result.text=res
        
        self.views.preview.visibility = "visible"
        self.views.upload.visibility = "visible"
    
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
    
