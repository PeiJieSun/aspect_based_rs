'''
    author: Peijie Sun
    e-mail: sun.hfut@gmail.com 
    released date: 04/18/2019
'''

import os, shutil
from datetime import datetime

class Logging():
    def __init__(self, log_path):
        self.filename = log_path
        
    def record(self, str_log):
        now = datetime.now()
        filename = self.filename
        print(str_log)
        with open(filename, 'a') as f:
            f.write("%s %s\r\n" % (now.strftime('%Y-%m-%d-%H:%M:%S'), str_log))
            f.flush()
    
    '''
    def __init__(self):
        pass
        
    def record(self, str_log):
        print(str_log)
    '''