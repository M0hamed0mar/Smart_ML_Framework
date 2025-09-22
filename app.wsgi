# app.wsgi
import sys
import os


path = '"D:\NTI_Project\Project"'
if path not in sys.path:
    sys.path.append(path)

from app import app as application