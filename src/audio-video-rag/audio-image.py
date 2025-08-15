from src.main import os
from src.main import boto3


aws_region = "us-east-1" # choose your region you operate in

#os.environ['FINANCIAL_MODEL_PREP_KEY'] = FINANCIAL_MODELING_PREP_API_KEY =
os.environ['AUDIO_KB_ID'] = AUDIO_KB_ID = 'REPLACE-WITH-YOU-KB-ID'
os.environ['IMAGE_KB_ID'] = IMAGE_KB_ID = 'REPLACE-WITH-YOU-KB-ID'

# Temp image file
temp_gen_image = "./delme.png"
markdown_filename = "./blogpost.md"


#from blog_writer import *
from utils.bedrock import *