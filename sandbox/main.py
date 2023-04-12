from dotenv import dotenv_values
from flask import Flask, request

config = dotenv_values(".env")

app = Flask(__name__)

@app.route('/')
def index():
  return 'Server Works!'
  
@app.route('/block', methods=['POST'])
def get_block():
  req = request.form
  print(req)
  return req