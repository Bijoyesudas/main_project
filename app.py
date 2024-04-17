from subprocess import call
from flask import Flask, jsonify, render_template, request,redirect, url_for
from gramformer import Gramformer
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


from flask import Flask, render_template, request
import webbrowser
import os
from flask_cors import CORS
import json

import lambdaTTS
import lambdaSpeechToScore
import lambdaGetSample


app = Flask(__name__)

# Initialize Gramformer and the paraphrasing model
gf = Gramformer(models=1, use_gpu=False)
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

def paraphrase(question, **kwargs):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=kwargs.get('max_length', 128),
        truncation=True,
    ).input_ids.to(device)
    
    outputs = model.generate(
        input_ids, **kwargs
    )
    
    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return res

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/CheckGrammer')
def checkGrammer():
    return render_template('checkGrammer.html')

@app.route('/process-text', methods=['POST'])
def process_text():
    data = request.get_json()
    text = data['text']
    corrected_text_set = gf.correct(text)
    corrected_text_list = list(corrected_text_set)
    return jsonify({'result': corrected_text_list})

@app.route('/Paraphraser')
def Paraphraser():
    return render_template('paraphraser.html')

@app.route('/para', methods=['POST'])
def process_paraphrase():
    data = request.get_json()
    input_sentence = data['text']
    # Call the paraphrase function with the input sentence
    paraphrased_sentences = paraphrase(input_sentence, num_beams=5, num_beam_groups=5, num_return_sequences=5, repetition_penalty=10.0, diversity_penalty=3.0, no_repeat_ngram_size=2, temperature=0.7, max_length=128)
    # Return the paraphrased sentences to the frontend
    return jsonify({'result': paraphrased_sentences})


@app.route('/run-script', methods=['GET'])
def run_script():
    
    # call(['python', 'webApp.py'])
    
    return render_template("main.html")


@app.route('/getAudioFromText', methods=['POST'])
def getAudioFromText():
    event = {'body': json.dumps(request.get_json(force=True))}
    return lambdaTTS.lambda_handler(event, [])


@app.route('/getSample', methods=['POST'])
def getNext():
    event = {'body':  json.dumps(request.get_json(force=True))}
    return lambdaGetSample.lambda_handler(event, [])


@app.route('/GetAccuracyFromRecordedAudio', methods=['POST'])
def GetAccuracyFromRecordedAudio():

    event = {'body': json.dumps(request.get_json(force=True))}
    print(event)
    lambda_correct_output = lambdaSpeechToScore.lambda_handler(event, [])
    return lambda_correct_output

if __name__ == '__main__':
    app.run(debug=True)
