import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pydantic
from typing import Optional, List
from happytransformer import HappyGeneration, GENSettings

models = {
    'gpt2': ("GPT2", "gpt2"),
    'gpt2-medium': ("GPT2", "gpt2-medium"), 
    'gpt2-large': ("GPT2", "gpt2-large"), 
    'gpt2-xl': ("GPT2", "gpt2-xl"),
    'gpt2-distil': ("GPT2", "distilgpt2"),
    'gpt-neo-125m': ("GPT-NEO", "EleutherAI/gpt-neo-125M"),
    'gpt-neo-1.3B': ("GPT-NEO", "EleutherAI/gpt-neo-1.3B"),
}

selected_models = os.getenv('MODELS', 'gpt2').split(',')

for model in selected_models:
    if models.get(model):
        print(f'Loading {model}...')
        models[model] = HappyGeneration(
            model_type=models[model][0], 
            model_name=models[model][1],
            use_auth_token=os.getenv('AUTH_TOKEN'),
        )

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/supported-models')
def supported_models():
    return jsonify(selected_models)

@app.route('/api/generate', methods=['POST'])
def generate():

    class Settings(pydantic.BaseModel):
        min_length: Optional[int]=10
        max_length: Optional[int]=50
        do_sample: Optional[bool]=False
        early_stopping: Optional[bool]=False
        num_beams: Optional[int]=1
        temperature: Optional[float]=1.0
        top_k: Optional[int]=50
        no_repeat_ngram_size: Optional[int]=0
        top_p: Optional[float]=1.0
        bad_words: Optional[List[str]]=None
        
    # Get post request data
    data = request.get_json()

    print(f'Processing request: {data}')

    # Get request arguments
    prompt = data.get('prompt')

    # Convert settings to Settings model
    settings = Settings(**data.get('settings'))

    # Get selected model
    model = models[data.get('model')]

    # Generate text
    text = model.generate_text(
        prompt, 
        args=GENSettings(**settings.dict()),
    ).text

    # Return response
    return jsonify({
        'prompt': prompt,
        'text': text
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)