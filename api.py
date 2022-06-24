from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pydantic
from typing import Optional, List
from happytransformer import HappyGeneration, GENSettings

happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-1.3B")

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

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

    # Generate text
    text = happy_gen.generate_text(
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