from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from happytransformer import HappyGeneration, GENSettings

happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-1.3B")

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate')
def generate():

    # Get request arguments
    prompt = request.args.get('prompt')

    settings = request.args
    
    # Remove prompt from settings
    settings.pop('prompt')

    # Generate text
    text = happy_gen.generate_text(
        prompt, 
        args=GENSettings(**settings),
    ).text

    # Return response
    return jsonify({
        'prompt': prompt,
        'text': text
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)