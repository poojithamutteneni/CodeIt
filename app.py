from flask import Flask, render_template, request, jsonify
from model1 import MultiAgentSystem
from model2 import generate_code as generate_code_mistral
from model3 import generate_code as generate_code_codegemma
from model4 import generate_code as generate_code_codegen
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import subprocess
import pylint.lint
from io import StringIO
import sys

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")


def evaluate_with_codebert(code: str):
    """Evaluate the quality of code using CodeBERT."""
    try:
        # cleaning the output
        cleaned_code = code.strip("```python").strip("```").strip()

        # Tokenizing the input code
        inputs = tokenizer(cleaned_code, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # get prob for quality
        softmax = torch.nn.Softmax(dim=1)
        probabilities = softmax(logits)

        return probabilities.max().item()  
    except Exception as e:
        # exception handling
        print(f"Error evaluating with CodeBERT: {e}")
        return 0.0  # Return a default score of 0 if evaluation fails


def evaluate_with_pylint(code: str):
    """Evaluate code using Pylint."""
    pylint_output = StringIO()
    sys.stdout = pylint_output

    with open('temp_code.py', 'w') as f:
        f.write(code)

    pylint_opts = ['--disable=C0114', '--disable=C0115', '--disable=C0116']
    pylint.lint.Run(['temp_code.py'] + pylint_opts, exit=False)

    sys.stdout = sys.__stdout__

    pylint_output.seek(0)
    output = pylint_output.getvalue()
    for line in output.splitlines():
        if line.startswith("Your code has been rated at"):
            score_line = line.strip()
            return score_line
    return "No pylint rating found."


def evaluate_with_flake8(code: str):
    """Evaluate code using Flake8."""
    with open("temp_code.py", "w") as f:
        f.write(code)

    result = subprocess.run(['flake8', 'temp_code.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    output = result.stdout.decode('utf-8')

    if output:
        num_issues = len(output.splitlines())
    else:
        num_issues = 0

    total_lines = len(code.splitlines())
    if total_lines == 0:
        score = 100
    else:
        score = max(0, (1 - (num_issues / total_lines)) * 100)

    return score


import time

def evaluate_runtime_and_code_quality(code: str):
    """Evaluate code execution time and quality score using CodeBERT."""
    try:
        # runtime
        start_time = time.perf_counter()
        try:
            exec(code, {}, {}) 
            execution_time = time.perf_counter() - start_time
        except SyntaxError as e:
            return {'error': f'Syntax Error in code: {e}'}

        # evaluation
        quality_score = evaluate_with_codebert(code)
        pylint_score = evaluate_with_pylint(code)
        flake8_score = evaluate_with_flake8(code)

        execution_time_in_microseconds = execution_time * 1_000  

        return {
            'quality_score': quality_score,
            'execution_time': execution_time_in_microseconds, 
            'pylint_score': pylint_score,
            'flake8_score': flake8_score
        }

    except Exception as e:
        return {'error': str(e)}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate_code_route():
    """Generate code based on the user's prompt and selected model."""
    data = request.json
    prompt = data['prompt']
    model = data['model']

    try:
        if model == 'model1':
            system = MultiAgentSystem()
            code = system.run(prompt)
        elif model == 'model2':
            code = generate_code_mistral(prompt)
        elif model == 'model3':
            code = generate_code_codegemma(prompt)
        elif model == 'model4':
            code = generate_code_codegen(prompt)
        else:
            return jsonify({'error': 'Invalid model selected'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'code': code})


@app.route('/evaluate', methods=['POST'])
def evaluate_code_route():
    """Evaluate the quality of the generated code."""
    data = request.json
    code = data.get('code', '')

    if not code:
        return jsonify({'error': 'Code must be provided for evaluation'}), 400

    try:
        # Evaluate both runtime and quality score
        result = evaluate_runtime_and_code_quality(code)

        if 'error' in result:
            app.logger.error(f"Evaluation error: {result['error']}")
            return jsonify({'error': result['error']}), 500

        # Log evaluation result for debugging
        app.logger.info(f"Evaluation result: {result}")

        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Unexpected error during evaluation: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
