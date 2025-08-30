from flask import Flask, render_template, request, jsonify, send_file, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
import os
import io
import secrets
from datetime import datetime
from services.refiner import PromptRefiner
from services.llm_client import LLMClient

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(16))

flask_env = os.environ.get('FLASK_ENV', 'development')
if flask_env == 'production':
    # Use memory storage for production (simple but functional)
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://"
    )
else:
    # Use default in-memory for development (suppresses warning)
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["200 per day", "50 per hour"]
    )

# Security setup
csrf = CSRFProtect(app)

# Initialize services
llm_client = LLMClient()
refiner = PromptRefiner(llm_client)

@app.route('/')
def index():
    """Main form page"""
    return render_template('index.html')

@app.route('/refine', methods=['POST'])
@limiter.limit("10 per minute")
def refine_prompt():
    """Refine the submitted prompt"""
    try:
        # Get and validate input
        original_prompt = request.form.get('prompt', '').strip()
        selected_tools = request.form.getlist('ai_tools')
        selected_techniques = request.form.getlist('prompt_techniques')
        custom_techniques = request.form.get('custom_techniques', '').strip()
        
        if not original_prompt:
            return jsonify({'error': 'Please provide a prompt to refine'}), 400
        
        if len(original_prompt) > 5000:
            return jsonify({'error': 'Prompt too long (max 5000 characters)'}), 400
        
        result = refiner.refine_prompt(original_prompt, selected_tools, selected_techniques, custom_techniques)
        
        # Store in session for download
        session['last_refined'] = {
            'original': original_prompt,
            'refined': result['refined_prompt'],
            'rationale': result['rationale'],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Error refining prompt: {str(e)}")
        return jsonify({'error': 'An error occurred while refining the prompt'}), 500

@app.route('/download')
def download_refined():
    """Download the refined prompt as a text file"""
    if 'last_refined' not in session:
        return "No refined prompt available", 404
    
    refined_data = session['last_refined']
    
    # Create text content
    content = f"""# Refined Prompt
Generated on: {refined_data['timestamp']}

## Original Prompt:
{refined_data['original']}

## Refined Prompt:
{refined_data['refined']}

## Rationale:
{refined_data['rationale']}
"""
    
    # Create file-like object
    file_obj = io.BytesIO(content.encode('utf-8'))
    file_obj.seek(0)
    
    return send_file(
        file_obj,
        as_attachment=True,
        download_name=f"refined_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mimetype='text/plain'
    )

@app.route('/healthz')
def health_check():
    """Health check endpoint"""
    return "ok"

if __name__ == '__main__':
    if flask_env == 'production':
        port = int(os.environ.get('PORT', 5000))
        debug_mode = False
    else:
        port = int(os.environ.get('PORT', 5001))  # Use 5001 locally to avoid AirPlay
        debug_mode = True
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
