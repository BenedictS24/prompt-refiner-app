# Prompt Refiner

A Flask web application that refines task prompts using best prompt-engineering practices. Works in heuristic mode without API keys, but can be enhanced with Gemini 1.5 Pro.

## Features

- **Heuristic Mode**: Rule-based prompt refinement (always available)
- **AI Enhancement**: Optional integration with Gemini 1.5 Pro
- **Best Practices**: Applies clear roles, objectives, constraints, structured output formats
- **Security**: CSRF protection, rate limiting, input validation
- **Export**: Download refined prompts as .txt files
- **Responsive**: Mobile-friendly interface

## Quick Start

### Local Development

1. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

2. Set environment variables (optional for AI enhancement):
\`\`\`bash
export GEMINI_API_KEY="your-gemini-api-key"
export SECRET_KEY="your-secret-key"
\`\`\`

3. Run the application:
\`\`\`bash
python app.py
\`\`\`

Visit `http://localhost:5000`

### Deploy to Render

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Set environment variables in Render dashboard:
   - `SECRET_KEY`: Generate a secure random key
   - `GEMINI_API_KEY`: (optional) Your Gemini API key for AI enhancement

The app will automatically deploy using the included `Procfile`.

## API Endpoints

- `GET /` - Main form interface
- `POST /refine` - Refine a prompt (rate limited)
- `GET /download` - Download last refined prompt
- `GET /healthz` - Health check

## How It Works

1. **Input**: User pastes a messy or basic prompt
2. **Analysis**: System analyzes prompt structure and identifies improvements
3. **Refinement**: Applies prompt engineering best practices:
   - Clear role definition
   - Specific objectives
   - Constraints and requirements
   - Step-by-step guidance
   - Structured output format
   - Self-check criteria
4. **Output**: Returns refined prompt with rationale

## Architecture

- `app.py` - Main Flask application
- `services/refiner.py` - Core refinement logic
- `services/llm_client.py` - Gemini client abstraction
- `templates/` - Jinja2 templates with Tailwind CSS
- Security features: CSRF, rate limiting, input validation

## License

MIT License
