import os
import json
from typing import Optional, Dict, Any, List

class LLMClient:
    """Gemini-only LLM client for prompt refinement"""
    
    def __init__(self):
        self.gemini_client = None
        self._init_gemini()
    
    def _init_gemini(self):
        """Initialize Gemini client if API key is available"""
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')
            except ImportError:
                print("google-generativeai not installed, Gemini unavailable")
            except Exception as e:
                print(f"Failed to initialize Gemini: {e}")
    
    def has_llm_available(self) -> bool:
        """Check if Gemini client is available"""
        return self.gemini_client is not None
    
    def refine_with_llm(self, original_prompt: str, selected_tools: List[str] = None) -> Dict[str, str]:
        """Refine prompt using Gemini"""
        print("[v0] Starting Gemini refinement...")
        
        tool_context = ""
        if selected_tools and selected_tools != ['unspecified']:
            tool_names = ', '.join([t.title() for t in selected_tools if t != 'unspecified'])
            tool_context = f"\n\nIMPORTANT: Optimize this prompt specifically for {tool_names}. Consider each tool's strengths and preferred prompt formats."
        
        system_prompt = f"""You are an expert prompt engineer. Your task is to refine user prompts to follow best practices:

1. Clear role definition
2. Specific objective
3. Clear constraints and requirements
4. Step-by-step guidance when needed
5. Structured output format
6. Examples when helpful
7. Brief self-check criteria{tool_context}

Meta-instruction: If the prompt requires current facts or examples, use search. If it requires structured problem-solving, use reasoning. If it's simple, refine directly.

Return your response as JSON with 'refined_prompt' and 'rationale' fields. Keep rationale concise (2-3 sentences max)."""
        
        user_message = f"Please refine this prompt following best practices:\n\n{original_prompt}"
        
        if self.gemini_client:
            try:
                print("[v0] Making Gemini API call...")
                response = self.gemini_client.generate_content(
                    f"{system_prompt}\n\n{user_message}",
                    generation_config={
                        "temperature": 0.3,
                        "max_output_tokens": 2000,
                    }
                )
                
                print(f"[v0] Gemini response received, length: {len(response.text)}")
                print(f"[v0] Raw response: {response.text[:200]}...")
                
                response_text = response.text.strip()
                
                # Remove markdown code blocks if present
                if response_text.startswith('```json'):
                    response_text = response_text[7:]  # Remove ```json
                if response_text.endswith('```'):
                    response_text = response_text[:-3]  # Remove ```

                response_text = response_text.strip()

                # Ensure proper JSON formatting
                try:
                    result = json.loads(response_text)
                    formatted_response = json.dumps(result, indent=4)  # Format JSON with indentation
                    print("[v0] Successfully formatted JSON response")
                    return json.loads(formatted_response)  # Return as a dictionary
                except json.JSONDecodeError as e:
                    print(f"[v0] JSON formatting failed: {e}, using fallback format")
                    refined_prompt = response_text
                    rationale = "Refined using Gemini 1.5 Flash with prompt engineering best practices."
                    return {'refined_prompt': refined_prompt, 'rationale': rationale}
                    
            except Exception as e:
                print(f"[v0] Gemini error: {e}")
                raise Exception(f"Gemini API error: {e}")
        
        raise Exception("Gemini client not available - please set GEMINI_API_KEY environment variable")
