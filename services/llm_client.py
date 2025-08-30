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
    
    def refine_with_llm(self, original_prompt: str, selected_tools: List[str] = None, selected_techniques: List[str] = None, custom_techniques: str = "") -> Dict[str, str]:
        """Refine prompt using Gemini"""
        print("[v0] Starting Gemini refinement...")
        
        technique_context = ""
        if selected_techniques and selected_techniques != ['auto']:
            technique_names = []
            for technique in selected_techniques:
                if technique == 'chain_of_thought':
                    technique_names.append('Chain of Thought (step-by-step reasoning)')
                elif technique == 'tree_of_thought':
                    technique_names.append('Tree of Thought (multiple reasoning paths)')
                elif technique == 'one_shot':
                    technique_names.append('One-shot Learning (single example)')
                elif technique == 'few_shot':
                    technique_names.append('Few-shot Learning (multiple examples)')
                elif technique == 'self_consistency':
                    technique_names.append('Self-consistency (multiple reasoning paths)')
            
            if technique_names:
                technique_context = f"\n\nTECHNIQUE REQUIREMENTS: Apply these prompt engineering techniques: {', '.join(technique_names)}."
        
        if custom_techniques:
            technique_context += f"\n\nCUSTOM TECHNIQUES: Also incorporate these custom techniques: {custom_techniques}."
        
        tool_context = ""
        if selected_tools and selected_tools != ['unspecified']:
            tool_names = ', '.join([t.title() for t in selected_tools if t != 'unspecified'])
            tool_context = f"\n\nTOOL OPTIMIZATION: Optimize this prompt specifically for {tool_names}. Consider each tool's strengths and preferred prompt formats."
        
        system_prompt = f"""You are an expert prompt engineer. Your task is to refine user prompts to follow best practices:

1. Clear role definition
2. Specific objective
3. Clear constraints and requirements
4. Step-by-step guidance when needed
5. Structured output format
6. Examples when helpful (especially for few-shot/one-shot techniques)
7. Brief self-check criteria{technique_context}{tool_context}

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
                
                # Try to parse as JSON
                try:
                    result = json.loads(response_text)
                    print("[v0] Successfully parsed JSON response")
                    
                    # Validate required fields
                    if 'refined_prompt' in result and 'rationale' in result:
                        return result
                    else:
                        print("[v0] Missing required fields in JSON response")
                        raise json.JSONDecodeError("Missing required fields", response_text, 0)
                        
                except json.JSONDecodeError as e:
                    print(f"[v0] JSON parsing failed: {e}, using fallback format")
                    # Extract refined prompt from response
                    refined_prompt = response.text
                    rationale = "Refined using Gemini 1.5 Flash with prompt engineering best practices."
                    return {'refined_prompt': refined_prompt, 'rationale': rationale}
                    
            except Exception as e:
                print(f"[v0] Gemini error: {e}")
                raise Exception(f"Gemini API error: {e}")
        
        raise Exception("Gemini client not available - please set GEMINI_API_KEY environment variable")
