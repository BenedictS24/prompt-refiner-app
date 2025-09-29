import os
import json
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

class LLMClient:
    """OpenAI-based LLM client for prompt refinement"""
    
    def __init__(self):
        self.openai_client = None
        self._init_openai()
    
    def _init_openai(self):
        """Initialize OpenAI client if API key is available"""
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            try:
                # Initialize OpenAI client with the API key from environment
                self.openai_client = OpenAI(
                    api_key=api_key,  # Explicitly pass the API key
                )
            except ImportError:
                print("openai package not installed, OpenAI unavailable")
            except Exception as e:
                print(f"Failed to initialize OpenAI: {e}")
    
    def has_llm_available(self) -> bool:
        """Check if OpenAI client is available"""
        return self.openai_client is not None
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on common patterns"""
        if re.search(r'[äöüß]', text.lower()):
            return "German"
        elif re.search(r'[àâäçéèêëïîôùûüÿñæœ]', text.lower()):
            return "French"
        elif re.search(r'[áéíóúüñ¿¡]', text.lower()):
            return "Spanish"
        elif re.search(r'[àèéìíîòóùú]', text.lower()):
            return "Italian"
        elif re.search(r'[ąćęłńóśźż]', text.lower()):
            return "Polish"
        elif re.search(r'[\u4e00-\u9fff]', text):
            return "Chinese"
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return "Japanese"
        elif re.search(r'[\uac00-\ud7af]', text):
            return "Korean"
        elif re.search(r'[\u0400-\u04ff]', text):
            return "Russian"
        else:
            return "English"
    
    def refine_with_llm(self, original_prompt: str, selected_tool: str = "unspecified", selected_techniques: List[str] = None, custom_techniques: str = "") -> Dict[str, str]:
        """Refine prompt using OpenAI"""
        print("[v0] Starting OpenAI refinement...")
        
        detected_language = self._detect_language(original_prompt)
        
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
                elif technique == 'instruct_reasoning':
                    technique_names.append('Explicit reasoning instructions (tell the AI to think step by step)')
                elif technique == 'instruct_search':
                    technique_names.append('Search instructions (tell the AI to search for current information when needed)')
            
            if technique_names:
                technique_context = f"\n\nTECHNIQUE REQUIREMENTS: Apply these prompt engineering techniques: {', '.join(technique_names)}."
        
        if custom_techniques:
            technique_context += f"\n\nCUSTOM TECHNIQUES: Also incorporate these custom techniques: {custom_techniques}."
        
        tool_context = ""
        if selected_tool and selected_tool != 'unspecified':
            tool_context = f"\n\nTOOL OPTIMIZATION: Optimize this prompt specifically for {selected_tool.title()}. Consider this tool's strengths and preferred prompt formats."
        
        meta_instruction = "Meta-instruction: If the prompt requires current facts or examples, use search. If it requires structured problem-solving, use reasoning. If it's simple, refine directly."
        
        if selected_techniques:
            if 'instruct_reasoning' in selected_techniques:
                meta_instruction += " IMPORTANT: Include explicit instructions for the AI to use step-by-step reasoning or thinking."
            if 'instruct_search' in selected_techniques:
                meta_instruction += " IMPORTANT: Include explicit instructions for the AI to search for current information when the task requires up-to-date facts."

        system_prompt = f"""You are an expert prompt engineer. Your task is to refine user prompts to follow best practices:

1. Clear role definition
2. Specific objective
3. Clear constraints and requirements
4. Step-by-step guidance when needed
5. Structured output format
6. Examples when helpful (especially for few-shot/one-shot techniques)
7. Brief self-check criteria{technique_context}{tool_context}

IMPORTANT: First detect the language of the original prompt. Then at the end of the refined prompt, add "Please answer in [detected language]" (e.g., "Please answer in German", "Please answer in Spanish", etc.). If the original prompt is in English, add "Please answer in English".

{meta_instruction}

Return your response as JSON with 'refined_prompt' and 'rationale' fields. Keep rationale concise (2-3 sentences max)."""
        
        user_message = f"Please refine this prompt following best practices:\n\n{original_prompt}"
        
        if self.openai_client:
            try:
                print("[v0] Making OpenAI API call...")
                response = self.openai_client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_completion_tokens=2000,
                    response_format={ "type": "json_object" }
                )
                
                response_text = response.choices[0].message.content.strip()
                print(f"[v0] OpenAI response received, content: {response_text[:200]}...")

                try:
                    result = json.loads(response_text)
                    print("[v0] Successfully parsed JSON response")
                    # Validate required fields
                    if 'refined_prompt' in result and 'rationale' in result:
                        # Ensure proper JSON structure
                        return {
                            'refined_prompt': result['refined_prompt'].strip(),
                            'rationale': result['rationale'].strip()
                        }
                    else:
                        print("[v0] Missing required fields in JSON response")
                        # Create properly formatted JSON instead of raising an error
                        return {
                            'refined_prompt': response_text.strip(),
                            'rationale': 'Refined using OpenAI with prompt engineering best practices.'
                        }

                except json.JSONDecodeError as e:
                    print(f"[v0] JSON parsing failed: {e}, using fallback format")
                    # Create properly formatted response
                    return {
                        'refined_prompt': response_text.strip(),
                        'rationale': 'Refined using OpenAI with prompt engineering best practices.'
                    }

                except Exception as e:
                    print(f"[v0] Unexpected error: {e}")
                    raise Exception("An unexpected error occurred while processing the response.")
                    
            except Exception as e:
                print(f"[v0] OpenAI error: {e}")
                raise Exception(f"OpenAI API error: {e}")
        
        raise Exception("OpenAI client not available - please set OPENAI_API_KEY environment variable")
