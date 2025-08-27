import re
from typing import Dict, List

class PromptRefiner:
    """Service for refining prompts using heuristic rules or LLM"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def refine_prompt(self, original_prompt: str) -> Dict[str, str]:
        """Main method to refine a prompt"""
        print(f"[v0] LLM client available: {self.llm_client and self.llm_client.has_llm_available()}")
        
        # Try LLM first if available
        if self.llm_client and self.llm_client.has_llm_available():
            try:
                print("[v0] Attempting LLM refinement...")
                result = self.llm_client.refine_with_llm(original_prompt)
                print(f"[v0] LLM refinement successful, result keys: {result.keys()}")
                return result
            except Exception as e:
                print(f"[v0] LLM refinement failed: {e}, falling back to heuristic")
        
        # Fallback to heuristic refinement
        print("[v0] Using heuristic refinement")
        return self._heuristic_refine(original_prompt)
    
    def _heuristic_refine(self, original_prompt: str) -> Dict[str, str]:
        """Refine prompt using heuristic rules"""
        refined_sections = []
        improvements = []
        
        # Analyze the original prompt
        analysis = self._analyze_prompt(original_prompt)
        
        # Add role if missing
        if not analysis['has_role']:
            refined_sections.append("**Role:** You are an expert assistant specialized in this task.")
            improvements.append("added clear role definition")
        
        # Add objective section
        objective = self._extract_or_create_objective(original_prompt)
        refined_sections.append(f"**Objective:** {objective}")
        if not analysis['has_clear_objective']:
            improvements.append("clarified objective")
        
        # Add the main task with improvements
        main_task = self._improve_task_description(original_prompt)
        refined_sections.append(f"**Task:** {main_task}")
        
        # Add constraints and requirements
        constraints = self._extract_or_add_constraints(original_prompt)
        if constraints:
            refined_sections.append(f"**Requirements:**\n{constraints}")
            improvements.append("added specific requirements")
        
        # Add output format if missing
        if not analysis['has_output_format']:
            output_format = self._suggest_output_format(original_prompt)
            refined_sections.append(f"**Output Format:** {output_format}")
            improvements.append("specified output format")
        
        # Add self-check
        refined_sections.append("**Self-Check:** Before responding, verify that your output meets all requirements and addresses the core objective.")
        improvements.append("added self-check criteria")
        
        refined_prompt = "\n\n".join(refined_sections)
        
        # Create rationale
        rationale = f"Applied heuristic refinement: {', '.join(improvements)}. Enhanced structure and clarity using prompt engineering best practices."
        
        return {
            'refined_prompt': refined_prompt,
            'rationale': rationale
        }
    
    def _analyze_prompt(self, prompt: str) -> Dict[str, bool]:
        """Analyze prompt to identify missing elements"""
        prompt_lower = prompt.lower()
        
        return {
            'has_role': any(word in prompt_lower for word in ['you are', 'act as', 'role:', 'as a']),
            'has_clear_objective': any(word in prompt_lower for word in ['objective:', 'goal:', 'purpose:', 'aim:']),
            'has_constraints': any(word in prompt_lower for word in ['must', 'should', 'requirements:', 'constraints:']),
            'has_output_format': any(word in prompt_lower for word in ['format:', 'structure:', 'output:', 'return']),
            'has_examples': any(word in prompt_lower for word in ['example:', 'for instance', 'such as']),
        }
    
    def _extract_or_create_objective(self, prompt: str) -> str:
        """Extract existing objective or create one"""
        # Look for explicit objectives
        objective_patterns = [
            r'objective[:\s]+([^.\n]+)',
            r'goal[:\s]+([^.\n]+)',
            r'purpose[:\s]+([^.\n]+)'
        ]
        
        for pattern in objective_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Create objective from first sentence or main verb
        sentences = prompt.split('.')
        first_sentence = sentences[0].strip()
        
        # Clean up and make it objective-focused
        if len(first_sentence) > 100:
            first_sentence = first_sentence[:100] + "..."
        
        return f"Complete the following task effectively: {first_sentence}"
    
    def _improve_task_description(self, prompt: str) -> str:
        """Improve the main task description"""
        # Remove redundant phrases and improve clarity
        improved = prompt.strip()
        
        # Add step-by-step guidance if the task seems complex
        if len(prompt) > 200 or any(word in prompt.lower() for word in ['analyze', 'create', 'develop', 'design']):
            if 'step' not in prompt.lower():
                improved += "\n\nApproach this systematically, considering each aspect carefully."
        
        return improved
    
    def _extract_or_add_constraints(self, prompt: str) -> str:
        """Extract existing constraints or add relevant ones"""
        constraints = []
        
        # Look for existing constraints
        constraint_indicators = ['must', 'should', 'need to', 'required', 'ensure']
        for indicator in constraint_indicators:
            pattern = rf'{indicator}[^.]*\.'
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            constraints.extend([match.strip() for match in matches])
        
        # Add common constraints if none found
        if not constraints:
            constraints = [
                "- Provide accurate and relevant information",
                "- Be clear and concise in your response",
                "- Address all aspects of the request"
            ]
        else:
            # Format existing constraints as bullet points
            constraints = [f"- {c}" for c in constraints]
        
        return "\n".join(constraints)
    
    def _suggest_output_format(self, prompt: str) -> str:
        """Suggest appropriate output format based on prompt content"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['list', 'steps', 'points']):
            return "Provide your response as a numbered or bulleted list."
        elif any(word in prompt_lower for word in ['analyze', 'compare', 'evaluate']):
            return "Structure your response with clear headings and detailed explanations."
        elif any(word in prompt_lower for word in ['code', 'script', 'program']):
            return "Provide code with comments and brief explanations."
        elif any(word in prompt_lower for word in ['summary', 'brief']):
            return "Provide a concise summary with key points highlighted."
        else:
            return "Provide a well-structured response that directly addresses the request."
