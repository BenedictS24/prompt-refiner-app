import re
from typing import Dict, List

class PromptRefiner:
    """Service for refining prompts using heuristic rules or LLM"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def refine_prompt(self, original_prompt: str, selected_tool: str = "unspecified", selected_techniques: List[str] = None, custom_techniques: str = "") -> Dict[str, str]:
        """Main method to refine a prompt"""
        if not selected_techniques:
            selected_techniques = ['auto']
            
        print(f"[v0] LLM client available: {self.llm_client and self.llm_client.has_llm_available()}")
        print(f"[v0] Selected AI tool: {selected_tool}")
        print(f"[v0] Selected techniques: {selected_techniques}")
        print(f"[v0] Custom techniques: {custom_techniques}")
        
        # Try LLM first if available
        if self.llm_client and self.llm_client.has_llm_available():
            try:
                print("[v0] Attempting LLM refinement...")
                result = self.llm_client.refine_with_llm(original_prompt, selected_tool, selected_techniques, custom_techniques)
                print(f"[v0] LLM refinement successful, result keys: {result.keys()}")
                return result
            except Exception as e:
                print(f"[v0] LLM refinement failed: {e}, falling back to heuristic")
        
        # Fallback to heuristic refinement
        print("[v0] Using heuristic refinement")
        return self._heuristic_refine(original_prompt, selected_tool, selected_techniques, custom_techniques)
    
    def _heuristic_refine(self, original_prompt: str, selected_tool: str, selected_techniques: List[str], custom_techniques: str) -> Dict[str, str]:
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
        main_task = self._improve_task_description(original_prompt, selected_techniques, custom_techniques)
        refined_sections.append(f"**Task:** {main_task}")
        
        # Add constraints and requirements
        constraints = self._extract_or_add_constraints(original_prompt)
        if constraints:
            refined_sections.append(f"**Requirements:**\n{constraints}")
            improvements.append("added specific requirements")
        
        technique_guidance = self._get_technique_specific_guidance(selected_techniques, custom_techniques)
        if technique_guidance:
            refined_sections.append(f"**Prompting Technique:**\n{technique_guidance}")
            improvements.append("applied prompt engineering techniques")
        
        tool_guidance = self._get_tool_specific_guidance(selected_tool)
        if tool_guidance:
            refined_sections.append(f"**AI Tool Optimization:**\n{tool_guidance}")
            improvements.append("added tool-specific optimization")
        
        # Add output format if missing
        if not analysis['has_output_format']:
            output_format = self._suggest_output_format(original_prompt)
            refined_sections.append(f"**Output Format:** {output_format}")
            improvements.append("specified output format")
        
        # Add self-check
        refined_sections.append("**Self-Check:** Before responding, verify that your output meets all requirements and addresses the core objective.")
        improvements.append("added self-check criteria")
        
        refined_prompt = "\n\n".join(refined_sections)
        
        rationale_parts = []
        if selected_tool != 'unspecified':
            rationale_parts.append(f"optimized for {selected_tool.title()}")
        
        technique_names = ', '.join([t.replace('_', ' ').title() for t in selected_techniques if t != 'auto'])
        if technique_names:
            rationale_parts.append(f"using {technique_names} techniques")
        if custom_techniques:
            rationale_parts.append(f"incorporating custom techniques: {custom_techniques}")
        
        rationale = f"Applied heuristic refinement"
        if rationale_parts:
            rationale += f" {' and '.join(rationale_parts)}"
        rationale += f": {', '.join(improvements)}. Enhanced structure and clarity using prompt engineering best practices."
        
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
    
    def _improve_task_description(self, prompt: str, selected_techniques: List[str], custom_techniques: str) -> str:
        """Improve the main task description with technique-specific enhancements"""
        improved = prompt.strip()
        
        if 'chain_of_thought' in selected_techniques:
            if 'step' not in improved.lower() and 'think' not in improved.lower():
                improved += "\n\nThink through this step-by-step, explaining your reasoning at each stage."
        
        if 'tree_of_thought' in selected_techniques:
            improved += "\n\nConsider multiple approaches to this problem, evaluate their merits, and choose the best path forward."
        
        if 'self_consistency' in selected_techniques:
            improved += "\n\nGenerate multiple solutions and verify consistency across approaches."
        
        # Add step-by-step guidance if the task seems complex
        if len(prompt) > 200 or any(word in prompt.lower() for word in ['analyze', 'create', 'develop', 'design']):
            if 'step' not in prompt.lower() and 'chain_of_thought' not in selected_techniques:
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
    
    def _get_technique_specific_guidance(self, selected_techniques: List[str], custom_techniques: str) -> str:
        """Generate technique-specific guidance"""
        if selected_techniques == ['auto'] and not custom_techniques:
            return ""
        
        guidance = []
        
        if 'chain_of_thought' in selected_techniques:
            guidance.append("- Use Chain of Thought: Break down complex reasoning into explicit steps")
        
        if 'tree_of_thought' in selected_techniques:
            guidance.append("- Use Tree of Thought: Explore multiple reasoning paths and select the best approach")
        
        if 'one_shot' in selected_techniques:
            guidance.append("- Use One-shot Learning: Provide one clear example to guide the response format")
        
        if 'few_shot' in selected_techniques:
            guidance.append("- Use Few-shot Learning: Include 2-3 examples demonstrating the desired output")
        
        if 'self_consistency' in selected_techniques:
            guidance.append("- Use Self-consistency: Generate multiple reasoning paths and choose the most consistent answer")
        
        if custom_techniques:
            guidance.append(f"- Custom Techniques: Apply {custom_techniques}")
        
        return "\n".join(guidance) if guidance else ""
    
    def _get_tool_specific_guidance(self, selected_tool: str) -> str:
        """Generate tool-specific optimization guidance"""
        if not selected_tool or selected_tool == 'unspecified':
            return ""
        
        guidance = []
        
        if selected_tool == 'chatgpt':
            guidance.append("- For ChatGPT: Use clear, conversational language and break complex tasks into steps")
        elif selected_tool == 'gemini':
            guidance.append("- For Gemini: Leverage multimodal capabilities and structured reasoning when applicable")
        elif selected_tool == 'claude':
            guidance.append("- For Claude: Emphasize analytical thinking and detailed explanations")
        elif selected_tool == 'perplexity':
            guidance.append("- For Perplexity: Include specific search terms and request citations for factual claims")
        elif selected_tool == 'copilot':
            guidance.append("- For Copilot: Be specific about code requirements, language, and desired functionality")
        
        return "\n".join(guidance) if guidance else ""
