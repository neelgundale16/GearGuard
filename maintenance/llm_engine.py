import os
import requests
from anthropic import Anthropic

class LLMEngine:
    
    
    def __init__(self):
        self.ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.llama_model = os.environ.get('LLAMA_MODEL', 'llama3.2')
        self.anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
        self.anthropic_client = Anthropic(api_key=self.anthropic_key) if self.anthropic_key else None
    
    def _call_ollama(self, prompt, system_prompt=None):
        """Call Ollama API (local Llama)"""
        try:
            url = f"{self.ollama_base_url}/api/generate"
            
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
            
            payload = {
                "model": self.llama_model,
                "prompt": full_prompt,
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            return response.json()['response']
        
        except requests.exceptions.ConnectionError:
            return None  # Ollama not running
        except Exception as e:
            print(f"Ollama error: {e}")
            return None
    
    def analyze_failure(self, equipment, description):
        """Llama-powered failure analysis"""
        from .models import MaintenanceRequest
        
        history = MaintenanceRequest.objects.filter(
            equipment=equipment, 
            status='completed'
        ).order_by('-created_at')[:5]
        
        context = f"Equipment: {equipment.name} ({equipment.equipment_id})\n"
        context += f"Status: {equipment.status}\n\n"
        context += "Recent History:\n"
        
        for req in history:
            context += f"- {req.created_at.date()}: {req.title} ({req.priority})\n"
        
        context += f"\nCurrent Issue: {description}"
        
        system_prompt = """You are an expert industrial maintenance engineer with 20 years of experience. 
Analyze the equipment failure and provide:
1. Root Cause (one line)
2. Immediate Actions (2-3 bullet points)
3. Severity Score (1-10)
4. Estimated Downtime (hours)
5. Estimated Cost ($)

Keep it concise and actionable."""
        
        # Try Llama first (free)
        analysis = self._call_ollama(context, system_prompt)
        
        if analysis:
            return {
                'success': True,
                'analysis': analysis,
                'model': f'Llama ({self.llama_model})',
                'cost': 0.00  # FREE!
            }
        
        # Fallback to Claude if available
        if self.anthropic_client:
            try:
                message = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=800,
                    messages=[
                        {"role": "user", "content": f"{system_prompt}\n\n{context}"}
                    ]
                )
                return {
                    'success': True,
                    'analysis': message.content[0].text,
                    'model': 'Claude (fallback)',
                    'cost': 0.003
                }
            except Exception as e:
                return {'success': False, 'error': f'Claude error: {str(e)}'}
        
        return {
            'success': False, 
            'error': 'Ollama not running. Start it with: ollama serve'
        }
    
    def generate_maintenance_plan(self, equipment):
        """Llama-powered maintenance planning"""
        
        prompt = f"""Equipment: {equipment.name}
Type: {equipment.work_center.name if equipment.work_center else 'Unknown'}
Status: {equipment.status}

Create a 6-month preventive maintenance plan with:
- Weekly tasks
- Monthly tasks
- Quarterly tasks
- Parts replacement schedule
- Estimated budget

Keep it structured and professional."""
        
        system_prompt = "You are a maintenance planning expert. Create detailed, actionable maintenance schedules."
        
        # Try Llama first (free)
        plan = self._call_ollama(prompt, system_prompt)
        
        if plan:
            return {
                'success': True,
                'plan': plan,
                'model': f'Llama ({self.llama_model})',
                'cost': 0.00
            }
        
        # Fallback to Claude
        if self.anthropic_client:
            try:
                message = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": f"{system_prompt}\n\n{prompt}"}]
                )
                return {
                    'success': True,
                    'plan': message.content[0].text,
                    'model': 'Claude (fallback)'
                }
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': 'Ollama not running'}