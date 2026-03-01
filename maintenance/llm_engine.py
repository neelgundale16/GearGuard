import os
from openai import OpenAI
from anthropic import Anthropic

class LLMEngine:
    """Real LLM Integration"""
    
    def __init__(self):
        self.openai_key = os.environ.get('OPENAI_API_KEY')
        self.anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
        
        self.openai_client = OpenAI(api_key=self.openai_key) if self.openai_key else None
        self.anthropic_client = Anthropic(api_key=self.anthropic_key) if self.anthropic_key else None
    
    def analyze_failure(self, equipment, description):
        """GPT-4 failure analysis"""
        if not self.openai_client:
            return {'success': False, 'error': 'Set OPENAI_API_KEY in .env'}
        
        from .models import MaintenanceRequest
        history = MaintenanceRequest.objects.filter(equipment=equipment, status='completed').order_by('-created_at')[:5]
        
        context = f"Equipment: {equipment.name}\nStatus: {equipment.status}\n\nRecent History:\n"
        for req in history:
            context += f"- {req.created_at.date()}: {req.title} ({req.priority})\n"
        context += f"\nCurrent Issue: {description}"
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Expert maintenance engineer. Provide: 1) Root cause 2) Actions 3) Severity (1-10) 4) Downtime estimate 5) Cost estimate."},
                    {"role": "user", "content": context}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            return {
                'success': True,
                'analysis': response.choices[0].message.content,
                'tokens': response.usage.total_tokens,
                'cost': response.usage.total_tokens * 0.00003
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def generate_maintenance_plan(self, equipment):
        """Claude maintenance planning"""
        if not self.anthropic_client:
            return {'success': False, 'error': 'Set ANTHROPIC_API_KEY in .env'}
        
        prompt = f"Equipment: {equipment.name}\nCreate 6-month preventive maintenance plan with tasks and schedule."
        
        try:
            message = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                'success': True,
                'plan': message.content[0].text,
                'model': 'claude-3-sonnet'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}