import os
import requests
import chromadb
from sentence_transformers import SentenceTransformer

class RAGSystem:
    """RAG with ChromaDB + Llama (FREE!)"""
    
    def __init__(self):
        persist_dir = os.path.join('data', 'chromadb')
        os.makedirs(persist_dir, exist_ok=True)
        
        self.client = chromadb.Client(chromadb.Settings(persist_directory=persist_dir))
        
        try:
            self.collection = self.client.get_collection("maintenance")
        except:
            self.collection = self.client.create_collection(
                "maintenance", 
                metadata={"hnsw:space": "cosine"}
            )
        
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Llama config
        self.ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.llama_model = os.environ.get('LLAMA_MODEL', 'llama3.2')
    
    def index_records(self):
        """Index all maintenance records"""
        from .models import MaintenanceRequest
        
        records = MaintenanceRequest.objects.all()
        docs, metas, ids = [], [], []
        
        for r in records:
            doc = f"Title: {r.title}\n"
            doc += f"Equipment: {r.equipment.name}\n"
            doc += f"Type: {r.request_type}\n"
            doc += f"Priority: {r.priority}\n"
            doc += f"Description: {r.description}\n"
            doc += f"Resolution: {r.notes}"
            
            docs.append(doc)
            metas.append({
                'id': r.id,
                'equipment_id': r.equipment.equipment_id,
                'priority': r.priority
            })
            ids.append(str(r.id))
        
        if docs:
            embeddings = self.embedder.encode(docs).tolist()
            self.collection.add(
                documents=docs,
                embeddings=embeddings,
                metadatas=metas,
                ids=ids
            )
        
        return len(docs)
    
    def search(self, query, n=5):
        """Semantic search"""
        query_emb = self.embedder.encode([query])[0].tolist()
        return self.collection.query(query_embeddings=[query_emb], n_results=n)
    
    def _call_ollama(self, prompt):
        """Call Ollama"""
        try:
            url = f"{self.ollama_base_url}/api/generate"
            payload = {
                "model": self.llama_model,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()['response']
        except:
            return None
    
    def answer_question(self, question):
        """RAG: Retrieve + Generate with Llama"""
        
        # Retrieve relevant cases
        results = self.search(question, n=3)
        
        if not results['documents'][0]:
            return {'success': False, 'error': 'No relevant history found'}
        
        context = "Relevant past maintenance cases:\n\n"
        context += "\n\n---\n\n".join(results['documents'][0])
        
        prompt = f"""You are a maintenance expert. Use the provided cases to answer the question accurately.

{context}

Question: {question}

Answer (be specific and reference the cases):"""
        
        # Generate answer with Llama
        answer = self._call_ollama(prompt)
        
        if answer:
            return {
                'success': True,
                'answer': answer,
                'sources': results['metadatas'][0],
                'method': 'RAG + Llama (FREE!)',
                'cost': 0.00
            }
        
        return {
            'success': False,
            'error': 'Ollama not running. Start with: ollama serve'
        }