import os
import chromadb
from sentence_transformers import SentenceTransformer

class RAGSystem:
    """RAG with ChromaDB"""
    
    def __init__(self):
        persist_dir = os.path.join('data', 'chromadb')
        os.makedirs(persist_dir, exist_ok=True)
        
        self.client = chromadb.Client(chromadb.Settings(persist_directory=persist_dir))
        
        try:
            self.collection = self.client.get_collection("maintenance")
        except:
            self.collection = self.client.create_collection("maintenance", metadata={"hnsw:space": "cosine"})
        
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def index_records(self):
        """Index all maintenance records"""
        from .models import MaintenanceRequest
        
        records = MaintenanceRequest.objects.all()
        docs, metas, ids = [], [], []
        
        for r in records:
            doc = f"Title: {r.title}\nEquipment: {r.equipment.name}\nType: {r.request_type}\nPriority: {r.priority}\nDescription: {r.description}\nResolution: {r.notes}"
            docs.append(doc)
            metas.append({'id': r.id, 'equipment_id': r.equipment.equipment_id, 'priority': r.priority})
            ids.append(str(r.id))
        
        if docs:
            embeddings = self.embedder.encode(docs).tolist()
            self.collection.add(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)
        
        return len(docs)
    
    def search(self, query, n=5):
        """Semantic search"""
        query_emb = self.embedder.encode([query])[0].tolist()
        return self.collection.query(query_embeddings=[query_emb], n_results=n)
    
    def answer_question(self, question):
        """RAG: Retrieve + Generate"""
        results = self.search(question, n=3)
        
        if not results['documents'][0]:
            return {'success': False, 'error': 'No relevant history'}
        
        context = "Relevant cases:\n\n" + "\n\n".join(results['documents'][0])
        
        from .llm_engine import LLMEngine
        llm = LLMEngine()
        if not llm.openai_client:
            return {'success': False, 'error': 'OpenAI API key needed'}
        
        try:
            response = llm.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Maintenance expert. Use provided cases to answer accurately."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return {
                'success': True,
                'answer': response.choices[0].message.content,
                'sources': results['metadatas'][0],
                'method': 'RAG'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}