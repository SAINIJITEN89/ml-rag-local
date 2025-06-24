import os
import json
import requests
from typing import List, Dict, Any, Optional
from simple_vectordb import SimpleVectorDB
from model_config import ModelConfig
import hashlib
import fitz  # PyMuPDF
import docx
import tiktoken


class OllamaEmbeddings:
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            if response.status_code == 200:
                embeddings.append(response.json()["embedding"])
            else:
                raise Exception(f"Failed to get embedding: {response.text}")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class DocumentProcessor:
    def __init__(self, chunk_size: int = 3000, chunk_overlap: int = 400):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def load_document(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.pdf':
            return self._extract_pdf(file_path)
        elif ext == '.docx':
            return self._extract_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _extract_pdf(self, file_path: str) -> str:
        text = ""
        try:
            doc = fitz.open(file_path)
            for page_num in range(doc.page_count):
                page = doc[page_num]
                # Try text extraction first (fast)
                page_text = page.get_text()
                if not page_text.strip():
                    # Try OCR if available
                    try:
                        pix = page.get_pixmap()
                        ocr_text = page.get_textpage_ocr().extractText()
                        page_text = ocr_text
                    except Exception as ocr_e:
                        print(f"Warning: OCR failed for page {page_num + 1} in {file_path}: {ocr_e}")
                        print("Note: Install tesseract-ocr for image-based PDF support")
                        page_text = ""
                text += page_text + "\n"
            doc.close()
        except Exception as e:
            print(f"Warning: Could not extract text from {file_path}: {e}")
        return text
    
    def _extract_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        full_text = []
        
        # Extract paragraphs
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text.append(paragraph.text)
        
        # Extract tables
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text:
                        row_text.append(cell_text)
                if row_text:
                    full_text.append(" | ".join(row_text))
        
        # Extract headers and footers from each section
        for section in doc.sections:
            # Extract headers
            if section.header:
                for paragraph in section.header.paragraphs:
                    if paragraph.text.strip():
                        full_text.append(f"[HEADER] {paragraph.text}")
            
            # Extract footers
            if section.footer:
                for paragraph in section.footer.paragraphs:
                    if paragraph.text.strip():
                        full_text.append(f"[FOOTER] {paragraph.text}")
        
        extracted_text = "\n".join(full_text)
        print(f"Extracted {len(extracted_text)} characters from DOCX")
        return extracted_text
    
    def chunk_text(self, text: str) -> List[str]:
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks


class RAGSystem:
    def __init__(self, persist_directory: str = "./vector_db", llm_model: str = "auto"):
        self.persist_directory = persist_directory
        self.vectordb = SimpleVectorDB(persist_directory)
        self.embeddings = OllamaEmbeddings()
        self.processor = DocumentProcessor()
        self.llm_url = "http://localhost:11434/api/generate"
        
        # Auto-select model based on system resources
        if llm_model == "auto":
            self.llm_model = ModelConfig.get_recommended_model('balanced')
            print(f"Auto-selected model: {self.llm_model}")
        else:
            self.llm_model = llm_model
    
    def add_document(self, file_path: str) -> None:
        print(f"Processing document: {file_path}")
        
        # Load and process document
        text = self.processor.load_document(file_path)
        chunks = self.processor.chunk_text(text)
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(chunks)
        
        # Create unique IDs for chunks
        doc_id = hashlib.md5(file_path.encode()).hexdigest()
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        
        # Prepare metadata
        metadatas = [{"source": file_path, "chunk_id": i} for i in range(len(chunks))]
        
        # Add to vector database
        self.vectordb.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(chunks)} chunks from {file_path}")
    
    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """Analyze document extraction and chunking without adding to vector DB"""
        print(f"Analyzing document: {file_path}")
        
        # Load and process document
        text = self.processor.load_document(file_path)
        chunks = self.processor.chunk_text(text)
        
        # Calculate statistics
        total_chars = len(text)
        total_tokens = len(self.processor.encoding.encode(text))
        chunk_sizes = [len(chunk) for chunk in chunks]
        chunk_tokens = [len(self.processor.encoding.encode(chunk)) for chunk in chunks]
        
        analysis = {
            'file_path': file_path,
            'total_characters': total_chars,
            'total_tokens': total_tokens,
            'num_chunks': len(chunks),
            'avg_chunk_chars': sum(chunk_sizes) / len(chunk_sizes) if chunks else 0,
            'avg_chunk_tokens': sum(chunk_tokens) / len(chunk_tokens) if chunks else 0,
            'min_chunk_tokens': min(chunk_tokens) if chunks else 0,
            'max_chunk_tokens': max(chunk_tokens) if chunks else 0,
            'chunk_token_distribution': chunk_tokens
        }
        
        print(f"Document Analysis:")
        print(f"  File: {file_path}")
        print(f"  Total characters: {total_chars:,}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Number of chunks: {len(chunks)}")
        print(f"  Average chunk size: {analysis['avg_chunk_tokens']:.0f} tokens")
        print(f"  Chunk size range: {analysis['min_chunk_tokens']}-{analysis['max_chunk_tokens']} tokens")
        
        return analysis
    
    def search(self, query: str, n_results: int = 15) -> List[Dict[str, Any]]:
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in vector database
        results = self.vectordb.query(query_embedding, n_results)
        
        # Format results
        search_results = []
        for i in range(len(results['documents'][0])):
            search_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return search_results
    
    def generate_response(self, query: str, context: str, model: str = None, timeout: int = 200) -> str:
        if model is None:
            model = self.llm_model
        prompt = f"""Answer this question using only the context provided. Be concise.

Context: {context}

Question: {query}

Answer:"""
        
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 1500
                    }
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                raise Exception(f"Failed to generate response: {response.text}")
        except requests.exceptions.Timeout:
            return "Response timed out. The model may be overloaded or the query too complex."
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def query(self, question: str, n_results: int = 15, debug: bool = False, timeout: int = 200) -> Dict[str, Any]:
        # Search for relevant documents
        search_results = self.search(question, n_results)
        
        # Prepare context from search results
        context = "\n\n".join([result['content'] for result in search_results])
        
        # Generate response using LLM
        response = self.generate_response(question, context, timeout=timeout)
        
        result = {
            'question': question,
            'answer': response,
            'sources': [result['metadata']['source'] for result in search_results],
            'context_chunks': len(search_results)
        }
        
        if debug:
            print(f"\n=== DEBUG: Retrieved {len(search_results)} chunks ===")
            for i, chunk_result in enumerate(search_results):
                print(f"\nChunk {i+1} (distance: {chunk_result['distance']:.4f}):")
                print(f"Source: {chunk_result['metadata']['source']}")
                print(f"Content preview: {chunk_result['content'][:200]}...")
                print(f"Full content length: {len(chunk_result['content'])} characters")
            
            print(f"\n=== Total context length: {len(context)} characters ===")
            result['debug_info'] = {
                'retrieved_chunks': search_results,
                'total_context_length': len(context)
            }
        
        return result