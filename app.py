import streamlit as st
import os
import pickle
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
import faiss
from sentence_transformers import SentenceTransformer
import hashlib
from typing import List, Tuple
import re

# Konfigurasi halaman
st.set_page_config(
	page_title="Chatbot Informatif DTSEN (RAG Pattern)",
	page_icon="🤖",
	layout="wide"
)

# Session state
if 'messages' not in st.session_state:
	st.session_state.messages = []
if 'vector_store' not in st.session_state:
	st.session_state.vector_store = None
if 'model' not in st.session_state:
	st.session_state.model = None

class KurniaChatbotRAG:
	def __init__(self):
		self.model_name = "paraphrase-multilingual-MiniLM-L12-v2"
		self.model = None
		self.index = None
		self.documents = []
		self.metadata = []

	def load_model(self):
		if st.session_state.model is None:
			with st.spinner("🔄 Memuat model AI..."):
				st.session_state.model = SentenceTransformer(self.model_name)
				st.rerun()
		self.model = st.session_state.model

	def extract_text_from_pdf(self, file_path: str) -> str:
		text = ""
		try:
			reader = PdfReader(file_path)
			for page in reader.pages:
				text += page.extract_text() + "\n"
		except Exception as e:
			st.error(f"Error membaca PDF: {str(e)}")
		return text

	def extract_text_from_docx(self, file_path: str) -> str:
		text = ""
		try:
			doc = Document(file_path)
			for paragraph in doc.paragraphs:
				text += paragraph.text + "\n"
		except Exception as e:
			st.error(f"Error membaca DOCX: {str(e)}")
		return text
    
	def process_documents(self, documents_dir: str = "sop_documents"):
		if not os.path.exists(documents_dir):
			os.makedirs(documents_dir)
			st.info(f"📁 Directory '{documents_dir}' dibuat. Upload dokumen DTSEN.")
			return False

		vector_store_path = "vector_store/faiss_index.index"
		metadata_path = "vector_store/metadata.pkl"

		if os.path.exists(vector_store_path) and os.path.exists(metadata_path):
			with st.spinner("🔄 Memuat database SOP..."):
				self.index = faiss.read_index(vector_store_path)
				with open(metadata_path, 'rb') as f:
					data = pickle.load(f)
					self.documents = data['documents']
					self.metadata = data['metadata']
				st.success(f"✅ Database dimuat: {len(self.documents)} dokumen")
				return True

		supported_extensions = ['.pdf', '.docx', '.txt']
		all_files = []

		for root, dirs, files in os.walk(documents_dir):
			for file in files:
				if any(file.lower().endswith(ext) for ext in supported_extensions):
					all_files.append(os.path.join(root, file))
        
			if not all_files:
				st.warning("⚠️ Tidak ada dokumen ditemukan.")
				return False

			with st.spinner(f"📄 Memproses {len(all_files)} dokumen..."):
				documents = []
				metadata = []

				for file_path in all_files:
					if file_path.endswith('.pdf'):
						text = self.extract_text_from_pdf(file_path)
					elif file_path.endswith('.docx'):
						text = self.extract_text_from_docx(file_path)
					else:
						with open(file_path, 'r', encoding='utf-8') as f:
							text = f.read()

					if text.strip():
						chunks = self.split_text_into_chunks(text)
						for i, chunk in enumerate(chunks):
							documents.append(chunk)
							metadata.append({
								'file': os.path.basename(file_path),
								'chunk': i,
								'total_chunks': len(chunks)
							})

				if documents:
					embeddings = self.model.encode(documents, show_progress_bar=False)
					dimension = embeddings.shape[1]
					self.index = faiss.IndexFlatL2(dimension)
					self.index.add(np.array(embeddings).astype('float32'))

					self.documents = documents
					self.metadata = metadata

					os.makedirs("vector_store", exist_ok=True)
					faiss.write_index(self.index, vector_store_path)
					with open(metadata_path, 'wb') as f:
						pickle.dump({
							'documents': documents,
							'metadata': metadata
						}, f)

					st.success(f"✅ Database dibuat: {len(documents)} chunk")
					return True
		return False

	def split_text_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
		words = text.split()
		chunks = []
		for i in range(0, len(words), chunk_size - overlap):
			chunk = ' '.join(words[i:i + chunk_size])
			chunks.append(chunk)
			if i + chunk_size >= len(words):
				break
		return chunks

	def search_documents(self, query: str, k: int = 3) -> List[Tuple[str, dict]]:
		if self.index is None or len(self.documents) == 0:
			return []

		query_embedding = self.model.encode([query])
		distances, indices = self.index.search(
			np.array(query_embedding).astype('float32'), 
			k
		)

		results = []
		for idx, distance in zip(indices[0], distances[0]):
			if idx < len(self.documents):
				results.append((
					self.documents[idx],
					{**self.metadata[idx], 'similarity': 1/(1+distance)}
				))
		return results

	# ==================== RAG PATTERN IMPLEMENTATION ====================
	def generate_response(self, query: str) -> str:
		"""Implementasi RAG Pattern Lengkap"""
		# STEP 1: RETRIEVAL - Ambil dokumen relevan
		relevant_chunks = self.search_documents(query, k=3)

		if not relevant_chunks:
			return self._fallback_response(query)

		# STEP 2: AUGMENTATION - Format konteks
		context = self._format_context(relevant_chunks)

		# STEP 3: GENERATION - Buat response
		return self._generate_rag_response(query, context, relevant_chunks)

	def _format_context(self, relevant_chunks):
		"""Format konteks untuk RAG"""
		context_parts = []
		for i, (chunk_text, meta) in enumerate(relevant_chunks, 1):
			source = meta['file']
			similarity = f"(Relevansi: {meta['similarity']:.2f})"
			context_parts.append(f"【Sumber {i}】{source} {similarity}\n{chunk_text}")
		return "\n\n" + "═"*50 + "\n\n".join(context_parts) + "\n" + "═"*50

	def _generate_rag_response(self, query, context, relevant_chunks):
		"""Generasi response dengan RAG pattern"""
		# Header response
		response = "## 📋 **HASIL PENCARIAN SOP DTSEN**\n\n"
		response += f"**Pertanyaan:** {query}\n\n"
		response += "**📚 Dokumen Referensi:**\n"

		# List sumber
		for i, (_, meta) in enumerate(relevant_chunks, 1):
			response += f"{i}. `{meta['file']}` (chunk {meta['chunk']+1}/{meta['total_chunks']})\n"

		response += "\n**🔍 Ringkasan:**\n"

		# Ringkasan konten
		all_text = " ".join([chunk[0] for chunk in relevant_chunks])
		summary = self._extract_summary(all_text)
		response += f"{summary}\n\n"

		# Langkah-langkah jika ada
		steps = self._extract_procedures(all_text)
		if steps:
			response += "**📝 Langkah-Langkah Prosedur:**\n"
			for step in steps[:5]:
				response += f"• {step}\n"
			response += "\n"

		# Informasi tambahan
		response += "**💡 Informasi Penting:**\n"
		response += "- Pastikan semua dokumen sudah divalidasi\n"
		response += "- Periksa kelengkapan sesuai checklist\n"
		response += "- Ikuti timeline yang ditentukan\n\n"

		# Footer dengan metadata RAG
		response += "---\n"
		response += f"*✅ Di-generate menggunakan RAG Pattern | {len(relevant_chunks)} dokumen referensi*\n"
		response += "*🔧 Sistem: Retrieval → Augmentation → Generation*"

		return response

	def _extract_summary(self, text, max_sentences=3):
		"""Ekstrak ringkasan dari teks"""
		sentences = re.split(r'[.!?]+', text)
		sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
		return ". ".join(sentences[:max_sentences]) + "."
    
	def _extract_procedures(self, text):
		"""Ekstrak langkah-langkah prosedur"""
		procedures = []
		lines = text.split('\n')
		for line in lines:
			line_lower = line.lower()
			if any(keyword in line_lower for keyword in ['langkah', 'tahap', 'prosedur', 'step', 'proses']):
				if len(line) > 10 and len(line) < 200:
					procedures.append(line.strip())
		return list(set(procedures))[:5]

	def _fallback_response(self, query):
		"""Response fallback"""
		return f"""
## ⚠️ **INFORMASI TIDAK DITEMUKAN**

Pertanyaan **"{query}"** tidak ditemukan dalam dokumen tentang DTSEN.

**Saran Pencarian:**
1. Gunakan kata kunci yang lebih spesifik
2. Periksa nama dokumen yang tepat
3. Konsultasi langsung dengan admin DTSEN

**Contoh Pertanyaan yang Bisa Dicoba:**
- "Prosedur input data primer"
- "Validasi dokumen pendukung" 
- "Timeline pengiriman data"
"""
	# ==================== END RAG PATTERN ====================

def main():
	st.title("🤖 Chatbot Informatif DTSEN dengan RAG Pattern")
	st.markdown("**Sistem Retrieval-Augmented Generation**")
	st.markdown("---")

	with st.sidebar:
		st.header("⚙️ Konfigurasi")

		st.subheader("📤 Upload SOP")
		uploaded_files = st.file_uploader(
			"Pilih file PDF/DOCX",
			type=['pdf', 'docx', 'txt'],
			accept_multiple_files=True
		)

		if uploaded_files:
			os.makedirs("sop_documents", exist_ok=True)
			for uploaded_file in uploaded_files:
				file_path = os.path.join("sop_documents", uploaded_file.name)
				with open(file_path, "wb") as f:
					f.write(uploaded_file.getbuffer())
			st.success(f"✅ {len(uploaded_files)} file diupload")
			st.rerun()
		
		st.divider()

		# ========== PIPELINE STATUS - SELALU AKTIF ==========
		st.subheader("📊 Pipeline Status")

		# Status model embedding
		status_col1, status_col2 = st.columns(2)
		with status_col1:
			if st.session_state.model is not None:
				st.success("🟢 Model Ready")
			else:
				st.error("🔴 Model Offline")
		
		with status_col2:
			if st.session_state.get('vector_store'):
				st.success("🟢 DB Ready")
			else:
				st.warning("🟡 DB Loading")

		# Real-time metrics
		if st.session_state.model:
			metrics_col1, metrics_col2 = st.columns(2)
			with metrics_col1:
				st.metric("Embedding Model", "SBERT-multilingual")
				st.metric("Dimensions", "384")

			with metrics_col2:
				if st.session_state.get('vector_store'):
					chatbot = st.session_state.vector_store
					doc_count = len(chatbot.documents)
					st.metric("SOP Documents", doc_count)
					st.metric("Vectors", chatbot.index.ntotal if chatbot.index else 0)
				else:
					st.metric("SOP Documents", "0")
					st.metric("Vectors", "0")
		
		# Pipeline visualization
		st.divider()
		st.subheader("🔄 RAG Pipeline")
		
		# Show pipeline steps with status
		pipeline_steps = [
			("1. Document Processing", "success" if st.session_state.get('vector_store') else "pending"),
			("2. Embedding Generation", "success" if st.session_state.model else "pending"),
			("3. Vector Storage", "success" if st.session_state.get('vector_store') else "pending"),
			("4. Query Processing", "success" if len(st.session_state.messages) > 0 else "ready"),
			("5. Response Generation", "success" if len(st.session_state.messages) > 1 else "ready")
		]

		for step, status in pipeline_steps:
			if status == "success":
				st.markdown(f"✅ {step}")
			elif status == "pending":
				st.markdown(f"⏳ {step}")
			else:
				st.markdown(f"📝 {step}")

		# Active stats
		st.divider()
		st.subheader("📈 Active Stats")
		st.metric("Chat Messages", len(st.session_state.messages))

		if st.session_state.messages:
				last_query = st.session_state.messages[-1]["content"] 
				if len(last_query) > 30:
						st.caption(f"Last query: {last_query[:30]}...")
		# ========== END PIPELINE STATUS ==========

	# Inisialisasi
	chatbot = KurniaChatbotRAG()
	chatbot.load_model()

	# Setup knowledge base
	if st.session_state.vector_store is None:
		with st.spinner("🔄 Setup RAG Pipeline..."):
			if chatbot.process_documents():
				st.session_state.vector_store = chatbot
				st.success("✅ RAG Pipeline siap!")
			else:
				st.warning("Upload dokumen SOP di sidebar")

	# Chat interface
	st.header("💬 Tanya Tentang DTSEN")

	# Contoh query
	st.caption("Contoh: 'Bagaimana prosedur validasi data?'")

	# Chat history
	for msg in st.session_state.messages: #[-10:]:
		with st.chat_message(msg["role"]):
			st.markdown(msg["content"])

	# Input
	if prompt := st.chat_input("Tulis pertanyaan..."):
		st.session_state.messages.append({"role": "user", "content": prompt})

		with st.chat_message("user"):
			st.markdown(prompt)

		with st.chat_message("assistant"):
			with st.spinner("🔍 RAG Processing..."):
				if st.session_state.vector_store:
					response = st.session_state.vector_store.generate_response(prompt)
				else:
					response = "System belum siap. Upload dokumen dulu."

				st.markdown(response)

		st.session_state.messages.append({"role": "assistant", "content": response})

	# Visual pipeline
	st.markdown("---")
	st.subheader("🔄 Real-time Status")
	if st.session_state.vector_store and st.session_state.messages:
		last_query = st.session_state.messages[-1]["content"] if st.session_state.messages[-1]["role"] == "user" else ""
		if last_query:
			with st.expander("Lihat RAG Process"):
				st.write("**Query:**", last_query)
				chatbot = st.session_state.vector_store
				results = chatbot.search_documents(last_query, k=2)
				if results:
					st.write("**Retrieved:**", len(results), "dokumen")
					for i, (text, meta) in enumerate(results):
						st.write(f"{i+1}. {meta['file']} (score: {meta['similarity']:.3f})")

	# Bagian informasi
	st.markdown("---")
	col1, col2, col3 = st.columns(3)

	with col1:
		st.info("**📌 Fitur:**\n- Pencarian SOP DTSEN\n- Multi-dokumen\n- Bahasa Indonesia")

	with col2:
		st.info("**🔧 Teknologi:**\n- Sentence Transformers\n- FAISS Vector Search\n- Streamlit UI")

	with col3:
		st.info("**🎯 Tujuan:**\n- Mempermudah akses SOP\n- Konsistensi informasi\n- Efisiensi waktu")

if __name__ == "__main__":
	main()