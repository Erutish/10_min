Multilingual Bangla RAG System
A robust Retrieval-Augmented Generation (RAG) pipeline for answering Bangla and English questions over the "HSC26 Bangla 1st Paper" textbook using advanced OCR, text extraction, embedding, vector search, and LLM-based answer generation.

Setup Guide
Colab/Jupyter Environment:
Clone the repo or upload scripts to your workspace.


Prepare Google Drive for input/output (PDF, intermediate files, etc.).


Install dependencies:

 python
CopyEdit
!pip install -U pinecone sentence-transformers openai easyocr pdf2image pymupdf opencv-python pillow


For OCR model weights:

 python
CopyEdit
!mkdir -p /root/.EasyOCR/model
!wget -q -O /root/.EasyOCR/model/craft_mlt_25k.pth "https://huggingface.co/xiaoyao9184/easyocr/resolve/master/craft_mlt_25k.pth"
!wget -q -O /root/.EasyOCR/model/bengali.pth "https://huggingface.co/xiaoyao9184/easyocr/resolve/master/bengali.pth"


Set your Pinecone and OpenAI API keys in the scripts.


Upload the Bangla PDF (e.g., Book.pdf) to Drive.


 Used Tools, Libraries, and Packages
OCR & Text Extraction:


easyocr (for Bangla OCR, robust on scans)


pdf2image, pymupdf (for direct PDF text extraction)


opencv-python, pillow (image pre-processing)


LLM Integration:


openai (for GPT-4.1-mini or GPT-4 generation)


Embedding & Retrieval:


sentence-transformers (intfloat/multilingual-e5-large)


pinecone (vector database)


Evaluation:


pandas (analysis, CSVs)


sentence-transformers (cosine similarity scoring)



📚 Sample Queries and Outputs
User Query (Bangla)
Model Answer
Ground Truth
Cosine Similarity
‘অপরিচিতা' গল্পটি প্রথম কোথায় প্রকাশিত হয়?
‘অপরিচিতা’ গল্পটি প্রথম প্রকাশিত হয় প্রমথ চৌধুরী সম্পাদিত মাসিক ‘সবুজপত্র’ পত্রিকার ১৩২১ বঙ্গাব্দের কার্তিক সংখ্যায়।
সবুজপত্র
0.760399580001831
অনুপমের প্রকৃত অভিভাবক কে?
অনুপমের প্রকৃত অভিভাবক তার মামা।
মামাকে
0.818320751190185
 How was Horish define in book?
রসিক মনের মানুষ
হরিশকে বইয়ে এমনভাবে সংজ্ঞায়িত করা হয়েছে যে, তিনি উদ্দীপকের শাফিক চরিত্রের প্রতিধ্বনি, যিনি উচ্ছল, রসিক এবং যেকোনো পরিবেশে দ্রুত নিজেকে মানিয়ে নিতে পারেন। এছাড়া, 'অপরিচিতা' গল্পে হরিশের গুণের মধ্যে আসর জমানো এবং ভাষাটা অত্যন্ত আঁটসাঁট হওয়ার বর্ণনা পাওয়া যায়। অর্থাৎ, হরিশ একজন প্রাণবন্ত, সামাজিকভাবে সক্রিয় এবং বুদ্ধিদীপ্ত চরিত্র হিসেবে চিত্রিত।
0.802231550216674

























Full result and evaluation matrices are saved as CSV in  rag_eval_results.csv.

📑 API Documentation
Base URL:
http://localhost:8000

POST /ask
Description:
 Submit a question (Bangla or English). The API retrieves relevant textbook passages and uses the RAG model to answer, grounded only in retrieved context.


Request:


Content-Type: application/json


Example:

 json
CopyEdit
{ "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?" }


Response:


Content-Type: application/json


Example:

 json
CopyEdit
{
  "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "answer": "শুম্ভুনাথ",
  "retrieved_chunks": [
    "[Passage 1] ...",
    "[Passage 2] ...",
    "... up to 5 passages ..."
  ]
}


Behavior:


The model checks passages first, then MCQs/lists/tables.


If the answer is not found, it returns: 'Not found in the book.'



How to run:
pip install fastapi uvicorn pinecone sentence-transformers openai


uvicorn fastapi:app --reload


See /docs for Swagger UI.




📊 Evaluation Matrix
We use two main evaluation strategies:
Answer Similarity (Cosine):


Compute semantic similarity (cosine score using E5) between LLM-generated answer and ground truth.


Results in rag_eval_results.csv.


Retriever Relevance:


For each question, check if the correct answer is present in the top-K retrieved chunks (via string match or semantic similarity).


Results in rag_relevance_eval_results.csv.


 Reflection & Project Questions
1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?
I used easyocr for robust OCR (handles noisy scans, Bangla scripts) and pymupdf for direct text extraction (faster if the PDF has Unicode text).
 Challenges: Bangla textbooks are often scanned images (not Unicode text). This led to garbled output from direct PDF extraction. OCR fixed this but sometimes misrecognized words—hence, both outputs were cleaned and reconciled with an LLM prompt.

2. What chunking strategy did you choose (e.g., paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?
I chunked data by page (per .txt), following the OCR/PDF extraction, then cleaned these into well-formed paragraphs and sections. This keeps context but keeps chunks short enough for LLM and vector search, aiding semantic retrieval.



3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?
We used intfloat/multilingual-e5-large from sentence-transformers, a state-of-the-art model for cross-lingual semantic retrieval. It maps both Bangla and English questions/passages to the same embedding space, enabling multilingual RAG.

4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?
For retrieval, we use cosine similarity between E5 embeddings of the question and all document chunks, implemented via Pinecone’s efficient vector search. This allows semantic matching even with different wording.

5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?
Both question and chunks are embedded using E5 with appropriate prompts ("query: ...", "passage: ...").


This alignment ensures fair, meaningful comparison.


For vague queries, top-K results may not be directly relevant.
 Potential solution: Increase K, improve chunking, or add fallback strategies.



6. Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?
Most results are highly relevant (cosine similarity near 1 for well-formed queries and answers).
 For borderline cases:
Improving chunking (paragraphs, QAs, MCQs)


Better OCR correction


Larger/more granular chunk set


Using more context windows per question
 can further enhance accuracy and groundedness.




