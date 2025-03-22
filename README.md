# GraphXtractRAG (Graph-based Extractive RAG)

<img src="images/BıOzU_LEGO.png" alt="BiOzU Logo" width="100"> <img src="images/OzU_LEGO.png" alt="OzU Logo" width="100">

### GraphXtractRAG: A Novel Query-Independent Approach to Lay Summarization Using Gemini and Graph-Based Extractive Techniques
**GraphXtractRAG** is a novel **Graph Neural Network (GNN)-based retrieval-augmented generation (RAG) framework** designed for generating query-independent summaries of scientific articles. Unlike traditional heuristic-based extractive methods, it dynamically learns sentence importance within a document's graph structure, enabling more accurate and context-aware sentence selection. Each sentence is represented as a node in a semantic graph, with edges capturing inter-sentence relationships. A trainable GNN-based selector extracts key sentences, which are then fed into a retrieval-enhanced generative model (Gemini) to generate fluent and coherent lay summaries. Trained using REINFORCE, GraphXtractRAG eliminates the need for extractive summary labels while achieving superior performance over PageRankRAG and SimilarityRAG on biomedical datasets (eLife & PLOS). This framework sets a new benchmark for lay summarization in biomedical informatics.

## 🚀 Features

- **🧠 GNN-Powered Sentence Selection:** Dynamically learns and identifies the most important sentences within a document's graph structure.
- **📄 Semantic Graph Representation:** Constructs a semantic graph where nodes represent sentences and edges represent inter-sentence relationships.
- **📝 Retrieval-Enhanced Generation:** Utilizes the Gemini model to generate fluent and coherent lay summaries based on selected key sentences.
- **⚡ Efficient and Scalable:** Capable of handling large-scale biomedical text corpora with improved performance over traditional methods.

---
## 🔧 Main Components

### 1. **🧠 Advanced Generation with Gemini Model**
   - 🚀 Utilizes the superior text understanding and generation capabilities of the Gemini model.
   - ✨ Provides improved performance, fluency, and adaptability in summarizing complex scientific texts.

### 2. **🔍 Query Independent RAG Architecture**
   - 🗂️ Directly leverages the article’s structural and semantic information without relying on external queries.
   - 🔄 Enhances the consistency and comprehensiveness of the generated lay summaries.
   
#### **🛠️ Methodology:**
   - **🧬 BioBERT for Sentence Embeddings**
     - 🛠️ Generates robust embeddings for each sentence.
     
   - **🧩 SimilarityRAG**
     - 🔗 Incorporates similarity with title embedding models to ensure that selected sentences align closely with the article’s core message.
     
   - **📈 PageRankRAG**
     - 🕸️ Constructs a graph representation of the article.
     - 📊 Employs the PageRank algorithm to rank sentences based on their connectivity and importance within the graph.
     
   - **🧠 GraphXtractRAG**
     - 🕸️ Constructs a graph representation of the article.
     - 🧠 Utilizes Graph-based Extractive Summary to identify and rank the most influential sentences.

### 3. **📝 Dynamic Input Creation**
   - 🧩 Merges the article’s graph-selected key sentences from **Title** and **All Sections** to form a dynamic prompt.
   - 🎯 Ensures the Gemini model processes critical content effectively for high-quality summary generation.

---

## 🛠️ Workflow

1. **📄 Data Preprocessing**
   - 🧹 Scientific articles are parsed to extract key sections such as the title, abstract, keywords, and more.
   - 🧠 Generates representative embeddings for each sentence using advanced embedding techniques.

2. **🕸️ Graph Construction**
   - 🛠️ Constructs a graph where nodes represent sentences and edges represent semantic similarities.

3. **🔍 RAG**
   - 🎯 The most relevant sentence detected with selected RAG approach.

4. **📝 Dynamic Input Creation**
   - 🧩 The selected sentences into a dynamic prompt tailored for the Gemini model.

5. **🧠 Summary Generation**
   - ✨ The Gemini model processes the dynamic prompt to generate a coherent, lay-friendly summary.

---

## 🛠 Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/CagataySavasli/GraphXtractRAG-BioLaySumm.git
cd BioLaySumm-BiOzU
poetry install
```

## 🤝 Contributions

Feel free to open an **issue**, submit a **pull request**, or discuss improvements!

📩 **For inquiries, reach out or create an issue.** 🚀

## 👨‍💻 Developers

- **Ahmet Çağatay Savaşlı** – Developer
- **Prof. Dr. Emre Sefer** – Advisor

*This project was developed within the [OzU Machine Learning in Finance and Bioinformatics Lab](https://ozu-mlfinbio-lab.github.io/).*