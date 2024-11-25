# BioLaySumm-BiOzU: Lay Summarization for Scientific Articles 

<img src="images/BiOzU_LEGO.png" alt="BiOzU Logo" width="100">
<img src="images/OzU_LEGO.png" alt="OzU Logo" width="100">

## Overview

**BioLaySumm-BiOzU** is a state-of-the-art model designed for the BioLaySumm competition. The project aims to generate high-quality lay summaries for scientific articles from eLife and PLOS datasets. By leveraging cutting-edge NLP techniques such as quantized T5-Small and BioBERT with a Retrieval-Augmented Generation (RAG) approach, BioLaySumm-BiOzU bridges the gap between complex scientific texts and layman-friendly summaries.

---

## Features

### Key Components:
1. **T5-Small Model with Quantization**
   - Efficient and optimized for resource-constrained environments.
   - Fine-tuned to handle scientific text summarization.

2. **RAG Approach with BioBERT**
   - Utilizes BioBERT embeddings to identify semantically significant sentences.
   - Ranks sentences based on their cosine similarity with the article's Title, Abstract, and keywords.

3. **Dynamic Prompt Creation**
   - The input to T5-Small is crafted using:
     - The article's **Title** and **Abstract**.
     - 10 key sentences selected based on similarity metrics:
       - 2 sentences with the highest similarity to the Title.
       - 3 sentences with the highest similarity to the Abstract.
       - 5 sentences with the highest overall similarity to the combined features.

### Scalable Design
- The model seamlessly processes diverse article lengths and ensures relevance and clarity in lay summaries.

---

## Workflow

1. **Data Processing**
   - Articles from eLife and PLOS datasets are preprocessed to extract key sections.
   - Keywords, Title, and Abstract are identified for similarity comparison.

2. **Sentence Embedding Extraction**
   - BioBERT generates embeddings for each sentence in the article.

3. **Cosine Similarity Calculation**
   - Similarity scores are computed between sentences and the Title, Abstract, and keywords.

4. **Sentence Selection**
   - Top 10 sentences are chosen based on predefined criteria for the summarization input.

5. **Lay Summary Generation**
   - T5-Small model generates the lay summary using the crafted prompt.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/BioLaySumm-BiOzU.git
cd BioLaySumm-BiOzU
poetry install
