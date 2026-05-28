# CorollaIQ 🚗

RAG-powered chatbot that answers questions about Australian-market Toyota Corolla brochures from 2010 to 2025, using Claude, Chroma, LangChain, Hugging Face embeddings, and a Gradio interface deployed on Hugging Face Spaces.

👉🏽 **Live demo:** [CorollaIQ on Hugging Face Spaces](https://huggingface.co/spaces/sri-bandara/corolla-iq)  

---

## Overview 📍

- **Input:** Natural language questions about Toyota Corolla brochures
- **Output:** Answers about trims, specs, features, fuel economy, safety, and model differences
- **Documents:** Australian-market Toyota Corolla brochures from 2010–2025
- **RAG Pipeline:** PDF loading, text splitting, embeddings, vector search, retrieved-context prompting
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store:** Chroma
- **LLM:** Claude Sonnet
- **Interface:** Gradio chatbot
- **Deployment:** Hugging Face Spaces

---

## Data Source 📊

The knowledge base for this project was built from Australian-market Toyota Corolla brochures covering model years from 2010 to 2025. These brochures contain information about Corolla variants, body styles, trims, specifications, standard features, safety equipment, technology, fuel economy, and model differences.

Each brochure is loaded as a PDF, split into smaller text chunks, embedded using a local Hugging Face sentence-transformer model, and stored in a Chroma vector database. When a user asks a question, the chatbot retrieves the most relevant brochure chunks and sends them to Claude as context for generating an answer.

---

## Process ⚙️


First, the Corolla brochure PDFs are loaded from the `brochures/` folder using LangChain’s PDF loader. Each brochure page is tagged with metadata, including the source file and model year extracted from the filename.

The documents are then split into overlapping chunks using `RecursiveCharacterTextSplitter`. These chunks are converted into embeddings using `sentence-transformers/all-MiniLM-L6-v2`, a lightweight local embedding model suitable for semantic search. The embeddings are stored in Chroma, which acts as the vector database for retrieval.

When the user asks a question, the app checks whether a year such as 2015, 2020, or 2024 appears in the query. If a year is detected, retrieval is filtered to brochure chunks from that year. If no year is detected, the retriever searches across all brochure documents. Maximal Marginal Relevance search is used to retrieve relevant but varied chunks.

The retrieved brochure context, current question, and recent chat history are then passed to Claude. Claude is instructed to answer using the brochure context and to avoid making up information when the retrieved documents are insufficient.

Pricing questions are handled separately. If the user asks about price, value, fair price, market value, or whether a Corolla is overpriced or underpriced, CorollaIQ redirects the user to my own Pricing model hosted on Hugging Face.

---

## Assumptions and Limitations 🚧

This project assumes that the brochure PDFs contain enough information to answer the user’s question. If a brochure section is missing, poorly extracted, or spread across multiple pages, retrieval may not always return the complete information needed for a perfect answer.

The chatbot is designed for Australian-market Toyota Corolla brochures only. It should not be treated as a general automotive assistant or a live vehicle database. Features, trims, fuel economy, and specifications can vary by year, body style, and grade, so users should specify the model year and trim where possible.

Because the app relies on PDF text extraction, some brochure tables or visual layouts may not be represented perfectly in the retrieved text. This can affect answers for detailed trim breakdowns or feature comparisons.

---

## Demo 🔮

<img width="1467" height="826" alt="Screenshot 2026-05-29 at 12 51 17 am" src="https://github.com/user-attachments/assets/5a42bebd-94ba-4909-a36e-2e8f6d3f3c4a" />

