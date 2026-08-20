# Voice-Enabled RAG System

## Overview

This project is developed for the Goa 2026 AI Hackathon.

The application accepts voice input, converts it into text, retrieves relevant information from the MSMARCO-XI dataset using a vector database, and generates grounded answers using a Retrieval-Augmented Generation (RAG) pipeline.

## Features

- Voice-to-Text using Sarvam or ElevenLabs
- Advanced document chunking
- Vector database retrieval
- LLM-powered answer generation
- Guardrails for safe and grounded responses
- Latency analytics (P50, P70, P100)
- Streamlit/FastAPI interface

## Tech Stack

- Python
- FastAPI
- Streamlit
- ChromaDB / FAISS
- Sentence Transformers
- Hugging Face
- LangChain

## Project Structure

backend/
frontend/
data/
vector_db/
notebooks/
latency/

## Team

Goa 2026 AI Hackathon Team
