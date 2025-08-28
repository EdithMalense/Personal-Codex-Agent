# Personal Codex Agent

A context-aware AI agent that answers questions about me as a candidate, built with RAG (Retrieval-Augmented Generation) and multiple personality modes.

🚀 **Live Demo**:  https://personal-codex-agent-4r8fbhlbqy48sv7gywvgj8.streamlit.app/ 
📹 **Video Walkthrough**: [Your demo video link here]

## Overview

This Personal Codex Agent represents me authentically by learning from my actual documents, code, and writing. It can answer questions about my background, skills, and experience while adapting its tone and style based on different modes.

## How it works
1. Upload CV + supporting docs (PDF/TXT/MD).
2. The app extracts text, chunks it, and builds a ChromaDB index with HuggingFace embeddings (local).
3. On each query, top-k chunks are retrieved using HuggingFace embeddings and sent to OpenAI LLM.
4. Responses cite sources by document title and chunk.

## Run locally
```bash
pip install streamlit chromadb pypdf openai torch transformers
export OPENAI_API_KEY=YOUR_KEY
streamlit run app.py
```

**Note:** Only `torch` and `transformers` are needed for embeddings - no heavy `sentence-transformers` package required!

## Deployment
- **Streamlit Cloud**: set `HF_API_TOKEN` as a secret.
- Note: First run will download the HuggingFace model (~90MB for all-MiniLM-L6-v2)

### Key Features

- **RAG Implementation**: Retrieves relevant context from my personal documents
- **Multiple Personality Modes**: Interview, storytelling, fast facts, and humble brag modes
- **Document Processing**: Handles PDFs, markdown, code files, and text documents
- **Real-time Chat Interface**: Built with Streamlit for easy interaction
- **Semantic Search**: Uses ChromaDB for intelligent document retrieval

## System Architecture

### Design Choices

1. **Tech Stack**:
   - **Frontend**: Streamlit (rapid prototyping, easy deployment)
   - **LLM**: OpenAI GPT-4 (high-quality responses)
   - **Vector DB**: ChromaDB (lightweight, local-first)
   - **Embeddings**: HuggingFace Transformers (embeddings - runs locally)

2. **RAG Pipeline**:
   - Document chunking with overlap for context preservation
   - Semantic similarity search for relevant context retrieval
   - Context-aware prompt engineering with retrieved documents

3. **Personality System**:
   - Mode-specific system prompts and temperature settings
   - Consistent first-person voice across all modes
   - Authentic representation based on actual documents

### Architecture Diagram
```
[User Query] → [Vector Search] → [Context Retrieval] → [LLM + Prompt] → [Response]
     ↓              ↓                   ↓                  ↓
[Streamlit UI] [ChromaDB] [Document Chunks] [Mode-specific Prompts]

```

## Sample Questions & Expected Answers

### "What kind of engineer are you?"
**Expected Response** (Interview Mode):
> I'm a [your specialty] engineer with [X years] of experience in [key areas]. My background spans [specific technologies/domains], and I'm particularly passionate about [what drives you]. I approach engineering problems with [your methodology], always focusing on [your priorities like scalability, user experience, etc.].


### "What are your strongest technical skills?"
**Expected Response** (Fast Facts Mode):
> • **Languages**: [Your top languages]
> • **Frameworks**: [Key frameworks you use]
> • **Tools**: [Development tools and platforms]
> • **Specialties**: [Your unique technical strengths]
> • **Recent Focus**: [What you've been learning/working on]

## Training Data

My agent learns from these authentic documents:

- **CV/Resume**: Professional background and experience
- **Blog Posts**: [Specific posts about your technical insights]
- **Project READMEs**: [Key projects that showcase your work]
- **Code Samples**: [Examples with thoughtful comments]
- **Personal Notes**: [Reflections on work style and values]

*All documents included in `/documents` folder*

## AI Collaboration Artifacts

### My Thinking Process

This project was built collaboratively with AI coding agents. Here's how:

#### Primary AI Agents Used
1. **Architecture Agent** (ChatGPT): System design and structure planning
2. **Implementation Agent** (GitHub Copilot): Code generation and debugging
3. **Documentation Agent** (Claude): README and comment generation

#### Key Collaboration Examples

**Prompt → AI Response → Final Implementation:**

```markdown
**Prompt**: "Create a RAG system with multiple personality modes"
**AI Response**: [Generated base RAG implementation]
**My Changes**: Added mode switching, customized prompts for my voice
**Commit**: `feat: Add personality modes with context-aware retrieval`
```

#### AI-Generated vs Manual Code Breakdown
- **AI Generated (~70%)**:
  - Base RAG implementation
  - Document processing pipeline  
  - Streamlit UI structure
  - Vector database integration

- **Manual Modifications (~30%)**:
  - Personality mode customization
  - Error handling improvements
  - UI/UX refinements
  - Personal prompt engineering

*Full conversation logs and prompt history available in `/ai_artifacts`*

## What I'd Improve with More Time

### Immediate Improvements (Next 4-8 hours)
- [ ] **Better Chunking Strategy**: Implement smart chunking that preserves context boundaries
- [ ] **Conversation Memory**: Add session-based chat history for follow-up questions
- [ ] **Source Attribution**: Show which documents informed each response
- [ ] **Advanced Modes**: Add more specialized modes (technical deep-dive, cultural fit, etc.)
- [ ] **Cleaner UI**: make the UI cleaner/ less complicated so it does easier for the user to use.


### Future Enhancements (Given more time)
- [ ] **Multi-modal Support**: Handle images, videos, and audio files
- [ ] **Dynamic Learning**: Allow real-time addition of new documents
- [ ] **Analytics Dashboard**: Track question patterns and response quality
- [ ] **Voice Interface**: Add speech-to-text and text-to-speech capabilities
- [ ] **Advanced RAG**: Implement graph-based retrieval or reranking models

### RAG Extensions
Currently using basic semantic similarity. Would extend with:
- **Hybrid Search**: Combine semantic and keyword search
- **Query Expansion**: Use LLM to generate alternative query phrasings  
- **Re-ranking**: Score and reorder results based on query intent
- **Hierarchical Retrieval**: Search at document and chunk levels

## Deployment
- **Streamlit Cloud**: set `HF_API_TOKEN` as a secret.
- Note: First run will download the HuggingFace model (~90MB for all-MiniLM-L6-v2)

### Deployment Process
1. Pushed to GitHub with secrets management
2. Connected Streamlit Cloud to repository
3. Added HF API key to Streamlit secrets
4. Automatic deployments on push to main branch

## Technical Details

### Performance Considerations
- Document chunking optimized for 1000-character chunks with 200-character overlap
- ChromaDB for fast in-memory vector search
- Streamlit caching for document processing

## Available Embedding Models
- sentence-transformers/all-MiniLM-L6-v2 (default, fast)
- sentence-transformers/all-mpnet-base-v2 (better quality)
- sentence-transformers/paraphrase-MiniLM-L6-v2
- sentence-transformers/distilbert-base-nli-mean-tokens

## Modes
- Interview, Personal storytelling, Fast facts, Humble brag, Self-reflection.

## Benefits of HuggingFace Embeddings
- No API costs for embeddings (runs locally)
- Works offline once model is downloaded
- Multiple model options for different quality/speed tradeoffs
- Embeddings are cached locally for faster rebuilds


### Security & Privacy
- API keys stored in environment variables
- No personal data logged or stored externally
- Documents processed locally in ChromaDB
- Temporary file handling with cleanup

## Bonus Features Implemented

- [x] **Multi-mode Personality System**: 4 distinct response modes
- [x] **Real-time Document Upload**: Easy dataset extension
- [x] **Sample Question Suggestions**: Guided interaction
- [ ] **Self-reflective Agent Mode** (planned)

## Demo Video

[Embed or link to your 5-minute walkthrough video]

**Video Covers**:
- Live demonstration of the agent
- Different personality modes in action
- Document upload and processing
- Sample questions and responses
- Technical architecture overview

---

**Built by**: Edith Malense  
**Contact**: edithkmalense24@gmail.com 
**Time Invested**: ~7 hours  
**AI Collaboration**: Heavy use of Claude, GitHub Copilot, and ChatGPT