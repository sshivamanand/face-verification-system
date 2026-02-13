# Face Verification System

**A real-time biometric face verification system powered by ArcFace embeddings and PostgreSQL vector search**

---

## Overview

This system leverages deep learning embeddings and vector similarity search to identify individuals from webcam input in real-time. It extracts 512-dimensional facial embeddings using the ArcFace model and performs efficient cosine similarity searches in PostgreSQL with the pgvector extension.

---

## Key Features

-  **Real-time face capture** using OpenCV
-  **Deep learning embeddings** using ArcFace (via DeepFace)
-  **512-dimensional vectors** for high accuracy
-  **PostgreSQL vector database** with pgvector extension
-  **Cosine similarity search** for nearest neighbor retrieval
-  **Complete pipeline**: Registration + Verification
-  **Duplicate detection** using similarity thresholds
-  **Modular architecture** for easy scaling

---


1. **Capture** face via webcam
2. **Extract** 512-dimensional ArcFace embedding
3. **Store** embedding in PostgreSQL (registration) OR **Compare** with stored vectors (verification)
4. **Return** identity match with similarity score

---

## Tech Stack

### Languages
- **Python** 3.8+

### Libraries & Frameworks
- **DeepFace** - ArcFace model for face embeddings
- **OpenCV** - Real-time video capture and processing
- **NumPy** - Numerical operations on vectors
- **psycopg2** - PostgreSQL database adapter

### Database
- **PostgreSQL** 13+
- **pgvector** - Vector similarity search extension

### Core Concepts
- Deep Metric Learning
- Face Embeddings
- Vector Similarity Search
- Cosine Distance
- Nearest Neighbor Retrieval

---

##  How It Works

###  Registration Flow

1. Capture face image using webcam
2. Extract 512D embedding using ArcFace
3. Check if similar face already exists (duplicate detection)
4. If unique, store embedding in PostgreSQL

###  Verification Flow

1. Capture face image from webcam
2. Extract 512D embedding using ArcFace
3. Compare against all database embeddings using cosine similarity
4. Return closest match with similarity score

**Similarity Calculation:**

**Decision Thresholds:**
- **≥ 80%** → Match Found
- **65–79%** → Possible Match
- **< 65%** → No Match

---

## 📁 Project Structure

```
face-verification-system/
│
├── main.py              # Verification pipeline
├── register.py          # Face registration script
├── verifier.py          # Embedding extraction logic
├── db.py                # Database connection and queries
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

---

##  Installation

### 1. Clone Repository

```bash
git clone https://github.com/sshivamanand/face-verification-system.git
cd face-verification-system
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PostgreSQL and pgvector

**PostgreSQL:**
- Download from [postgresql.org/download](https://www.postgresql.org/download/)
- Install and start the PostgreSQL service

**pgvector Extension:**

Connect to your PostgreSQL database and run:

```sql
CREATE EXTENSION vector;
```

### 5. Create Database Table

```sql
CREATE TABLE faces (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    embedding vector(512) NOT NULL
);
```

**Optional: Create index for faster searches**

```sql
CREATE INDEX ON faces USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

---

##  Usage

### Register a New Face

```bash
python register.py
```

**What happens:**
1. Opens webcam
2. Prompts you to press `SPACE` to capture
3. Asks for name input
4. Stores face embedding in database

**Example Output:**
```
Webcam opened
Press SPACE to capture image

Extracting embedding...
Enter name: Shivam

✓ Face registered successfully!
```

---

### Verify Face

```bash
python main.py
```

**What happens:**
1. Opens webcam
2. Press `SPACE` to capture face
3. Searches database for matches
4. Returns similarity score and identity

**Example Output:**
```
Webcam opened
Press SPACE to capture image

Extracting embedding...
Searching in vector database...

Result:
----------
Similarity: 91.23%
✓ MATCH FOUND: Shivam
```

---

## ⚡ Performance

| Metric | Value |
|--------|-------|
| **Embedding Dimension** | 512 |
| **Inference Latency** | ~200–500 ms per verification (CPU) |
| **Database Retrieval** | O(log n) with vector indexing |
| **Similarity Metric** | Cosine Distance |

---

### Database Schema

```sql
Table: faces
├── id (SERIAL PRIMARY KEY)
├── name (TEXT)
└── embedding (vector(512))
```

---

## License

This project is licensed under the MIT License.

---
