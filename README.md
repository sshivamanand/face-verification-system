# Face Verification System

> A biometric face verification system using ArcFace embeddings and PostgreSQL vector search

A face verification system that combines deep learning facial embeddings with efficient vector similarity search. Built with the ArcFace model for feature extraction and PostgreSQL's pgvector extension for scalable identity matching.

---

## Features

- **Real-time Face Registration** - Capture and register faces using webcam
- **High-Accuracy Verification** - ArcFace model with 512-dimensional embeddings
- **Fast Vector Search** - PostgreSQL pgvector with optimized IVFFlat indexing
- **Confidence Scoring** - Cosine similarity-based match confidence
- **Production-Ready** - Modular architecture with proper separation of concerns
- **Easy Configuration** - Environment-based settings management

---

## 🛠️ Technologies

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.8+ |
| **Deep Learning** | DeepFace (ArcFace model) |
| **Computer Vision** | OpenCV, RetinaFace |
| **Database** | PostgreSQL 13+ |
| **Vector Search** | pgvector extension |

---

## 📁 Project Structure

```
face-verification-system/
│
├── main.py                      # Face verification script
├── register.py                  # Face registration script
├── verifier.py                  # Embedding extraction and search logic
├── db.py                        # PostgreSQL connection handler
├── requirements.txt             # Python dependencies
├── face_verification.ipynb      # Interactive demo notebook
├── .env                         # Environment variables (not committed)
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- PostgreSQL 13 or higher
- Webcam (for registration and verification)

### 1. Database Setup

**Install PostgreSQL and pgvector extension:**

```bash
# On Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# Install pgvector
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

**Create and configure database:**

```sql
-- Connect to PostgreSQL
psql -U postgres

-- Create database
CREATE DATABASE face_verification_db;
\c face_verification_db;

-- Enable pgvector extension
CREATE EXTENSION vector;

-- Create faces table
CREATE TABLE faces (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    embedding VECTOR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create optimized vector index
CREATE INDEX faces_embedding_idx
ON faces
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Update statistics
ANALYZE faces;
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=face_verification_db
DB_USER=your_username
DB_PASSWORD=your_password
```

### 3. Installation

**Clone the repository:**

```bash
git clone https://github.com/sshivamanand/face-verification-system.git
cd face-verification-system
```

**Create virtual environment:**

```bash
python -m venv venv

# On Linux/Mac
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Register a New Face

```bash
python register.py
```

**Steps:**
1. Webcam window opens automatically
2. Position your face in the frame
3. Press **SPACE** to capture
4. Enter the person's name when prompted
5. Embedding is extracted and stored in PostgreSQL

**Output:**
```
Opening webcam for registration...
Press SPACE to capture face, ESC to exit
Face captured!
Enter person name: John Doe
Extracting facial embedding...
Storing in database...
✓ Successfully registered: John Doe
```

### Verify a Face

```bash
python main.py
```

**Steps:**
1. Webcam window opens automatically
2. Position your face in the frame
3. Press **SPACE** to capture
4. System searches the database
5. Results displayed with confidence score

**Example Output:**
```
Opening webcam for verification...
Press SPACE to verify face, ESC to exit
Face captured!
Extracting embedding...
Searching in vector database...

╔══════════════════════════════╗
║      VERIFICATION RESULT     ║
╠══════════════════════════════╣
║ Similarity: 94.23%           ║
║ Decision: ✓ MATCH FOUND      ║
║ Identity: John Doe           ║
╚══════════════════════════════╝
```

---

## Key Concepts

### ArcFace Embeddings
- **512-dimensional** feature vectors
- State-of-the-art face recognition accuracy
- Trained on millions of face images
- Angular margin loss for better discrimination

### Vector Similarity Search
- **Cosine similarity** for measuring face similarity
- **IVFFlat indexing** for fast approximate nearest neighbor search
- Optimized for high-dimensional vectors
- Scalable to millions of faces

### Production Optimizations
- Database connection pooling
- Efficient indexing strategies
- Modular code architecture
- Environment-based configuration

---

## License

This project is licensed under the MIT License.
