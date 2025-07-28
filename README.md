# Adobe_Round 1B

This is the submission for the Adobe India Hackathon Round 1B for the team **CodeTrinetra**.

In an age where document overload and ambiguity hinder productivity, our solution stands out by enabling persona-driven extraction of the most relevant content from unstructured PDF collections. This is especially valuable for users trying to derive actionable insights from large sets of documents based on specific roles and tasks. Our model understands the intent, evaluates document context semantically, and delivers high-quality JSON outputs with filtered, ranked insights. We encourage you to explore our solution and see how context-aware intelligence can transform information retrieval from documents.


## Approach

The goal is to extract semantically relevant information from PDFs, tailored to a specific role and task. Our system performs an intelligent scan of all documents, ranks text blocks by importance using vector similarity, and outputs only the most relevant parts in a clean, structured format.

We achieve this through:

- **Understanding the user’s goal**  
  We combine the role and task from the input into a simple, clear query that guides the document search.

- **Breaking down the documents**  
  Each PDF is carefully split into smaller chunks or blocks of text to focus on meaningful sections.

- **Smart text comparison**  
  We use a pre-trained transformer model to understand the meaning of both the query and the text blocks.

- **Finding the most relevant parts**  
  The system compares how closely each block matches the query and ranks them by relevance.

- **Creating clear, useful output**  
  Only the most important parts are selected and saved into a structured JSON file with helpful information like page number and source file.


The system runs entirely offline, without any network dependencies, and completes processing within strict resource and timing constraints.

## Key Features

- Persona-based content analysis  
- Importance ranking of extracted sections  
- Multi-collection document processing  
- Structured JSON output with metadata  


## Workflow

1. **Input Parsing**  
   - The system begins by reading `challenge1b_input.json`.
   - This JSON file contains:
     - The **role** (e.g., “Food Contractor”)
     - The **job_to_be_done** (e.g., “Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items.”)
     - The list of **PDF filenames** to be processed.
   - These fields are preprocessed into a semantic query for downstream filtering.

2. **Text Extraction**  
   - We use [`PyMuPDF`](https://pymupdf.readthedocs.io/en/latest/) (`fitz`) for parsing PDF files.
   - Each document is scanned page-by-page.
   - Extracted text is broken down into blocks or paragraphs while preserving layout fidelity using PyMuPDF’s layout-based tokenization.
   - Each block is assigned metadata such as page number and position within the page.

3. **Semantic Matching**  
   - The role and task are concatenated to form a natural language query.
   - This query is embedded into a dense vector representation using the [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model.
   - Each block of extracted text from the PDFs is also embedded using the same model.
   - This ensures that both query and document data are in the same semantic space.

4. **Relevance Scoring**  
   - We compute cosine similarity between the query embedding and each block embedding.
   - The blocks are ranked based on similarity scores.
   - A filtering threshold is applied to retain only those blocks that are contextually relevant to the persona and their job-to-be-done.
   - This step ensures that noise and irrelevant content are excluded.

5. **Filtering & Output Generation**  
   - The top-scoring blocks are compiled into a structured output format.
   - Output is written to `challenge1b_output.json`.
   - Each JSON contains persona, job_to_be_done, input_documents, processing_timestamp, extracted_sections and sub_section_analysis.
   - All outputs are generated offline, with no internet access.

### Output JSON Structure
```json
{
  "metadata": {
    "input_documents": [
      "Breakfast Ideas.pdf",
      "Dinner Ideas - Mains_1.pdf",
      "Dinner Ideas - Sides_1.pdf"
    ],
    "persona": "food contractor",
    "job_to_be_done": "Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items.",
    "processing_timestamp": "2025-07-27T11:54:33.175696"
  },
  "extracted_sections": [
    {
      "document": "Breakfast Ideas.pdf",
      "section_title": "Veggie Wrap Ingredients:",
      "importance_rank": 1,
      "page_number": 13
    },
    {
      "document": "Dinner Ideas - Sides_1.pdf",
      "section_title": "Ingredients:",
      "importance_rank": 2,
      "page_number": 2
    }
  ],
  "subsection_analysis": [
    {
      "document": "Dinner Ideas - Sides_1.pdf",
      "refined_text": "o Arrange meats, cheeses, olives, and vegetables on a platter. o Serve with breadsticks.",
      "page_number": 3
    },
    {
      "document": "Dinner Ideas - Sides_4.pdf",
      "refined_text": "o Serve as a dip or side dish. Som Tum (Green Papaya Salad)",
      "page_number": 15
    }
  ]
}
```


## Models and Libraries Used

- **`sentence-transformers/all-MiniLM-L6-v2`**  
  - A 90MB transformer model optimized for semantic similarity tasks.
  - Embeds both queries and document content into dense vector space.
  - Lightweight and fast enough to run under CPU constraints.
  - Pre-downloaded and included in the Docker container to ensure offline compatibility.

- **`PyMuPDF (fitz)`**  
  - High-performance PDF parser used for layout-preserving extraction.
  - Provides precise block-level segmentation with page positioning.

- **`torch`**  
  - Backend library required by `sentence-transformers` for tensor operations and model inference.

- **`transformers`**  
  - HuggingFace’s interface used to load and configure the transformer model architecture.

- **`numpy`**  
  - Used for vector computations including cosine similarity scoring.

All libraries are installed via `requirements.txt` and handled inside the Docker container.

## Compliance with Hackathon Constraints

This solution adheres to all constraints outlined in the Adobe Hackathon Round 1B guidelines:

| Constraint           | Our Compliance                                          |
|----------------------|----------------------------------------------------------|
| Execution Time       | Completes under 10 seconds for a 50-page PDF            |
| Model Size           | Uses a 90MB model (`all-MiniLM-L6-v2`)                  |
| Network Access       | No internet access required; runs fully offline         |
| Runtime              | Runs on CPU-only (amd64) environment                    |
| System Requirements  | Tested on 8 CPU / 16 GB RAM environment                 |

## How to Build and Run

To evaluate our solution, please use the exact instructions below as per the hackathon guidelines.

### Build the Docker image
```bash
docker build --platform linux/amd64 -t codetrinetra:round1b .
```

### Run the Docker container

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none codetrinetra:round1b
```

## Team Members

- **Diya Anna Varghese**
- **Gubba Pavani**
- **Merin Theres Jose**



