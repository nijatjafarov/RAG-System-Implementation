# RAG System Implementation

## Dataset Statistics
| Domain       | Source      | Count |
|--------------|-------------|-------|
| Law          | E-Qanun     | 1,500 |
| News         | Azertac     | 5,000 |
| Encyclopedia | AzWikipedia | 2,505 |

## Performance
### Retrieval
| Embedding Model    | MRR      | Recall@5 | nDCG@5   |
| ------------------ | -------- | -------- | -------- |
| Gemini Embedding 2 | **0.78** | **0.68** | **0.66** |
| BGE-M3             | 0.74     | 0.66     | 0.63     |
| Qwen3-Embedding    | 0.74     | 0.64     | 0.62     |
| Snowflake Arctic   | 0.73     | 0.65     | 0.61     |

### Generation
|Embedding Model   |LLM            |Semantic Similarity|LLM as a Judge|Abstention Accuracy|
|------------------|---------------|-------------------|--------------|-------------------|
|Snowflake Arctic  | DeepSeek-V3.2 | 0.73              | 0.77         | **0.78**          |
|Gemini Embedding 2| DeepSeek-V3.2 | 0.73              | 0.78         | 0.75              |
|Qwen3-Embedding   | DeepSeek-V3.2 | 0.73              | 0.77         | 0.74              |
|Qwen3-Embedding   | Gemini 3 Flash| 0.74              | 0.78         | 0.70              |
|BGE-M3            | DeepSeek-V3.2 | 0.73              | 0.75         | 0.73              |
|BGE-M3            | Gemini 3 Flash| 0.74              | 0.77         | 0.69              |
|Snowflake Arctic  | Gemini 3 Flash| 0.74              | 0.76         | 0.70              |
|Gemini Embedding 2| Gemini 3 Flash| **0.75**          | 0.78         | 0.67              |
|BGE-M3            | GPT-5.4       | **0.75**          | **0.79**     | 0.63              |
|Snowflake Arctic  | GPT-5.4       | 0.74              | 0.77         | 0.62              |
|Qwen3-Embedding   | GPT-5.4       | 0.74              | 0.76         | 0.61              |
|Gemini Embedding 2| GPT-5.4       | 0.74              | 0.78         | 0.58              |
|Snowflake Arctic  | Kimi K2.5     | 0.48              | 0.66         | 0.55              |
|Qwen3-Embedding   | Kimi K2.5     | 0.48              | 0.67         | 0.52              |
|BGE-M3            | Kimi K2.5     | 0.46              | 0.66         | 0.52              |
|Gemini Embedding 2| Kimi K2.5     | 0.46              | 0.66         | 0.47              |

## Usage
To run the view:
```bash
git clone https://github.com/nijatjafarov/AzRAGBench.git
cd AzRAGBench
pip install pandas streamlit
streamlit run scripts/view.py
```