# Resume-Job Matching System

A comprehensive Python application that matches resumes from a dataset with job postings scraped from Indeed, providing similarity scores and rankings to help identify the best candidates for specific positions.

## Features

- **Resume Processing**: Loads and processes resume data from CSV files
- **Job Scraping**: Scrapes job postings from Indeed (with fallback to sample data)
- **Multiple Matching Algorithms**: 
  - TF-IDF similarity
  - Keyword matching
  - Skill-based matching
  - Combined scoring
- **Two Interface Options**:
  - Command-line interface
  - Web interface using Streamlit
- **Visualization**: Interactive charts and graphs for score analysis
- **Top Candidate Ranking**: Shows top 10-12 matching resumes with detailed scores

## Installation

### Quick Setup

1. Clone or download the project files
2. Run the setup script:
   ```bash
   python setup.py
   ```

### Manual Installation

1. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn requests beautifulsoup4 nltk lxml html5lib streamlit plotly
   ```

2. Download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Usage

### Prepare Your Data

1. Download your resume dataset CSV file
2. Place it in the project directory
3. Ensure it has columns: `ID`, `Resume_str`, `Resume_html`, `Category`

### Command Line Interface

Run the main script:
```bash
python resume_matcher.py
```

Follow the prompts to:
- Enter job title to search for
- Enter location (optional)
- View matching results for each job found

### Web Interface (Recommended)

1. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Open your browser and go to `http://localhost:8501`

3. Upload your resume CSV file

4. Configure job search parameters

5. Click "Start Job Matching" to begin

## How It Works

### Data Processing
- Cleans and preprocesses resume text
- Removes HTML tags, special characters, and stopwords
- Applies lemmatization for better text analysis

### Job Scraping
- Scrapes job postings from Indeed based on search criteria
- Extracts job title, company, location, and description
- Handles rate limiting and error recovery

### Matching Algorithm
The system uses three different scoring methods:

1. **TF-IDF Similarity (50% weight)**
   - Computes cosine similarity between job description and resume vectors
   - Captures semantic similarity between documents

2. **Keyword Matching (30% weight)**
   - Calculates Jaccard similarity between job and resume keywords
   - Measures direct word overlap

3. **Skill-Based Matching (20% weight)**
   - Matches specific skills from predefined categories
   - Categories: Technical, Soft Skills, Business, Education, Experience

### Final Score
Combined score = (0.5 × TF-IDF) + (0.3 × Keyword) + (0.2 × Skill)

## File Structure

```
resume-job-matcher/
├── resume_matcher.py          # Main command-line application
├── streamlit_app.py          # Web interface application
├── setup.py                  # Setup script
├── requirements.txt          # Package dependencies
├── README.md                 # This file
└── Resume.csv               # Your resume dataset (place here)
```

## Configuration

### Skill Categories
You can modify the skill keywords in the `skill_keywords` dictionary:
- `technical`: Programming languages, frameworks, tools
- `soft`: Communication, leadership, teamwork skills
- `business`: Management, strategy, operations
- `education`: Degrees, certifications, training
- `experience`: Seniority levels, years of experience

### Scoring Weights
Adjust the weights in the `match_resumes_to_job` method:
```python
combined_score = (0.5 * tfidf_score + 0.3 * keyword_score + 0.2 * skill_score)
```

## Sample Output

### Command Line
```
JOB POSTING DETAILS
================================================================================
Title: Senior Data Scientist
Company: Tech Corp
Location: San Francisco, CA
Description: We are seeking an experienced Data Scientist to join our team...

TOP MATCHING RESUMES
================================================================================

1. RESUME ID: 16852973
   Category: INFORMATION-TECHNOLOGY
   Combined Score: 0.847
   TF-IDF Score: 0.823
   Keyword Score: 0.756
   Skill Score: 0.912
   Resume Preview: Data Scientist with 5+ years of experience in machine learning...
```

### Web Interface
- Interactive dashboard with charts and graphs
- Filterable results
- Detailed score breakdowns
- Resume preview functionality

## Troubleshooting

### Common Issues

1. **"No CSV file found"**
   - Make sure your resume CSV file is in the project directory
   - Check that the file has the correct column names

2. **"Error scraping jobs"**
   - Indeed may be blocking requests; the system will use sample data
   - Try different search terms or check your internet connection

3. **"NLTK data not found"**
   - Run the setup script again
   - Manually download NLTK data using the commands above

4. **"Import error"**
   - Check that all packages are installed correctly
   - Try reinstalling packages using the setup script

### Performance Tips

- For large datasets (>5000 resumes), consider using a subset for testing
- Adjust `max_features` in TfidfVectorizer for better performance
- Use fewer job postings for initial testing

## Limitations

- Indeed scraping may be blocked; fallback sample data is provided
- Processing time increases with dataset size
- Requires internet connection for job scraping
- Text-based matching may not capture all nuances of job requirements

## Future Enhancements

- Add support for PDF resume parsing
- Implement deep learning models for better semantic matching
- Add support for multiple job boards
- Include candidate ranking explanations
- Add export functionality for results

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

## License

This project is for educational and research purposes. Please respect the terms of service of job boards when scraping data.
