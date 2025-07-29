#!/usr/bin/env python3
"""
Resume-Job Matching System
A complete system to match resumes with job postings and rank them by relevance.
"""

import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class ResumeJobMatcher:
    def __init__(self, resume_csv_path: str):
        """
        Initialize the Resume-Job Matcher
        
        Args:
            resume_csv_path: Path to the resume CSV file
        """
        self.resume_df = None
        self.job_postings = []
        self.tfidf_vectorizer = None
        self.resume_vectors = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load resume data
        self.load_resume_data(resume_csv_path)
        
        # Skills keywords for different categories
        self.skill_keywords = {
            'technical': ['python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular', 
                         'node.js', 'django', 'flask', 'tensorflow', 'pytorch', 'machine learning',
                         'data science', 'aws', 'azure', 'docker', 'kubernetes', 'git', 'linux'],
            'soft': ['communication', 'leadership', 'teamwork', 'problem solving', 'analytical',
                    'creative', 'adaptable', 'organized', 'detail-oriented', 'collaborative'],
            'business': ['management', 'strategy', 'marketing', 'sales', 'finance', 'accounting',
                        'operations', 'project management', 'business development', 'consulting'],
            'education': ['bachelor', 'master', 'phd', 'degree', 'certification', 'training'],
            'experience': ['years', 'experience', 'senior', 'junior', 'manager', 'director', 'lead']
        }
    
    def load_resume_data(self, csv_path: str):
        """Load and preprocess resume data"""
        print("Loading resume data...")
        try:
            self.resume_df = pd.read_csv(csv_path)
            print(f"Loaded {len(self.resume_df)} resumes")
            
            # Clean resume text
            self.resume_df['cleaned_resume'] = self.resume_df['Resume_str'].apply(self.clean_text)
            
            # Create TF-IDF vectors for all resumes
            self.create_resume_vectors()
            
        except Exception as e:
            print(f"Error loading resume data: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def create_resume_vectors(self):
        """Create TF-IDF vectors for all resumes"""
        print("Creating resume vectors...")
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # Fit and transform resume texts
        self.resume_vectors = self.tfidf_vectorizer.fit_transform(
            self.resume_df['cleaned_resume']
        )
        
        print(f"Created vectors with {self.resume_vectors.shape[1]} features")
    
    def scrape_indeed_jobs(self, job_title: str, location: str = "", num_jobs: int = 10) -> List[Dict]:
        """
        Scrape job postings from Indeed
        
        Args:
            job_title: Job title to search for
            location: Location to search in
            num_jobs: Number of jobs to scrape
            
        Returns:
            List of job posting dictionaries
        """
        print(f"Scraping Indeed for '{job_title}' jobs...")
        
        jobs = []
        base_url = "https://www.indeed.com/jobs"
        
        # Headers to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        params = {
            'q': job_title,
            'l': location,
            'start': 0
        }
        
        try:
            for start in range(0, min(num_jobs, 100), 10):  # Indeed shows 10 jobs per page
                params['start'] = start
                
                response = requests.get(base_url, params=params, headers=headers)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find job cards
                job_cards = soup.find_all('div', class_='job_seen_beacon')
                
                for card in job_cards:
                    if len(jobs) >= num_jobs:
                        break
                    
                    try:
                        # Extract job details
                        title_elem = card.find('h2', class_='jobTitle')
                        title = title_elem.text.strip() if title_elem else "N/A"
                        
                        company_elem = card.find('span', class_='companyName')
                        company = company_elem.text.strip() if company_elem else "N/A"
                        
                        location_elem = card.find('div', class_='companyLocation')
                        job_location = location_elem.text.strip() if location_elem else "N/A"
                        
                        summary_elem = card.find('div', class_='summary')
                        summary = summary_elem.text.strip() if summary_elem else ""
                        
                        # Try to get job link for more details
                        link_elem = card.find('a', {'data-jk': True})
                        job_link = f"https://www.indeed.com/viewjob?jk={link_elem['data-jk']}" if link_elem else ""
                        
                        jobs.append({
                            'title': title,
                            'company': company,
                            'location': job_location,
                            'summary': summary,
                            'link': job_link,
                            'full_description': summary  # We'll use summary as description for now
                        })
                        
                    except Exception as e:
                        print(f"Error parsing job card: {e}")
                        continue
                
                # Rate limiting
                time.sleep(1)
                
                if len(jobs) >= num_jobs:
                    break
                    
        except Exception as e:
            print(f"Error scraping jobs: {e}")
            # Return sample jobs if scraping fails
            return self.get_sample_jobs(job_title)
        
        print(f"Scraped {len(jobs)} jobs")
        return jobs
    
    def get_sample_jobs(self, job_title: str) -> List[Dict]:
        """Return sample job postings for testing"""
        sample_jobs = [
            {
                'title': f'Senior {job_title}',
                'company': 'Tech Corp',
                'location': 'San Francisco, CA',
                'summary': f'We are seeking an experienced {job_title} to join our team. Must have 5+ years of experience in relevant technologies including Python, SQL, and cloud platforms.',
                'link': 'https://example.com/job1',
                'full_description': f'We are seeking an experienced {job_title} to join our team. Must have 5+ years of experience in relevant technologies including Python, SQL, and cloud platforms. Strong communication skills and ability to work in a team environment required.'
            },
            {
                'title': f'{job_title} Manager',
                'company': 'Innovation Ltd',
                'location': 'New York, NY',
                'summary': f'Looking for a {job_title} manager with leadership experience. Must have strong background in project management and team leadership.',
                'link': 'https://example.com/job2',
                'full_description': f'Looking for a {job_title} manager with leadership experience. Must have strong background in project management and team leadership. MBA preferred but not required.'
            }
        ]
        return sample_jobs
    
    def calculate_similarity_score(self, job_description: str, resume_idx: int) -> float:
        """Calculate similarity score between job and resume"""
        
        # Clean job description
        cleaned_job = self.clean_text(job_description)
        
        # Transform job description using fitted vectorizer
        job_vector = self.tfidf_vectorizer.transform([cleaned_job])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(job_vector, self.resume_vectors[resume_idx]).flatten()[0]
        
        return similarity
    
    def calculate_keyword_match_score(self, job_description: str, resume_text: str) -> float:
        """Calculate keyword matching score"""
        
        job_words = set(self.clean_text(job_description).split())
        resume_words = set(self.clean_text(resume_text).split())
        
        # Calculate overlap
        common_words = job_words.intersection(resume_words)
        
        if len(job_words) == 0:
            return 0.0
        
        # Calculate Jaccard similarity
        jaccard_score = len(common_words) / len(job_words.union(resume_words))
        
        return jaccard_score
    
    def calculate_skill_match_score(self, job_description: str, resume_text: str) -> float:
        """Calculate skill-based matching score"""
        
        job_text = job_description.lower()
        resume_text = resume_text.lower()
        
        skill_scores = []
        
        for category, skills in self.skill_keywords.items():
            job_skills = sum(1 for skill in skills if skill in job_text)
            resume_skills = sum(1 for skill in skills if skill in resume_text)
            
            if job_skills > 0:
                category_score = min(resume_skills / job_skills, 1.0)
                skill_scores.append(category_score)
        
        return np.mean(skill_scores) if skill_scores else 0.0
    
    def match_resumes_to_job(self, job_description: str, top_k: int = 12) -> List[Dict]:
        """
        Match resumes to a job posting and return top matches
        
        Args:
            job_description: Job posting description
            top_k: Number of top matches to return
            
        Returns:
            List of top matching resumes with scores
        """
        print(f"Matching resumes to job posting...")
        
        matches = []
        
        for idx, row in self.resume_df.iterrows():
            # Calculate different similarity scores
            tfidf_score = self.calculate_similarity_score(job_description, idx)
            keyword_score = self.calculate_keyword_match_score(job_description, row['Resume_str'])
            skill_score = self.calculate_skill_match_score(job_description, row['Resume_str'])
            
            # Combined score (weighted average)
            combined_score = (0.5 * tfidf_score + 0.3 * keyword_score + 0.2 * skill_score)
            
            matches.append({
                'resume_id': row['ID'],
                'category': row['Category'],
                'tfidf_score': tfidf_score,
                'keyword_score': keyword_score,
                'skill_score': skill_score,
                'combined_score': combined_score,
                'resume_text': row['Resume_str'][:500] + "..." if len(row['Resume_str']) > 500 else row['Resume_str']
            })
        
        # Sort by combined score and return top matches
        matches.sort(key=lambda x: x['combined_score'], reverse=True)
        
        print(f"Found {len(matches)} matches")
        return matches[:top_k]
    
    def display_results(self, job_info: Dict, matches: List[Dict]):
        """Display matching results"""
        print("\n" + "="*80)
        print("JOB POSTING DETAILS")
        print("="*80)
        print(f"Title: {job_info['title']}")
        print(f"Company: {job_info['company']}")
        print(f"Location: {job_info['location']}")
        print(f"Description: {job_info['full_description'][:200]}...")
        print(f"Link: {job_info['link']}")
        
        print("\n" + "="*80)
        print("TOP MATCHING RESUMES")
        print("="*80)
        
        for i, match in enumerate(matches, 1):
            print(f"\n{i}. RESUME ID: {match['resume_id']}")
            print(f"   Category: {match['category']}")
            print(f"   Combined Score: {match['combined_score']:.3f}")
            print(f"   TF-IDF Score: {match['tfidf_score']:.3f}")
            print(f"   Keyword Score: {match['keyword_score']:.3f}")
            print(f"   Skill Score: {match['skill_score']:.3f}")
            print(f"   Resume Preview: {match['resume_text'][:150]}...")
            print("-" * 50)
    
    def run_matching_pipeline(self, job_title: str, location: str = "", num_jobs: int = 3):
        """Run the complete matching pipeline"""
        print("Starting Resume-Job Matching Pipeline...")
        
        # Step 1: Scrape job postings
        jobs = self.scrape_indeed_jobs(job_title, location, num_jobs)
        
        if not jobs:
            print("No jobs found. Exiting.")
            return
        
        # Step 2: Match resumes to each job
        for i, job in enumerate(jobs, 1):
            print(f"\n{'='*20} PROCESSING JOB {i}/{len(jobs)} {'='*20}")
            
            # Get top matches for this job
            matches = self.match_resumes_to_job(job['full_description'])
            
            # Display results
            self.display_results(job, matches)
            
            # Ask user if they want to continue to next job
            if i < len(jobs):
                user_input = input(f"\nPress Enter to continue to next job, or 'q' to quit: ")
                if user_input.lower() == 'q':
                    break

def main():
    """Main function to run the resume matching system"""
    
    # Configuration
    RESUME_CSV_PATH = "Resume.csv"  # Update this path
    
    print("Welcome to the Resume-Job Matching System!")
    print("="*50)
    
    # Check if resume file exists
    if not os.path.exists(RESUME_CSV_PATH):
        print(f"Error: Resume file '{RESUME_CSV_PATH}' not found.")
        print("Please make sure the resume CSV file is in the current directory.")
        return
    
    try:
        # Initialize the matcher
        matcher = ResumeJobMatcher(RESUME_CSV_PATH)
        
        # Get user input
        job_title = input("Enter job title to search for: ").strip()
        location = input("Enter location (optional): ").strip()
        
        if not job_title:
            print("Job title is required!")
            return
        
        # Run the matching pipeline
        matcher.run_matching_pipeline(job_title, location)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your input and try again.")

if __name__ == "__main__":
    main()
