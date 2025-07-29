#!/usr/bin/env python3
"""
Streamlit Web App for Resume-Job Matching System
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
@st.cache_data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

download_nltk_data()

class StreamlitResumeJobMatcher:
    def __init__(self):
        self.resume_df = None
        self.tfidf_vectorizer = None
        self.resume_vectors = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
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
    
    @st.cache_data
    def load_resume_data(_self, uploaded_file):
        """Load and preprocess resume data"""
        try:
            _self.resume_df = pd.read_csv(uploaded_file)
            
            # Clean resume text
            _self.resume_df['cleaned_resume'] = _self.resume_df['Resume_str'].apply(_self.clean_text)
            
            # Create TF-IDF vectors for all resumes
            _self.create_resume_vectors()
            
            return True
            
        except Exception as e:
            st.error(f"Error loading resume data: {e}")
            return False
    
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
    
    def scrape_indeed_jobs(self, job_title: str, location: str = "", num_jobs: int = 5) -> List[Dict]:
        """Scrape job postings from Indeed with progress bar"""
        
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
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            for i, start in enumerate(range(0, min(num_jobs, 50), 10)):
                params['start'] = start
                
                status_text.text(f'Scraping jobs... {len(jobs)}/{num_jobs}')
                progress_bar.progress(min(len(jobs) / num_jobs, 1.0))
                
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
                            'full_description': summary
                        })
                        
                    except Exception as e:
                        continue
                
                # Rate limiting
                time.sleep(1)
                
                if len(jobs) >= num_jobs:
                    break
                    
        except Exception as e:
            st.warning(f"Error scraping jobs: {e}")
            # Return sample jobs if scraping fails
            jobs = self.get_sample_jobs(job_title, num_jobs)
        
        progress_bar.progress(1.0)
        status_text.text(f'Scraped {len(jobs)} jobs successfully!')
        
        return jobs
    
    def get_sample_jobs(self, job_title: str, num_jobs: int = 5) -> List[Dict]:
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
            },
            {
                'title': f'Junior {job_title}',
                'company': 'StartUp Inc',
                'location': 'Austin, TX',
                'summary': f'Entry-level {job_title} position. Fresh graduates welcome. Training provided.',
                'link': 'https://example.com/job3',
                'full_description': f'Entry-level {job_title} position. Fresh graduates welcome. Training provided. Must have basic knowledge of relevant technologies.'
            }
        ]
        return sample_jobs[:num_jobs]
    
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
        """Match resumes to a job posting and return top matches"""
        
        matches = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_resumes = len(self.resume_df)
        
        for idx, row in self.resume_df.iterrows():
            # Update progress
            progress_bar.progress((idx + 1) / total_resumes)
            status_text.text(f'Analyzing resume {idx + 1}/{total_resumes}')
            
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
                'resume_text': row['Resume_str']
            })
        
        progress_bar.progress(1.0)
        status_text.text(f'Analysis complete! Found {len(matches)} matches')
        
        # Sort by combined score and return top matches
        matches.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return matches[:top_k]

def main():
    st.set_page_config(
        page_title="Resume-Job Matcher",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📄 Resume-Job Matching System")
    st.markdown("---")
    
    # Initialize session state
    if 'matcher' not in st.session_state:
        st.session_state.matcher = StreamlitResumeJobMatcher()
    
    if 'resume_loaded' not in st.session_state:
        st.session_state.resume_loaded = False
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Resume CSV File",
        type=['csv'],
        help="Upload the resume dataset CSV file"
    )
    
    if uploaded_file is not None:
        if not st.session_state.resume_loaded:
            with st.spinner("Loading resume data..."):
                if st.session_state.matcher.load_resume_data(uploaded_file):
                    st.session_state.resume_loaded = True
                    st.sidebar.success(f"✅ Loaded {len(st.session_state.matcher.resume_df)} resumes")
                else:
                    st.sidebar.error("❌ Failed to load resume data")
    
    # Main content
    if st.session_state.resume_loaded:
        # Dataset overview
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Resumes", len(st.session_state.matcher.resume_df))
        
        with col2:
            st.metric("Categories", st.session_state.matcher.resume_df['Category'].nunique())
        
        with col3:
            st.metric("Avg Resume Length", 
                     f"{st.session_state.matcher.resume_df['Resume_str'].str.len().mean():.0f} chars")
        
        # Category distribution chart
        st.subheader("📈 Resume Categories Distribution")
        
        category_counts = st.session_state.matcher.resume_df['Category'].value_counts()
        
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            labels={'x': 'Category', 'y': 'Count'},
            title="Number of Resumes by Category"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Job search section
        st.subheader("🔍 Job Search & Matching")
        
        col1, col2 = st.columns(2)
        
        with col1:
            job_title = st.text_input("Job Title", value="Data Scientist")
            location = st.text_input("Location (optional)", value="")
        
        with col2:
            num_jobs = st.slider("Number of Jobs to Scrape", 1, 10, 3)
            top_k = st.slider("Top Matches to Show", 5, 20, 12)
        
        if st.button("🚀 Start Job Matching", type="primary"):
            if job_title:
                # Step 1: Scrape jobs
                st.subheader("📋 Scraping Job Postings...")
                jobs = st.session_state.matcher.scrape_indeed_jobs(job_title, location, num_jobs)
                
                if jobs:
                    # Step 2: Match resumes to jobs
                    st.subheader("🎯 Matching Resumes to Jobs")
                    
                    # Create tabs for each job
                    job_tabs = st.tabs([f"Job {i+1}: {job['title'][:30]}..." for i, job in enumerate(jobs)])
                    
                    for i, (tab, job) in enumerate(zip(job_tabs, jobs)):
                        with tab:
                            # Display job details
                            st.markdown(f"### {job['title']}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Company:** {job['company']}")
                                st.write(f"**Location:** {job['location']}")
                            
                            with col2:
                                if job['link']:
                                    st.markdown(f"**Link:** [View Job]({job['link']})")
                            
                            st.write(f"**Description:** {job['full_description']}")
                            
                            st.markdown("---")
                            
                            # Match resumes
                            with st.spinner("Matching resumes..."):
                                matches = st.session_state.matcher.match_resumes_to_job(
                                    job['full_description'], top_k
                                )
                            
                            # Display matches
                            st.subheader(f"🏆 Top {len(matches)} Matching Resumes")
                            
                            # Create a dataframe for better display
                            match_df = pd.DataFrame(matches)
                            
                            # Display score distribution
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=list(range(1, len(matches) + 1)),
                                y=[m['combined_score'] for m in matches],
                                mode='lines+markers',
                                name='Combined Score',
                                line=dict(color='blue', width=2),
                                marker=dict(size=8)
                            ))
                            fig.update_layout(
                                title='Matching Scores Distribution',
                                xaxis_title='Resume Rank',
                                yaxis_title='Score',
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display detailed results
                            for j, match in enumerate(matches, 1):
                                with st.expander(f"#{j} - Resume ID: {match['resume_id']} (Score: {match['combined_score']:.3f})"):
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write(f"**Category:** {match['category']}")
                                        st.write(f"**Combined Score:** {match['combined_score']:.3f}")
                                        st.write(f"**TF-IDF Score:** {match['tfidf_score']:.3f}")
                                    
                                    with col2:
                                        st.write(f"**Keyword Score:** {match['keyword_score']:.3f}")
                                        st.write(f"**Skill Score:** {match['skill_score']:.3f}")
                                    
                                    st.write("**Resume Preview:**")
                                    st.text_area(
                                        f"Resume {match['resume_id']}",
                                        match['resume_text'][:500] + "...",
                                        height=100,
                                        key=f"resume_{i}_{j}"
                                    )
                            
                            # Score comparison chart
                            st.subheader("📊 Score Comparison")
                            
                            score_df = pd.DataFrame({
                                'Resume': [f"Resume {m['resume_id']}" for m in matches[:10]],
                                'TF-IDF': [m['tfidf_score'] for m in matches[:10]],
                                'Keyword': [m['keyword_score'] for m in matches[:10]],
                                'Skill': [m['skill_score'] for m in matches[:10]],
                                'Combined': [m['combined_score'] for m in matches[:10]]
                            })
                            
                            fig = px.bar(
                                score_df,
                                x='Resume',
                                y=['TF-IDF', 'Keyword', 'Skill', 'Combined'],
                                title="Score Breakdown for Top 10 Matches",
                                barmode='group'
                            )
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error("No jobs found. Please try a different search term.")
            
            else:
                st.error("Please enter a job title.")
    
    else:
        st.info("👆 Please upload a resume CSV file to get started.")
        
        # Show sample data format
        st.subheader("📋 Expected CSV Format")
        sample_data = pd.DataFrame({
            'ID': ['12345678', '87654321'],
            'Resume_str': ['Sample resume text here...', 'Another resume text...'],
            'Resume_html': ['<div>HTML content</div>', '<div>HTML content</div>'],
            'Category': ['INFORMATION-TECHNOLOGY', 'FINANCE']
        })
        st.dataframe(sample_data)

if __name__ == "__main__":
    main()