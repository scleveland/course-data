import os
import json
import time
import random
import concurrent.futures
from typing import List, Dict, Optional, Tuple, Type, Any, Callable
import PyPDF2
from vertexai.preview.generative_models import GenerativeModel
import vertexai
from google.api_core import exceptions as google_exceptions
from google.oauth2 import service_account
from concurrent.futures import ThreadPoolExecutor, as_completed

class GeminiCourseExtractor:
    def __init__(self, project_id: str, keyfile_path: str, location: str = "us-central1"):
        """Initialize the Gemini extractor with Vertex AI using a service account key file.
        
        Args:
            project_id: Your Google Cloud project ID
            keyfile_path: Path to the service account key JSON file
            location: Google Cloud region (default: us-central1)
        """
        self.project_id = project_id
        self.keyfile_path = keyfile_path
        self.location = location
        self.model = None
        self.credentials = None
        self.courses: List[Dict] = []
        self.current_institution: Optional[str] = None
        self.initialize_vertex_ai()
    
    def initialize_vertex_ai(self) -> None:
        """Initialize the Vertex AI with the Gemini model using service account credentials."""
        try:
            # Load the service account credentials
            self.credentials = service_account.Credentials.from_service_account_file(
                self.keyfile_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            
            # Initialize Vertex AI with the credentials
            vertexai.init(
                project=self.project_id,
                location=self.location,
                credentials=self.credentials
            )
            
            # Initialize the Gemini 2.5 Flash preview model
            self.model = GenerativeModel("gemini-2.5-flash-preview-05-20")
            print("Successfully initialized Vertex AI with service account credentials")
            
        except Exception as e:
            print(f"Error initializing Vertex AI: {e}")
            raise
    
    def retry_with_backoff(
        self,
        func: Callable,
        *args,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        exceptions: tuple[Type[Exception], ...] = (
            google_exceptions.ResourceExhausted,
            google_exceptions.ServiceUnavailable,
            google_exceptions.DeadlineExceeded,
            Exception  # Catch-all for other transient errors
        )
    ) -> Any:
        """Retry a function with exponential backoff."""
        delay = initial_delay
        for attempt in range(1, max_retries + 1):
            try:
                return func(*args)
            except exceptions as e:
                if attempt == max_retries:
                    print(f"Max retries ({max_retries}) reached. Last error: {e}")
                    raise
                
                # Add jitter to avoid thundering herd
                sleep_time = min(delay * (1 + random.random() * 0.1), max_delay)
                print(f"Attempt {attempt} failed with error: {e}. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                delay *= backoff_factor
                
        raise Exception("Max retries reached")

    def process_chunk(self, chunk_data: Tuple[int, str, str]) -> Tuple[int, List[Dict]]:
        """Process a single chunk of text and return its courses with retry logic."""
        chunk_num, chunk, filename = chunk_data
        print(f"Processing chunk {chunk_num}...")
        
        def _process() -> List[Dict]:
            return self.extract_courses_with_gemini(chunk, filename) or []
            
        try:
            chunk_courses = self.retry_with_backoff(
                _process,
                max_retries=3,
                initial_delay=2.0,
                backoff_factor=2.0
            )
            
            if chunk_courses:
                print(f"Extracted {len(chunk_courses)} courses from chunk {chunk_num}")
                return (chunk_num, chunk_courses)
            return (chunk_num, [])
            
        except Exception as e:
            print(f"Error in chunk {chunk_num} after retries: {e}")
            return (chunk_num, [])

    def process_pdf(self, filepath: str, max_workers: int = 4) -> List[Dict]:
        """Process a single PDF file and return extracted courses using parallel processing."""
        filename = os.path.basename(filepath)
        print(f"Processing: {filename}")
        
        # Read the entire PDF with retry
        def _read_pdf() -> str:
            text = ""
            try:
                with open(filepath, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                return text
            except Exception as e:
                print(f"Error reading PDF: {e}")
                return ""
        
        text = self.retry_with_backoff(_read_pdf, max_retries=3, initial_delay=1.0)
            
        if not text:
            print(f"Warning: Could not extract text from {filepath}")
            return []
            
        # Split text into chunks with overlap to prevent splitting courses
        chunk_size = 10000  # Smaller chunks for better parallelization
        overlap = 1000  # Overlap to prevent splitting courses
        chunks = []
        start = 0
        chunk_num = 1
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append((chunk_num, text[start:end], filename))
            start = end - overlap  # Overlap with next chunk
            chunk_num += 1
        
        all_courses = []
        completed_chunks = set()
        
        # Process chunks in parallel with rate limiting
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {}
            for chunk in chunks:
                future = executor.submit(self.process_chunk, chunk)
                future_to_chunk[future] = chunk[0]
                
                # Small delay between submissions to avoid overwhelming the API
                time.sleep(0.5)
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_num = future_to_chunk[future]
                try:
                    chunk_num, chunk_courses = future.result()
                    if chunk_courses:
                        all_courses.extend(chunk_courses)
                    completed_chunks.add(chunk_num)
                    print(f"Completed {len(completed_chunks)}/{len(chunks)} chunks")
                except Exception as e:
                    print(f"Error processing chunk {chunk_num} after submission: {e}")
        
        print(f"Completed processing {filename}. Extracted {len(all_courses)} courses.")
        return all_courses
    
    def read_entire_pdf(self, filepath: str) -> str:
        """Read the entire PDF file and return its text content."""
        text = ""
        try:
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    def extract_courses_with_gemini(self, text: str, filename: str) -> List[Dict]:
        """Use Gemini to extract course information from text."""
        prompt = """Extract all courses from the following university catalog text. 
        For each course, return a JSON array of objects with these fields:
        - course_id (e.g., 'ENG 100')
        - subject (the subject area code, e.g., 'ENG' for English, 'MATH' for Mathematics)
        - title (course title)
        - description (course description)
        - credits (number of credits, if mentioned)
        - prerequisites (if any)
        - program (the major, degree, or certificate program this course belongs to, if mentioned)
        - department (the academic department offering the course, e.g., 'Computer Science', 'English')
        - college (the college or school within the university that offers this course, e.g., 'College of Arts and Sciences')
        
        For the program field, identify the most relevant academic program.
        For the department field, identify the academic department that offers the course.
        For the college field, identify the college or school within the university.
        
        Return ONLY valid JSON, nothing else. Example:
        [
            {
                "course_id": "ENG 100",
                "title": "English Composition",
                "description": "...",
                "credits": 3,
                "prerequisites": "None",
                "program": "English BA",
                "department": "English Department",
                "college": "College of Arts and Sciences"
            }
        ]
        
        Here's the text to process:
        """
        
        # Set the institution based on filename
        self.set_institution_from_filename(filename)
        
        try:
            # Limit text size to avoid context window issues
            content = text[:8000]
            
            # Create the full prompt
            full_prompt = prompt + content
            
            # Generate content with error handling
            try:
                response = self.model.generate_content(full_prompt)
            except Exception as e:
                print(f"Error generating content: {e}")
                return []
            
            # Debug: Print the raw response structure
            print("Response structure:", dir(response))
            
            # Extract text from response
            response_text = None
            try:
                # Try different ways to get the response text
                if hasattr(response, 'text'):
                    response_text = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        if candidate.content.parts:
                            response_text = candidate.content.parts[0].text
                
                if not response_text:
                    print("Could not extract text from response")
                    print("Full response:", response)
                    return []
                
                print("Raw response (first 500 chars):", response_text[:500])
                
                # Clean the response to extract just the JSON
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0]
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0]
                
                # Parse the JSON
                courses = json.loads(response_text)
                
                # Ensure we have a list
                if not isinstance(courses, list):
                    courses = [courses]
                
                # Add institution to each course
                for course in courses:
                    course['institution'] = self.current_institution
                
                print(f"Successfully extracted {len(courses)} courses")
                return courses
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Problematic response: {response_text}")
                return []
                
            except Exception as e:
                print(f"Error processing response: {e}")
                print(f"Response type: {type(response)}")
                print(f"Response dir: {dir(response)}")
                if hasattr(response, 'prompt_feedback'):
                    print(f"Prompt feedback: {response.prompt_feedback}")
                return []
                
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def set_institution_from_filename(self, filename: str) -> None:
        """Set the institution name based on the filename."""
        filename_lower = filename.lower()
        if 'hilo' in filename_lower:
            self.current_institution = 'University of Hawaii at Hilo'
        elif 'west oahu' in filename_lower or 'west-oahu' in filename_lower:
            self.current_institution = 'University of Hawaii West Oʻahu'
        elif 'windward' in filename_lower:
            self.current_institution = 'Windward Community College'
        else:
            self.current_institution = 'Unknown Institution'

def process_pdfs_with_gemini(project_id: str, keyfile_path: str, catalogs_dir: str, output_dir: str, max_workers: int = 20):
    """Process all PDFs in the catalogs directory using Gemini."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the extractor with service account credentials
    extractor = GeminiCourseExtractor(project_id, keyfile_path)
    
    all_courses = []
    
    # Process each PDF in the catalogs directory
    for filename in sorted(os.listdir(catalogs_dir)):
        if filename.lower().endswith('.pdf'):
            filepath = os.path.join(catalogs_dir, filename)
            print(f"\nProcessing {filename}...")
            
            try:
                # Process the PDF and extract courses
                courses = extractor.process_pdf(filepath, max_workers=max_workers)
                if courses:
                    all_courses.extend(courses)
                    print(f"Extracted {len(courses)} courses from {filename}")
                    
                    # Save progress after each file
                    output_path = os.path.join(output_dir, 'gemini_extracted_courses.json')
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(all_courses, f, indent=2, ensure_ascii=False)
                    print(f"Progress saved to: {output_path}")
                else:
                    print(f"No courses extracted from {filename}")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\nExtracted a total of {len(all_courses)} courses")
    print(f"Final results saved to: {os.path.join(output_dir, 'gemini_extracted_courses.json')}")
    
    return all_courses

if __name__ == "__main__":
    # Configuration
    GOOGLE_CLOUD_PROJECT = "librechat-its-rci"  # Google Cloud project ID
    SERVICE_ACCOUNT_KEYFILE = "keyfile.json"  # Path to your service account key file
    
    # Define directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CATALOGS_DIR = os.path.join(BASE_DIR, "catalogs")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    
    # Ensure keyfile exists
    keyfile_path = os.path.join(BASE_DIR, SERVICE_ACCOUNT_KEYFILE)
    if not os.path.exists(keyfile_path):
        print(f"Error: Service account key file not found at {keyfile_path}")
        print("Please place your service account key file in the project directory and update the filename if needed.")
        exit(1)
    
    # Process the PDFs
    try:
        process_pdfs_with_gemini(
            project_id=GOOGLE_CLOUD_PROJECT,
            keyfile_path=keyfile_path,
            catalogs_dir=CATALOGS_DIR,
            output_dir=OUTPUT_DIR,
            max_workers=20
        )
    except Exception as e:
        print(f"An error occurred: {e}")
