import streamlit as st
import json
import random
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import time
import re
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
import torch
from langchain_huggingface import HuggingFaceEndpoint
# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
# Configure page
st.set_page_config(
    page_title="CSE Employability Test Prep",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more appealing UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .test-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white; /* Text color for test card */
    }
    .score-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        color: white; /* Text color for score card */
    }
    .question-box {
        background: #f8f9fa; /* Light grey background */
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #333333; /* Dark grey text for readability */
    }
    .answer-option {
        background: white;
        padding: 0.8rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        cursor: pointer;
        transition: all 0.3s;
        color: #333333; /* Dark grey text for readability */
    }
    .answer-option:hover {
        background: #e3f2fd;
        border-color: #1f77b4;
        color: #1f77b4; /* Make text color match border on hover */
    }
    .correct-answer {
        background: #c8e6c9 !important;
        border-color: #4caf50 !important;
    }
    .incorrect-answer {
        background: #ffcdd2 !important;
        border-color: #f44336 !important;
    }
    .timer {
        font-size: 1.5rem;
        color: #ff5722;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: #fff3e0;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
# These variables persist across reruns of the app
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = ""
if 'current_test' not in st.session_state:
    st.session_state.current_test = None
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'test_start_time' not in st.session_state:
    st.session_state.test_start_time = None
if 'progress_data' not in st.session_state:
    st.session_state.progress_data = []
if 'essay_topic' not in st.session_state:
    st.session_state.essay_topic = ""
if 'coding_problems' not in st.session_state:
    st.session_state.coding_problems = []
if 'mode' not in st.session_state:
    st.session_state.mode = "dashboard" # Can be "dashboard", "test", "practice", "practice_questions", "results", "practice_results_review"

# Test configurations
# Defines the structure and content for each test type
TEST_CONFIGS = {
    "English Usage Test": {
        "topics": ["Articles, Prepositions and Voice", "Phrases, Idioms and Sequencing",
                  "Reading Comprehension", "Sentence Correction and Speech", "Synonyms, Antonyms and Spellings"],
        "time_limit": 30, # minutes
        "question_count": 20,
        "icon": "📚"
    },
    "Analytical Reasoning Test": {
        "topics": ["Logical Reasoning", "Critical Reasoning", "Odd One Out and Analogies", "Series and Coding-Decoding"],
        "time_limit": 45,
        "question_count": 25,
        "icon": "🧠"
    },
    "Quantitative Ability Test": {
        "topics": ["Speed, Distance, Time and Work", "Profit, Loss and Interest",
                  "Ratio, Percentage and Progressions", "Number System, Algebra and Equations",
                  "Geometry, Mensuration and Trigonometry", "Statistics", "Data Interpretation"],
        "time_limit": 60,
        "question_count": 30,
        "icon": "📊"
    },
    "Written English Test": {
        "topics": ["Essay Writing (min. 120 words)"],
        "time_limit": 30,
        "question_count": 1, # Only one essay topic
        "icon": "✍️"
    },
    "Coding Test": {
        "topics": ["Programming Problems", "Algorithm Implementation"],
        "time_limit": 90,
        "question_count": 2, # Number of coding problems
        "icon": "💻"
    },
    "Domain Test (DSA)": {
        "topics": ["Data Structures", "Algorithms", "Time Complexity", "Space Complexity"],
        "time_limit": 45,
        "question_count": 25,
        "icon": "🔧"
    }
}

# --- Groq API and Question Generation Functions ---
def initialize_groq_client():  # Consider renaming to initialize_deepseek_client
    """Initializes the DeepSeek-R1 LLM client via Hugging Face Inference API."""
    
    # Get the Hugging Face API token from environment variables
    # Streamlit Cloud secrets are automatically loaded into os.environ
    hf_api_token = os.getenv("HF_TOKEN")  # Make sure this matches your secret name
    
    if not hf_api_token:
        st.error("Hugging Face API token (HF_TOKEN) not found. Please add it to your Streamlit Cloud secrets.")
        return None
    
    try:
        # Use HuggingFaceEndpoint for models hosted on Hugging Face's inference API
        llm = HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-R1",  # Use repo_id instead of endpoint_url
            huggingfacehub_api_token=hf_api_token,
            task="text-generation",
            model_kwargs={
                "temperature": 0.3,
                "max_new_tokens": 2000,
                "top_k": 50,
                "top_p": 0.95,
                "do_sample": True,
                # Add stop sequences if needed
                # "stop": ["<|endoftext|>", "<|im_end|>"]
            }
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Hugging Face Inference API client for DeepSeek-R1: {str(e)}. Please check your API token and model availability.")
        return None

def generate_questions(test_type, topic, count=5, difficulty="Medium"):
    """
    Generates multiple-choice questions using the Groq API.
    Includes robust JSON parsing and falls back to sample questions on failure.
    """
    llm = initialize_groq_client()
    if not llm:
        st.warning(f"Using sample questions for {test_type} - {topic} ({difficulty}). Groq API key is missing or invalid.")
        print(f"[Warning] {error_msg}")
        return create_sample_questions(test_type, topic, count, difficulty)

    # Define prompt templates based on test type for tailored question generation
    # Each prompt specifies the desired JSON format and content
    # The key change in the prompt is explicitly asking for a SINGLE JSON ARRAY.
    # We will still use regex to be extra safe.
    base_prompt_template = """Generate {count} multiple choice questions for '{topic}' of '{difficulty}' difficulty for a CSE employability test.
        Each question should have 4 options (A, B, C, D) and include the correct answer letter (e.g., 'A') with explanation.
        **Format your entire response as a single JSON array of objects. Do not include any text before or after the JSON.**
        Each object must have 'question', 'options' (an array of strings), 'correct_answer' (a single letter 'A','B','C','D'), and 'explanation' keys.
        Ensure options are distinct and plausible. Avoid repeating questions.
        Example format:
        [
            {{
                "question": "Which of the following is an example of an article?",
                "options": ["A) quickly", "B) and", "C) the", "D) run"],
                "correct_answer": "C",
                "explanation": "The word 'the' is a definite article."
            }},
            {{
                "question": "Another question here?",
                "options": ["A) opt1", "B) opt2", "C) opt3", "D) opt4"],
                "correct_answer": "A",
                "explanation": "Explanation for another question."
            }}
        ]
    """

    if test_type == "English Usage Test":
        # Pass variables to format method of the string
        prompt_template = base_prompt_template
    elif test_type == "Analytical Reasoning Test":
        prompt_template = base_prompt_template + \
            """
            \nInstructions:
            - Each question must test logical thinking, deductions, sequences, or patterns.
            - Strictly avoid questions involving flowcharts, visual reasoning diagrams, or complex geometric figures. Focus on text-based logical puzzles, series, coding-decoding, or critical thinking scenarios.
            - For each question, include: 'question', 'options', 'correct_answer', 'explanation'.
            - Ensure questions are unique and do not repeat previous questions within the generated set."""
    elif test_type == "Quantitative Ability Test":
        prompt_template = base_prompt_template + \
            "\nInclude numerical problems with clear mathematical solutions."
    elif test_type == "Domain Test (DSA)":
        prompt_template = base_prompt_template + \
            "\nFocus on practical DSA concepts and implementation."
    else:
        st.error(f"Question generation not implemented for {test_type}.")
        return []

    # Define the PromptTemplate with the correct input variables
    prompt = PromptTemplate(
        input_variables=["count", "topic", "difficulty"],
        template=prompt_template
    )

    try:
        chain = LLMChain(llm=llm, prompt=prompt)
        # Pass the variables as a dictionary to chain.invoke()
        # This is the correct way to pass input variables to LLMChain
        response_obj = chain.invoke({"count": count, "topic": topic, "difficulty": difficulty})
        response = response_obj['text'] # Extract the string response from the dictionary

        # --- DEBUGGING STEP: Print the raw AI response ---
        st.write("--- Debugging AI Response ---")
        st.text_area("Raw AI Response (for debugging):", value=response, height=300)
        st.write("----------------------------")
        # --- END DEBUGGING STEP ---

        # Robust JSON parsing using regex to find the array
        questions_data = []
        # Look for a JSON array pattern [ ... ]
        # re.DOTALL makes '.' match newlines, re.S is an alias for re.DOTALL
        json_array_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)

        if json_array_match:
            json_string = json_array_match.group(0)
            try:
                questions_data = json.loads(json_string)
            except json.JSONDecodeError as e:
                st.warning(f"JSON parsing error after regex extraction: {e}. AI response might still be malformed inside the array. Using sample questions.")
                return create_sample_questions(test_type, topic, count, difficulty)
        else:
            # If a single array isn't found, try to find individual JSON objects
            # This is to handle the exact scenario you provided where multiple objects are concatenated
            json_objects_matches = re.findall(r'\{\s*"question":\s*".*?"(?:,\s*".*?":\s*.*?)*?\s*\}', response, re.DOTALL)
            if json_objects_matches:
                for obj_str in json_objects_matches:
                    try:
                        questions_data.append(json.loads(obj_str))
                    except json.JSONDecodeError as e:
                        st.warning(f"Could not parse individual JSON object: {obj_str[:100]}... Error: {e}")
                        # Continue to try other objects even if one fails
            else:
                st.warning("No valid JSON structure (array or individual objects) found in AI response. Using sample questions.")
                return create_sample_questions(test_type, topic, count, difficulty)

        # Validate the structure of each question object
        valid_questions = []
        for q_data in questions_data:
            if all(key in q_data for key in ['question', 'options', 'correct_answer', 'explanation']) and \
               isinstance(q_data.get('options'), list) and len(q_data.get('options', [])) == 4 and \
               q_data.get('correct_answer') in ['A', 'B', 'C', 'D']:
                valid_questions.append(q_data)

        if valid_questions:
            # Randomly sample 'count' questions if more were generated
            if len(valid_questions) > count:
                return random.sample(valid_questions, count)
            return valid_questions
        else:
            st.warning("AI generated questions were malformed or empty after validation. Using sample questions.")
            return create_sample_questions(test_type, topic, count, difficulty)

    except Exception as e:
        st.error(f"Error generating questions from Groq: {str(e)}. Using sample questions.")
        print(f"Exception details: {e}")  # Log the exception for debugging
        return create_sample_questions(test_type, topic, count, difficulty)

def create_sample_questions(test_type, topic, count, difficulty="Medium"):
    """
    Provides fallback sample questions if AI generation fails or API key is missing.
    Ensures 'count' questions are returned, even by repeating existing samples if needed.
    """
    all_sample_q = []

    # Define a comprehensive set of sample questions for each test type and difficulty
    # This ensures a fallback when the AI cannot generate
    if test_type == "English Usage Test":
        english_samples = [
            {"question": "Choose the correct article: __ apple a day keeps the doctor away.", "options": ["A) A", "B) An", "C) The", "D) No article"], "correct_answer": "B", "explanation": "Use 'an' before words that start with a vowel sound.", "difficulty": "Easy"},
            {"question": "Identify the idiom: 'Break a leg'", "options": ["A) To injure oneself", "B) To wish good luck", "C) To stop working", "D) To run fast"], "correct_answer": "B", "explanation": "'Break a leg' is an idiom used to wish someone good luck, especially before a performance.", "difficulty": "Easy"},
            {"question": "Correct the sentence: 'She go to school.'", "options": ["A) She goes to school.", "B) She going to school.", "C) She went to school.", "D) She gone to school."], "correct_answer": "A", "explanation": "For third-person singular subjects (she, he, it) in the present tense, add '-es' or '-s' to the verb.", "difficulty": "Easy"},
            {"question": "Identify the passive voice: 'The dog chased the cat.'", "options": ["A) The dog chased the cat.", "B) The cat was chased by the dog.", "C) Chasing the cat was the dog.", "D) The cat chasing the dog."], "correct_answer": "B", "explanation": "In passive voice, the subject receives the action. 'The cat' (subject) receives the action of 'was chased'.", "difficulty": "Medium"},
            {"question": "Choose the most appropriate preposition: 'He is good ___ physics.'", "options": ["A) at", "B) in", "C) on", "D) for"], "correct_answer": "A", "explanation": "'Good at' is the correct idiom to express proficiency in a subject or skill.", "difficulty": "Medium"},
            {"question": "Which of these words is a synonym for 'Abundant'?", "options": ["A) Scarce", "B) Plentiful", "C) Rare", "D) Limited"], "correct_answer": "B", "explanation": "'Abundant' means existing or available in large quantities; 'plentiful' has a similar meaning.", "difficulty": "Medium"},
            {"question": "Complete the sentence with the correct phrasal verb: 'They decided to ___ the meeting until next week.'", "options": ["A) put off", "B) put on", "C) put up", "D) put down"], "correct_answer": "A", "explanation": "'Put off' means to postpone or delay something.", "difficulty": "Hard"},
            {"question": "Identify the error: 'Despite of the rain, they went for a walk.'", "options": ["A) 'Despite of'", "B) 'the rain'", "C) 'they went'", "D) 'for a walk'"], "correct_answer": "A", "explanation": "The correct phrase is either 'despite the rain' or 'in spite of the rain'. 'Despite of' is incorrect.", "difficulty": "Hard"},
            {"question": "Which word is an antonym for 'Ephemeral'?", "options": ["A) Fleeting", "B) Permanent", "C) Transient", "D) Momentary"], "correct_answer": "B", "explanation": "'Ephemeral' means lasting for a very short time. 'Permanent' is its direct opposite.", "difficulty": "Hard"},
        ]
        all_sample_q = [q for q in english_samples if q['difficulty'] == difficulty or difficulty == "Any"]
    elif test_type == "Analytical Reasoning Test":
        analytical_samples = [
            {"question": "Find the missing number in the series: 2, 4, 6, 8, __", "options": ["A) 9", "B) 10", "C) 12", "D) 14"], "correct_answer": "B", "explanation": "This is an arithmetic progression where each number increases by 2.", "difficulty": "Easy"},
            {"question": "Which of the following is different from the rest?", "options": ["A) Car", "B) Bus", "C) Bicycle", "D) Truck"], "correct_answer": "C", "explanation": "A bicycle is human-powered, while the others are motorized vehicles.", "difficulty": "Easy"},
            {"question": "If 'CAT' is coded as 'FDU', how is 'DOG' coded?", "options": ["A) GRJ", "B) HQK", "C) IPL", "D) GRK"], "correct_answer": "A", "explanation": "Each letter is shifted by +3 positions: C->F, A->D, T->U. Applying the same to DOG: D->G, O->R, G->J.", "difficulty": "Medium"},
            {"question": "All dogs are mammals. Some mammals are pets. Therefore, some dogs are pets. Is this statement:", "options": ["A) True", "B) False", "C) Cannot be determined", "D) Irrelevant"], "correct_answer": "C", "explanation": "This is an invalid syllogism. The premises don't guarantee that the pets that are mammals are also dogs. It cannot be determined.", "difficulty": "Medium"},
            {"question": "A, B, C, D, E are sitting in a row. C is to the immediate left of D. B is to the immediate right of E. E is between A and B. Who is in the middle?", "options": ["A) A", "B) B", "C) C", "D) E"], "correct_answer": "D", "explanation": "The arrangement is A E B C D. So, E is in the middle.", "difficulty": "Hard"},
            {"question": "If 5 people can complete a task in 10 days, how many days will 10 people take to complete the same task?", "options": ["A) 5 days", "B) 7 days", "C) 10 days", "D) 20 days"], "correct_answer": "A", "explanation": "This is an inverse proportion. (5 people * 10 days) = (10 people * X days) => 50 = 10X => X = 5 days.", "difficulty": "Hard"},
        ]
        all_sample_q = [q for q in analytical_samples if q['difficulty'] == difficulty or difficulty == "Any"]
    elif test_type == "Quantitative Ability Test":
        quantitative_samples = [
            {"question": "What is 10% of 200?", "options": ["A) 10", "B) 20", "C) 30", "D) 40"], "correct_answer": "B", "explanation": "10% of 200 is (10/100) * 200 = 0.10 * 200 = 20.", "difficulty": "Easy"},
            {"question": "If a car travels at 60 km/h for 2 hours, how far does it travel?", "options": ["A) 30 km", "B) 60 km", "C) 120 km", "D) 180 km"], "correct_answer": "C", "explanation": "Distance = Speed × Time = 60 km/h × 2 h = 120 km.", "difficulty": "Easy"},
            {"question": "A sum of money doubles itself in 5 years at simple interest. What is the rate of interest per annum?", "options": ["A) 10%", "B) 15%", "C) 20%", "D) 25%"], "correct_answer": "C", "explanation": "If a sum doubles, interest = principal. So, I = P. Using I = PRT/100, P = P * R * 5 / 100 => 1 = 5R/100 => R = 20%.", "difficulty": "Medium"},
            {"question": "If the length of a rectangle is 10 cm and its area is 50 sq cm, what is its width?", "options": ["A) 4 cm", "B) 5 cm", "C) 6 cm", "D) 7 cm"], "correct_answer": "B", "explanation": "Area = Length × Width. So, 50 = 10 × Width => Width = 5 cm.", "difficulty": "Medium"},
            {"question": "A mixture contains milk and water in the ratio 5:1. On adding 5 liters of water, the ratio of milk to water becomes 5:2. What is the quantity of milk in the original mixture?", "options": ["A) 20 liters", "B) 25 liters", "C) 30 liters", "D) 35 liters"], "correct_answer": "B", "explanation": "Let milk = 5x, water = x. After adding 5L water: 5x / (x+5) = 5/2. Solving gives x=5. So original milk = 5x = 25 liters.", "difficulty": "Hard"},
            {"question": "If 1/3 of a number is 20, what is 2/5 of that number?", "options": ["A) 12", "B) 24", "C) 36", "D) 48"], "correct_answer": "B", "explanation": "Let the number be N. (1/3)N = 20 => N = 60. Then (2/5)N = (2/5) * 60 = 2 * 12 = 24.", "difficulty": "Hard"},
        ]
        all_sample_q = [q for q in quantitative_samples if q['difficulty'] == difficulty or difficulty == "Any"]
    elif test_type == "Domain Test (DSA)":
        dsa_samples = [
            {"question": "Which data structure uses LIFO principle?", "options": ["A) Queue", "B) Stack", "C) Linked List", "D) Array"], "correct_answer": "B", "explanation": "Stack follows the Last-In, First-Out (LIFO) principle, meaning the last element added is the first one to be removed.", "difficulty": "Easy"},
            {"question": "What is the time complexity to access an element in an array by its index?", "options": ["A) O(1)", "B) O(log n)", "C) O(n)", "D) O(n log n)"], "correct_answer": "A", "explanation": "Array elements can be accessed directly using their index, which takes constant time.", "difficulty": "Easy"},
            {"question": "Which algorithm is used to find the minimum spanning tree in a graph?", "options": ["A) Dijkstra's Algorithm", "B) Bellman-Ford Algorithm", "C) Prim's or Kruskal's Algorithm", "D) Floyd-Warshall Algorithm"], "correct_answer": "C", "explanation": "Prim's and Kruskal's algorithms are common algorithms used to find a minimum spanning tree in a weighted undirected graph.", "difficulty": "Medium"},
            {"question": "What is the primary disadvantage of using a hash table for data storage?", "options": ["A) Slow insertion", "B) High memory usage", "C) Collision handling overhead", "D) Not suitable for large datasets"], "correct_answer": "C", "explanation": "Hash collisions, where different keys map to the same index, require additional logic (like chaining or open addressing), adding overhead and complexity.", "difficulty": "Medium"},
            {"question": "Which sorting algorithm has a worst-case time complexity of O(n log n) and is a comparison sort?", "options": ["A) Quick Sort", "B) Merge Sort", "C) Heap Sort", "D) Both B and C"], "correct_answer": "D", "explanation": "Both Merge Sort and Heap Sort guarantee O(n log n) worst-case time complexity, whereas Quick Sort's worst-case is O(n^2).", "difficulty": "Hard"},
            {"question": "Which data structure is suitable for implementing a symbol table where operations like search, insert, and delete are frequently performed?", "options": ["A) Array", "B) Linked List", "C) Hash Table or Balanced Binary Search Tree", "D) Queue"], "correct_answer": "C", "explanation": "Hash tables offer average O(1) time for these operations. Balanced BSTs (like AVL trees or Red-Black trees) offer O(log n) worst-case time, both making them suitable.", "difficulty": "Hard"},
        ]
        all_sample_q = [q for q in dsa_samples if q['difficulty'] == difficulty or difficulty == "Any"]
    else:
        # Generic fallback samples if no specific samples for a test type
        all_sample_q = [
            {"question": "What is the capital of France?", "options": ["A) Berlin", "B) Paris", "C) Rome", "D) Madrid"], "correct_answer": "B", "explanation": "Paris is the capital and most populous city of France.", "type": "general", "difficulty": "Easy"},
            {"question": "What is the largest planet in our solar system?", "options": ["A) Earth", "B) Mars", "C) Jupiter", "D) Saturn"], "correct_answer": "C", "explanation": "Jupiter is the largest planet in our solar system by volume and mass.", "type": "general", "difficulty": "Easy"},
            {"question": "Which year did the Titanic sink?", "options": ["A) 1910", "B) 1912", "C) 1914", "D) 1916"], "correct_answer": "B", "explanation": "The RMS Titanic sank on April 15, 1912, after striking an iceberg.", "type": "general", "difficulty": "Medium"},
        ]
        all_sample_q = [q for q in all_sample_q if q.get('difficulty') == difficulty or difficulty == "Any"]
        if not all_sample_q:
            return [] # No samples found at all

    # Ensure the requested 'count' of questions is returned, even by repeating
    if not all_sample_q:
        return [] # Still no samples after filtering

    if len(all_sample_q) < count:
        # If fewer unique samples than requested, repeat and shuffle
        repeated_samples = (all_sample_q * ((count // len(all_sample_q)) + 1))[:count]
        random.shuffle(repeated_samples)
        return repeated_samples
    else:
        # Otherwise, pick a random sample of the desired count
        return random.sample(all_sample_q, count)


def generate_essay_topic():
    """Generates an essay topic using the Groq API or falls back to a sample."""
    llm = initialize_groq_client()
    if not llm:
        return random.choice([
            "The Impact of Artificial Intelligence on Future Software Development",
            "Cybersecurity Challenges in the Digital Age",
            "The Role of Cloud Computing in Modern Business",
            "Ethical Considerations in Software Engineering",
            "The Future of Remote Work in the Tech Industry"
        ])

    prompt = PromptTemplate(
        input_variables=[],
        template="""Generate a concise and thought-provoking essay topic for CSE students' employability test.
        The topic should be related to technology, engineering, or professional development.
        Return only the topic title, without any introductory or concluding remarks. Ensure the topic is unique and not generic.
        """
    )

    try:
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run()
        return response.strip().replace('"', '')
    except Exception as e:
        st.error(f"Error generating essay topic: {str(e)}. Using a sample topic.")
        return random.choice([
            "The Impact of Artificial Intelligence on Future Software Development",
            "Cybersecurity Challenges in the Digital Age",
            "The Role of Cloud Computing in Modern Business",
            "Ethical Considerations in Software Engineering",
            "The Future of Remote Work in the Tech Industry"
        ])
    

def generate_coding_problems():
    """
    Generates coding problems using the Groq API.
    Includes robust JSON parsing and falls back to sample problems on failure.
    """
    llm = initialize_groq_client()
    if not llm:
        st.warning("Using sample coding problems. Groq API key is missing or invalid.")
        return generate_coding_problems_fallback()

    # The key change in the prompt is explicitly asking for a SINGLE JSON ARRAY.
    # We will still use regex to be extra safe.
    prompt = PromptTemplate(
    input_variables=[], # No explicit input variables needed here as template is static
    template="""Generate 2 distinct coding problems suitable for a **CSE employability test at a hiring company**.
    Each problem must have: 'title', 'description', 'difficulty' (**Medium/Hard**), and 'example' (showing input and expected output).
    Focus on commonly assessed areas like **data structures (arrays, linked lists, trees, graphs, hash maps), algorithms (sorting, searching, dynamic programming, greedy algorithms), time complexity analysis, and edge case handling**.
    Ensure the two problems are completely different from each other and **require more than a trivial solution**.
    The problems should simulate typical interview questions, emphasizing **optimal solutions and analytical thinking**.
    **Format your entire response as a single JSON array of objects. Do not include any text before or after the JSON.**
    Each object must have these exact keys.
    Example format:
    [
        {{
            "title": "Merge K Sorted Lists",
            "description": "You are given an array of k linked-lists, each sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.",
            "difficulty": "Hard",
            "example": "Input: lists = [[1,4,5],[1,3,4],[2,6]]\\nOutput: [1,1,2,3,4,4,5,6]"
        }},
        {{
            "title": "Longest Palindromic Substring",
            "description": "Given a string s, return the longest palindromic substring in s. A substring is a contiguous non-empty sequence of characters within a string.",
            "difficulty": "Medium",
            "example": "Input: s = 'babad'\\nOutput: 'bab' (or 'aba')"
        }}
    ]
    """
)

    try:
        chain = LLMChain(llm=llm, prompt=prompt)
        response_obj = chain.invoke({}) # Pass an empty dictionary as no variables are in the prompt
        response = response_obj['text'] # Extract the string response

        # --- DEBUGGING STEP: Print the raw AI response ---
        st.write("--- Debugging AI Coding Response ---")
        st.text_area("Raw AI Coding Response (for debugging):", value=response, height=300)
        st.write("----------------------------")
        # --- END DEBUGGING STEP ---


        # Robust JSON parsing for coding problems using regex
        problems_data = []
        json_array_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)

        if json_array_match:
            json_string = json_array_match.group(0)
            try:
                problems_data = json.loads(json_string)
            except json.JSONDecodeError as e:
                st.warning(f"JSON parsing error for coding problems after regex extraction: {e}. AI response might still be malformed inside the array. Using sample problems.")
                return generate_coding_problems_fallback()
        else:
            # If a single array isn't found, try to find individual JSON objects
            json_objects_matches = re.findall(r'\{\s*"title":\s*".*?"(?:,\s*".*?":\s*.*?)*?\s*\}', response, re.DOTALL)
            if json_objects_matches:
                for obj_str in json_objects_matches:
                    try:
                        problems_data.append(json.loads(obj_str))
                    except json.JSONDecodeError as e:
                        st.warning(f"Could not parse individual coding problem JSON object: {obj_str[:100]}... Error: {e}")
            else:
                st.warning("No valid JSON structure (array or individual objects) found in AI response for coding problems. Using sample problems.")
                return generate_coding_problems_fallback()

        # Basic validation for coding problems
        valid_problems = []
        for p_data in problems_data:
            if all(key in p_data for key in ['title', 'description', 'difficulty', 'example']):
                valid_problems.append(p_data)

        if len(valid_problems) >= 2: # Ensure at least 2 problems are returned, take first two
            return valid_problems[:2]
        else:
            st.warning("AI generated coding problems were malformed or less than 2. Using sample problems.")
            return generate_coding_problems_fallback()

    except Exception as e:
        st.error(f"Error generating coding problems from Groq: {str(e)}. Using sample problems.")
        return generate_coding_problems_fallback()

def generate_coding_problems_fallback():
    """Provides fallback sample coding problems."""
    sample_problems = [
        {
            "title": "Two Sum",
            "description": "Given an array of integers `nums` and an integer `target`, return indices of the two numbers such as they add up to `target`.",
            "difficulty": "Easy",
            "example": "Input: nums = [2,7,11,15], target = 9\nOutput: [0,1] (Because nums[0] + nums[1] = 2 + 7 = 9)"
        },
        {
            "title": "Palindrome Check",
            "description": "Write a function to check if a given string is a palindrome. A palindrome reads the same forwards and backwards, ignoring case and non-alphanumeric characters.",
            "difficulty": "Medium",
            "example": "Input: 'Racecar'\nOutput: True\n\nInput: 'hello'\nOutput: False"
        },
        {
            "title": "Fibonacci Sequence",
            "description": "Write a function that generates the first `n` numbers in the Fibonacci sequence. The sequence starts with 0 and 1, and each subsequent number is the sum of the two preceding ones.",
            "difficulty": "Easy",
            "example": "Input: n = 5\nOutput: [0, 1, 1, 2, 3]"
        },
        {
            "title": "Factorial Calculation",
            "description": "Write a recursive function to calculate the factorial of a non-negative integer `n`. The factorial of a number `n` is the product of all integers from 1 to `n`.",
            "difficulty": "Easy",
            "example": "Input: n = 4\nOutput: 24 (Because 4 * 3 * 2 * 1 = 24)"
        }
    ]
    # Ensure exactly 2 problems are returned
    if len(sample_problems) >= 2:
        return random.sample(sample_problems, 2)
    else:
        # If somehow fewer than 2 samples are available, repeat to get 2
        return (sample_problems * 2)[:2]

# --- Streamlit UI Functions ---

def main():
    """Main function to run the Streamlit application."""
    st.markdown('<h1 class="main-header">🎓 CSE Employability Test Preparation</h1>', unsafe_allow_html=True)

    # API Key Input/Check
    if not st.session_state.groq_api_key:
        st.info("Please enter your **Groq API key** to generate AI-powered questions and explanations. You can get one from [Groq Console](https://console.groq.com/keys).")
        
        # Prioritize environment variable
        env_api_key = os.getenv("GROQ_API_KEY")
        if env_api_key:
            st.session_state.groq_api_key = env_api_key
            st.success("API key loaded from environment variable!")
            st.rerun() # Rerun to apply the API key and proceed
        else:
            api_key_input = st.text_input("Enter Groq API Key:", type="password", key="groq_api_key_input_widget")
            if api_key_input:
                st.session_state.groq_api_key = api_key_input
                st.success("API key saved from input field!")
                st.rerun() # Rerun to apply the API key and proceed
            else:
                st.warning("You can use the app with sample questions, but AI-generated content requires an API key. Please add it to a `.env` file as `GROQ_API_KEY='your_key'` or enter it above.")

    # Sidebar Navigation
    st.sidebar.title("📋 Test Menu")

    if st.sidebar.button("🏠 Dashboard", key="dashboard_btn"):
        reset_session_state_for_dashboard()
        st.session_state.mode = "dashboard"
        st.rerun()

    if st.sidebar.button("💡 Practice Mode", key="practice_mode_btn"):
        reset_session_state_for_dashboard()
        st.session_state.mode = "practice"
        st.rerun()

    # Render content based on the current mode
    if st.session_state.mode == "dashboard":
        show_dashboard()
    elif st.session_state.mode == "test":
        show_test_interface()
    elif st.session_state.mode == "practice":
        show_practice_mode()
    elif st.session_state.mode == "practice_questions":
        show_mcq_interface(is_practice_mode=True)
    elif st.session_state.mode == "results":
        show_results()
    elif st.session_state.mode == "practice_results_review":
        show_detailed_mcq_review(is_practice_mode=True)

def reset_session_state_for_dashboard():
    """Resets all relevant session state variables to default for a fresh start."""
    st.session_state.current_test = None
    st.session_state.questions = []
    st.session_state.current_question = 0
    st.session_state.score = 0
    st.session_state.answers = []
    st.session_state.test_start_time = None
    st.session_state.essay_topic = ""
    st.session_state.coding_problems = []

# --- UI Display Functions ---

def show_dashboard():
    """Displays the main dashboard with test options and progress overview."""

    # Progress Overview (only if data exists)
    if st.session_state.progress_data:
        st.markdown("---")
        st.markdown("## 📈 Your Progress")
        col1, col2, col3 = st.columns(3)

        with col1:
            total_tests = len(st.session_state.progress_data)
            st.markdown(f'<div class="score-card"><h3>{total_tests}</h3><p>Tests Completed</p></div>', unsafe_allow_html=True)

        with col2:
            avg_score = sum(d['score'] for d in st.session_state.progress_data) / len(st.session_state.progress_data)
            st.markdown(f'<div class="score-card"><h3>{avg_score:.1f}%</h3><p>Average Score</p></div>', unsafe_allow_html=True)

        with col3:
            best_score = max(d['score'] for d in st.session_state.progress_data)
            st.markdown(f'<div class="score-card"><h3>{best_score:.1f}%</h3><p>Best Score</p></div>', unsafe_allow_html=True)

        # Progress Chart
        df = pd.DataFrame(st.session_state.progress_data)
        fig = px.line(df, x='date', y='score', color='test_type',
                     title='Score Progress Over Time',
                     labels={'score': 'Score (%)', 'date': 'Date'},
                     hover_data=['test_type', 'score'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Complete tests to see your progress here!")

    # Test Selection
    st.markdown("---")
    st.markdown("## 📚 Select a Test")

    # Global Difficulty Selector for Dashboard Tests
    difficulty_levels = ["Easy", "Medium", "Hard"]
    selected_difficulty_dashboard = st.selectbox("Select **Overall Test Difficulty**:", difficulty_levels, key="dashboard_difficulty_select")

    cols = st.columns(2)
    for i, (test_name, config) in enumerate(TEST_CONFIGS.items()):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"""
                <div class="test-card">
                    <h3>{config['icon']} {test_name}</h3>
                    <p><strong>Topics:</strong> {len(config['topics'])} areas</p>
                    <p><strong>Time:</strong> {config['time_limit']} minutes</p>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"Start {test_name}", key=f"start_{test_name}"):
                    # Reset state for a new test run
                    st.session_state.questions = []
                    st.session_state.answers = []
                    st.session_state.current_question = 0
                    st.session_state.score = 0
                    st.session_state.essay_topic = ""
                    st.session_state.coding_problems = []

                    st.session_state.current_test = test_name
                    st.session_state.test_start_time = datetime.now()
                    st.session_state.mode = "test"

                    with st.spinner(f"Generating {test_name} content... This may take a moment."):
                        if test_name == "Written English Test":
                            st.session_state.essay_topic = generate_essay_topic()
                        elif test_name == "Coding Test":
                            st.session_state.coding_problems = generate_coding_problems()
                        else:
                            all_questions = []
                            # Distribute questions among topics
                            questions_per_topic = config['question_count'] // len(config['topics'])
                            remaining_questions = config['question_count'] % len(config['topics'])

                            for topic in config['topics']:
                                q_count = questions_per_topic
                                if remaining_questions > 0:
                                    q_count += 1
                                    remaining_questions -= 1
                                questions = generate_questions(test_name, topic, q_count, selected_difficulty_dashboard)
                                all_questions.extend(questions)

                            # Shuffle and select to ensure randomness and target count
                            random.shuffle(all_questions)
                            st.session_state.questions = all_questions[:config['question_count']]
                            if not st.session_state.questions:
                                st.error(f"Failed to generate questions for {test_name}. Please check your API key or try again.")
                                st.session_state.mode = "dashboard" # Go back to dashboard on failure
                                return

                    st.rerun() # Rerun to start the test interface

def show_test_interface():
    """Displays the general test interface with timer, delegating to specific test types."""
    test_name = st.session_state.current_test
    config = TEST_CONFIGS[test_name]

    st.markdown(f"---")
    st.markdown(f"## {config['icon']} {test_name}")

    if st.session_state.test_start_time:
        elapsed = datetime.now() - st.session_state.test_start_time
        remaining_seconds = int(timedelta(minutes=config['time_limit']).total_seconds() - elapsed.total_seconds())

        if remaining_seconds <= 0:
            st.markdown('<div class="timer">⏰ Time\'s up! Test completed.</div>', unsafe_allow_html=True)
            st.error("Time's up! Your test has been automatically submitted.")
            st.session_state.mode = "results" # Move to results automatically
            st.rerun()
            return

        minutes, seconds = divmod(remaining_seconds, 60)
        st.markdown(f'<div class="timer">⏱️ Time Remaining: {minutes:02d}:{seconds:02d}</div>', unsafe_allow_html=True)

        # Delegate to specific test content based on type
        if test_name == "Written English Test":
            show_essay_interface()
        elif test_name == "Coding Test":
            show_coding_interface()
        else:
            show_mcq_interface() # Default for other test types
    else:
        st.warning("Test timer not started. Please go back to the dashboard and start a test.")
        if st.button("Back to Dashboard", key="test_interface_back_to_dashboard"):
            st.session_state.mode = "dashboard"
            st.rerun()

def show_mcq_interface(is_practice_mode=False):
    """
    Displays the multiple choice question interface.
    Handles navigation, answer submission, and progress display.
    `is_practice_mode` determines immediate feedback and result review type.
    """
    if not st.session_state.questions:
        st.error("No questions available. Please go back to the dashboard or select a topic in practice mode.")
        if is_practice_mode:
            if st.button("Back to Practice Selection", key="back_to_practice_from_empty_q"):
                st.session_state.mode = "practice"
                reset_session_state_for_dashboard()
                st.rerun()
        else:
            if st.button("Back to Dashboard", key="back_to_dashboard_from_empty_q"):
                st.session_state.mode = "dashboard"
                reset_session_state_for_dashboard()
                st.rerun()
        return

    current_q_index = st.session_state.current_question
    total_questions = len(st.session_state.questions)

    # Check if all questions are answered/skipped
    if current_q_index >= total_questions:
        if is_practice_mode:
            st.session_state.mode = "practice_results_review" # Detailed review for practice
            st.rerun()
        else:
            st.session_state.mode = "results" # Summary results for full test
            st.rerun()
        return

    question_data = st.session_state.questions[current_q_index]

    progress = (current_q_index + 1) / total_questions
    st.progress(progress, text=f"Question {current_q_index + 1} of {total_questions}")

    st.markdown(f'<div class="question-box"><h4>{question_data["question"]}</h4></div>', unsafe_allow_html=True)

    # Pre-select user's previous answer if available (for navigation)
    selected_option_value = None
    if len(st.session_state.answers) > current_q_index:
        prev_answer = st.session_state.answers[current_q_index]
        if prev_answer and prev_answer.get("user_answer_text") not in ["Skipped", "Not Answered"]:
            selected_option_value = prev_answer.get("user_answer_text")

    initial_index = None
    if selected_option_value in question_data["options"]:
        initial_index = question_data["options"].index(selected_option_value)

    selected_option = st.radio(
        "Choose your answer:",
        question_data["options"],
        key=f"mcq_q_{current_q_index}_{st.session_state.current_test}_radio", # Unique key for each radio button
        index=initial_index # Set initial selection
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Submit Answer", key=f"submit_mcq_{current_q_index}_{st.session_state.current_test}"):
            answer_letter = selected_option[0] if selected_option else None
            user_marked_option_text = selected_option if selected_option else "Not Answered"

            if answer_letter is None:
                st.warning("Please select an answer before submitting.")
            else:
                correct = (answer_letter == question_data["correct_answer"])

                answer_entry = {
                    "question": question_data["question"],
                    "options": question_data["options"],
                    "user_answer_letter": answer_letter,
                    "user_answer_text": user_marked_option_text,
                    "correct_answer_letter": question_data["correct_answer"],
                    "correct_answer_text": next(opt for opt in question_data["options"] if opt.startswith(question_data["correct_answer"] + ")")),
                    "is_correct": correct,
                    "explanation": question_data["explanation"]
                }

                # Update or append the answer
                if len(st.session_state.answers) <= current_q_index:
                    st.session_state.answers.append(answer_entry)
                else:
                    st.session_state.answers[current_q_index] = answer_entry

                # Provide immediate feedback in practice mode
                if is_practice_mode:
                    # Recalculate score for immediate display
                    st.session_state.score = sum(1 for ans in st.session_state.answers if ans.get("is_correct"))
                    if correct:
                        st.success("✅ Correct!")
                    else:
                        st.error(f"❌ Incorrect.")
                
                time.sleep(0.5) # Short delay for feedback visibility
                st.session_state.current_question += 1
                st.rerun()

    with col2:
        if st.button("Skip Question", key=f"skip_mcq_{current_q_index}_{st.session_state.current_test}"):
            answer_entry = {
                "question": question_data["question"],
                "options": question_data["options"],
                "user_answer_letter": "Skipped",
                "user_answer_text": "Skipped",
                "correct_answer_letter": question_data["correct_answer"],
                "correct_answer_text": next(opt for opt in question_data["options"] if opt.startswith(question_data["correct_answer"] + ")")),
                "is_correct": False,
                "explanation": question_data["explanation"]
            }
            if len(st.session_state.answers) <= current_q_index:
                st.session_state.answers.append(answer_entry)
            else:
                st.session_state.answers[current_q_index] = answer_entry
            
            st.session_state.current_question += 1
            st.rerun()

def show_essay_interface():
    """Displays the essay writing interface."""
    if not st.session_state.essay_topic:
        st.error("No essay topic generated. Please go back to the dashboard and try again.")
        if st.button("Back to Dashboard", key="back_to_dashboard_from_empty_essay"):
            st.session_state.mode = "dashboard"
            st.rerun()
        return

    st.markdown(f"### Essay Topic: {st.session_state.essay_topic}")
    st.markdown("**Instructions:** Write a well-structured essay of at least 120 words on the given topic. Focus on clarity, coherence, and correct grammar.")

    # Pre-fill if essay was already written (e.g., during rerun)
    current_essay_text = ""
    # Check if the answers list is not empty and the first element contains an essay_text (for written tests)
    if st.session_state.answers and len(st.session_state.answers) > 0 and \
       st.session_state.answers[0].get("essay_text") is not None:
        current_essay_text = st.session_state.answers[0]["essay_text"]

    essay_text = st.text_area("Your Essay:", height=300, max_chars=2000, key="essay_input", value=current_essay_text)
    word_count = len(essay_text.split()) if essay_text.strip() else 0 # Robust word count

    st.markdown(f"**Word Count:** {word_count} / 120 (minimum)")

    if st.button("Submit Essay", key="submit_essay_btn"):
        if word_count >= 120:
            st.success("✅ Essay submitted successfully! Review your score and feedback below.")

            # Simple placeholder scoring for essay
            score = min(100, (word_count / 120) * 80 + 20)
            st.session_state.score = score

            # Store or update the essay answer in session state
            essay_answer_data = {
                "essay_topic": st.session_state.essay_topic,
                "essay_text": essay_text,
                "word_count": word_count,
                "score_evaluated": score # This is the "score" for the essay
            }
            if not st.session_state.answers:
                st.session_state.answers.append(essay_answer_data)
            else:
                st.session_state.answers[0] = essay_answer_data # Assuming only one essay for this test

            st.session_state.test_start_time = None # End the timer
            st.session_state.mode = "results"
            st.rerun()
        else:
            st.error("❌ Essay must be at least 120 words long to submit.")

def show_coding_interface():
    """Displays the coding test interface."""
    if not st.session_state.coding_problems:
        st.error("No coding problems available. Please go back to the dashboard.")
        if st.button("Back to Dashboard", key="back_to_dashboard_from_empty_code"):
            st.session_state.mode = "dashboard"
            st.rerun()
        return

    st.markdown("### Coding Problems")
    st.markdown("**Instructions:** Solve the following programming problems. Write clean, efficient code. You can use any programming language you prefer, but focus on the logic.")

    user_solutions_status = [] # To track if all problems have some input

    # Initialize or retrieve user's solutions for persistence across reruns
    # Ensures the answers list is set up correctly for coding problems
    if not st.session_state.answers or \
       not (len(st.session_state.answers) > 0 and st.session_state.answers[0].get("type") == "coding_test"):
        st.session_state.answers = [
            {"type": "coding_test", "problems_solved": []}
        ]
        # Populate problems_solved based on actual coding_problems
        for p in st.session_state.coding_problems:
            st.session_state.answers[0]["problems_solved"].append({"problem_title": p['title'], "user_code": ""})
    
    # Ensure problems_solved list aligns with the number of coding problems
    # This handles cases where problems might be re-generated or changed
    if len(st.session_state.answers[0]["problems_solved"]) != len(st.session_state.coding_problems):
        new_problems_solved = []
        for p in st.session_state.coding_problems:
            found_existing = False
            for existing_sol in st.session_state.answers[0]["problems_solved"]:
                if existing_sol["problem_title"] == p['title']:
                    new_problems_solved.append(existing_sol)
                    found_existing = True
                    break
            if not found_existing:
                new_problems_solved.append({"problem_title": p['title'], "user_code": ""})
        st.session_state.answers[0]["problems_solved"] = new_problems_solved


    for i, problem in enumerate(st.session_state.coding_problems):
        st.markdown(f"---")
        st.markdown(f"#### Problem {i+1}: {problem['title']} ({problem['difficulty']})")
        st.markdown(f"**Description:** {problem['description']}")
        st.code(problem['example'], language="text")

        st.markdown(f"**Your Solution (Problem {i+1}):**")
        
        # Get the current user's code for this specific problem from session state
        current_solution_code = st.session_state.answers[0]["problems_solved"][i]["user_code"]

        code = st.text_area(f"Write your code for '{problem['title']}' here:", height=200, key=f"code_{i}_{st.session_state.current_test}", value=current_solution_code)

        # Update the session state immediately as the user types
        st.session_state.answers[0]["problems_solved"][i]["user_code"] = code

        user_solutions_status.append(bool(code.strip())) # Track if code was entered

    st.markdown("---")

    if st.button("Submit All Solutions", key="submit_all_code_btn"):
        if not all(user_solutions_status):
            st.warning("Please provide solutions for all problems before submitting.")
        else:
            st.success("✅ All coding solutions submitted! Review your score and feedback below.")
            st.info("💡 In a real test, your code would be run against hidden test cases for evaluation. For this simulation, a placeholder score is provided.")

            # Placeholder score - in a real application, this would be determined by a backend judge
            st.session_state.score = 85 # Example score

            st.session_state.test_start_time = None # End the timer
            st.session_state.mode = "results"
            st.rerun()

def show_detailed_mcq_review(is_practice_mode=False):
    """
    Displays a detailed review for MCQ tests/practice sessions, showing questions,
    user's answer, correct answer, and explanation.
    """
    if is_practice_mode:
        st.markdown("---")
        st.markdown("## 📝 Detailed Practice Review")
    else:
        st.markdown("---")
        st.markdown("### 📝 Detailed Review")

    if st.session_state.answers:
        for i, answer_data in enumerate(st.session_state.answers):
            # Ensure it's an MCQ answer data structure
            if "question" in answer_data and "options" in answer_data:
                st.markdown(f"**Q{i+1}:** {answer_data['question']}")

                user_answer_display = answer_data['user_answer_text']
                if answer_data.get("is_correct", False):
                    st.success(f"✅ Your Answer: {user_answer_display} (Correct)")
                elif answer_data.get("user_answer_letter") == "Skipped":
                    st.warning(f"⏭️ You Skipped this question.")
                else:
                    st.error(f"❌ Your Answer: {user_answer_display} (Incorrect)")

                st.markdown(f"**Correct Answer:** {answer_data['correct_answer_text']}")
                st.info(f"**Explanation:** {answer_data['explanation']}")
                st.markdown("---")
    else:
        st.info("No questions were attempted in this session.")

    # Action buttons for practice review
    if is_practice_mode:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Start New Practice", key="start_new_practice_from_results"):
                st.session_state.mode = "practice"
                reset_session_state_for_dashboard()
                st.rerun()
        with col2:
            if st.button("🏠 Back to Dashboard", key="back_to_dashboard_from_practice_results"):
                reset_session_state_for_dashboard()
                st.session_state.mode = "dashboard"
                st.rerun()

def show_results():
    """Displays the final results for a completed test."""
    test_name = st.session_state.current_test
    config = TEST_CONFIGS[test_name]

    st.markdown(f"---")
    st.markdown(f"## 🎯 {test_name} Results")

    if test_name == "Written English Test":
        # Retrieve essay score
        if st.session_state.answers and st.session_state.answers[0].get("score_evaluated") is not None:
            score = st.session_state.answers[0]["score_evaluated"]
        else:
            score = 0
        st.markdown(f'<div class="score-card"><h2>Score: {score:.1f}%</h2></div>', unsafe_allow_html=True)

        if score >= 80:
            st.success("🌟 **Excellent!** Your essay demonstrates strong writing skills and meets the length requirement.")
        elif score >= 60:
            st.info("👍 **Good work!** Your essay is well-structured. Keep practicing to improve clarity and depth.")
        else:
            st.warning("📚 **Keep practicing!** Focus on meeting the word count, improving structure, and enhancing your arguments.")

        st.markdown("### Your Essay Submission:")
        if st.session_state.answers and st.session_state.answers[0].get("essay_text"):
            essay_data = st.session_state.answers[0]
            st.write(f"**Topic:** {essay_data['essay_topic']}")
            st.write(f"**Word Count:** {essay_data['word_count']}")
            st.text_area("Submitted Essay:", value=essay_data['essay_text'], height=250, disabled=True)
        else:
            st.info("No essay was submitted.")

    elif test_name == "Coding Test":
        score = st.session_state.score # Score is set by show_coding_interface after submission
        st.markdown(f'<div class="score-card"><h2>Score: {score:.1f}%</h2></div>', unsafe_allow_html=True)

        if score >= 80:
            st.success("🌟 **Excellent coding skills!** Your solutions seem well-thought-out.")
        elif score >= 60:
            st.info("👍 **Good programming fundamentals.** Keep practicing to refine your solutions and handle edge cases.")
        else:
            st.warning("📚 **Keep coding!** Practice more problems to improve your problem-solving and implementation skills.")

        st.markdown("### Your Submitted Solutions:")
        # Check if answers contain coding test data before iterating
        if st.session_state.answers and st.session_state.answers[0].get("type") == "coding_test" and \
           st.session_state.answers[0].get("problems_solved"):
            for i, solved_problem in enumerate(st.session_state.answers[0]["problems_solved"]):
                st.markdown(f"**Problem {i+1}:** {solved_problem['problem_title']}")
                st.code(solved_problem['user_code'], language="python") # Display as Python code
        else:
            st.info("No solutions were submitted or recorded.")

    else:
        # MCQ test results
        total_questions = len(st.session_state.questions)
        correct_answers = sum(1 for ans in st.session_state.answers if ans.get("is_correct"))
        percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        st.session_state.score = percentage # Update session score for progress tracking

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f'<div class="score-card"><h3>{correct_answers}/{total_questions}</h3><p>Correct Answers</p></div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="score-card"><h3>{percentage:.1f}%</h3><p>Score</p></div>', unsafe_allow_html=True)

        with col3:
            if percentage >= 80:
                grade = "A"
                color = "#4caf50"
            elif percentage >= 60:
                grade = "B"
                color = "#ff9800"
            else:
                grade = "C"
                color = "#f44336"

            st.markdown(f'<div class="score-card" style="background-color: {color}"><h3>Grade {grade}</h3><p>Performance</p></div>', unsafe_allow_html=True)

        # Show detailed MCQ review
        show_detailed_mcq_review(is_practice_mode=False)

    # Save progress data for the dashboard charts
    if st.session_state.mode == "results":
        st.session_state.progress_data.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "test_type": test_name,
            "score": st.session_state.score
        })

    # Action buttons after test results
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Retake Test", key="retake_test_btn"):
            # Prepare for retake by resetting relevant state and regenerating content
            current_test_name_for_retake = st.session_state.current_test
            reset_session_state_for_dashboard()
            st.session_state.current_test = current_test_name_for_retake
            st.session_state.test_start_time = datetime.now()
            st.session_state.mode = "test"

            with st.spinner(f"Preparing {current_test_name_for_retake} for retake..."):
                if current_test_name_for_retake == "Written English Test":
                    st.session_state.essay_topic = generate_essay_topic()
                elif current_test_name_for_retake == "Coding Test":
                    st.session_state.coding_problems = generate_coding_problems()
                else:
                    config = TEST_CONFIGS[current_test_name_for_retake]
                    all_questions = []
                    questions_per_topic = config['question_count'] // len(config['topics'])
                    remaining_questions = config['question_count'] % len(config['topics'])

                    # Use 'Medium' difficulty for retake by default, could be stored from original run
                    default_retake_difficulty = "Medium"
                    for topic in config['topics']:
                        q_count = questions_per_topic
                        if remaining_questions > 0:
                            q_count += 1
                            remaining_questions -= 1
                        questions = generate_questions(current_test_name_for_retake, topic, q_count, default_retake_difficulty)
                        all_questions.extend(questions)
                    random.shuffle(all_questions)
                    st.session_state.questions = all_questions[:config['question_count']]
                    if not st.session_state.questions:
                        st.error(f"Failed to generate questions for {current_test_name_for_retake}. Please check your API key or try again.")
                        st.session_state.mode = "dashboard" # Fallback to dashboard
                        st.rerun()

            st.rerun()

    with col2:
        if st.button("🏠 Back to Dashboard", key="back_to_dashboard_btn"):
            reset_session_state_for_dashboard()
            st.session_state.mode = "dashboard"
            st.rerun()

def show_practice_mode():
    """Displays the practice mode selection interface."""
    st.markdown("---")
    st.markdown("## 🎯 Practice Mode")
    st.markdown("Practice specific topics without time pressure and get instant feedback.")

    # Exclude essay and coding tests from practice mode as they are distinct
    practice_test_types = {k: v for k, v in TEST_CONFIGS.items() if k not in ["Written English Test", "Coding Test"]}

    if st.session_state.mode == "practice":
        selected_test_type = st.selectbox("Select Test Type:", list(practice_test_types.keys()), key="practice_test_type_select")

        if selected_test_type:
            config = TEST_CONFIGS[selected_test_type]

            st.markdown(f"**Available Topics for {selected_test_type}:**")
            st.write(", ".join(config['topics']))

            selected_topic = st.selectbox("Select Topic to Practice:", config['topics'], key="practice_topic_select")

            difficulty_levels = ["Easy", "Medium", "Hard"]
            selected_difficulty_practice = st.selectbox("Select Difficulty Level:", difficulty_levels, key="practice_difficulty_select")

            num_questions = st.slider("Number of Questions:", 1, 10, 5, key="practice_num_questions_slider")

            if st.button("Start Practice Session", key="start_practice_session_btn"):
                # Reset practice-specific state
                st.session_state.questions = []
                st.session_state.answers = []
                st.session_state.current_question = 0
                st.session_state.score = 0
                st.session_state.current_test = selected_test_type # Store the test type for context

                with st.spinner(f"Generating practice questions for {selected_topic} ({selected_difficulty_practice})..."):
                    st.session_state.questions = generate_questions(selected_test_type, selected_topic, num_questions, selected_difficulty_practice)
                    if not st.session_state.questions:
                        st.error("Could not generate practice questions. Please try a different topic or check your API key.")
                        st.session_state.mode = "practice"
                        return

                st.session_state.mode = "practice_questions"
                st.rerun()
    elif st.session_state.mode == "practice_questions":
        show_mcq_interface(is_practice_mode=True)
    elif st.session_state.mode == "practice_results_review":
        show_detailed_mcq_review(is_practice_mode=True)


# Run the main application
if __name__ == "__main__":
    main()
