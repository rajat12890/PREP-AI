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

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="CSE Employability Test Prep",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

# Initialize session state
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
if 'mode' not in st.session_state: # Added for practice mode
    st.session_state.mode = "dashboard" # Can be "dashboard", "test", "practice", "practice_questions", "results"

# Test configurations
TEST_CONFIGS = {
    "English Usage Test": {
        "topics": ["Articles, Prepositions and Voice", "Phrases, Idioms and Sequencing",
                  "Reading Comprehension", "Sentence Correction and Speech", "Synonyms, Antonyms and Spellings"],
        "time_limit": 30,
        "question_count": 20,
        "icon": "📚"
    },
    "Analytical Reasoning Test": {
        "topics": ["Logical Reasoning", "Critical Reasoning", "Flowcharts and Visual Reasoning",
                  "Odd One Out and Analogies", "Series and Coding-Decoding"],
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
        "question_count": 1,
        "icon": "✍️"
    },
    "Coding Test": {
        "topics": ["Programming Problems", "Algorithm Implementation"],
        "time_limit": 90,
        "question_count": 2,
        "icon": "💻"
    },
    "Domain Test (DSA)": {
        "topics": ["Data Structures", "Algorithms", "Time Complexity", "Space Complexity"],
        "time_limit": 45,
        "question_count": 25,
        "icon": "🔧"
    }
}

def initialize_groq_client():
    """Initialize Groq client with API key"""
    if st.session_state.groq_api_key:
        try:
            llm = ChatGroq(
                groq_api_key=st.session_state.groq_api_key,
                model_name="llama3-70b-8192", # Using a powerful model for better quality
                temperature=0.7,
                max_tokens=2000
            )
            return llm
        except Exception as e:
            st.error(f"Error initializing Groq client: {str(e)}. Please check your API key and internet connection.")
            return None
    return None

def generate_questions(test_type, topic, count=5, difficulty="Medium"): # Added difficulty parameter
    """Generate practice questions using Groq"""
    llm = initialize_groq_client()
    if not llm:
        st.warning(f"Using sample questions for {test_type} - {topic} ({difficulty}). Groq API key is missing or invalid.")
        return create_sample_questions(test_type, topic, count, difficulty) # Pass difficulty to fallback

    # Update prompt templates to include difficulty
    if test_type == "English Usage Test":
        prompt_template = """Generate {count} multiple choice questions for '{topic}' of '{difficulty}' difficulty for a CSE employability test.
            Each question should have 4 options (A, B, C, D) and include the correct answer letter (e.g., 'A') with explanation.
            Format as a JSON array of objects. Each object must have 'question', 'options' (an array of strings), 'correct_answer' (a single letter 'A','B','C','D'), and 'explanation' keys.
            Make questions practical and relevant to technical communication. Ensure options are distinct and plausible.
            Example format:
            [
                {{
                    "question": "Which of the following is an example of an article?",
                    "options": ["A) quickly", "B) and", "C) the", "D) run"],
                    "correct_answer": "C",
                    "explanation": "The word 'the' is a definite article."
                }}
            ]
            """
    elif test_type == "Analytical Reasoning Test":
        prompt_template = """Generate {count} analytical reasoning questions for '{topic}' of '{difficulty}' difficulty for a CSE employability test.
            Each question should have 4 options (A, B, C, D) and include the correct answer letter (e.g., 'A') with detailed explanation.
            Format as a JSON array of objects. Each object must have 'question', 'options' (an array of strings), 'correct_answer' (a single letter 'A','B','C','D'), and 'explanation' keys.
            Focus on logical thinking and problem-solving skills.
            Example format:
            [
                {{
                    "question": "If all A are B, and all B are C, then all A are what?",
                    "options": ["A) B", "B) C", "C) D", "D) A"],
                    "correct_answer": "B",
                    "explanation": "This is a basic syllogism. Since all A are B and all B are C, it logically follows that all A are C."
                }}
            ]
            """
    elif test_type == "Quantitative Ability Test":
        prompt_template = """Generate {count} quantitative ability questions for '{topic}' of '{difficulty}' difficulty for a CSE employability test.
            Each question should have 4 options (A, B, C, D) and include the correct answer letter (e.g., 'A') with step-by-step solution/explanation.
            Format as a JSON array of objects. Each object must have 'question', 'options' (an array of strings), 'correct_answer' (a single letter 'A','B','C','D'), and 'explanation' keys.
            Include numerical problems with clear mathematical solutions.
            Example format:
            [
                {{
                    "question": "What is 20% of 150?",
                    "options": ["A) 20", "B) 30", "C) 40", "D) 50"],
                    "correct_answer": "B",
                    "explanation": "To find 20% of 150, calculate (20/100) * 150 = 0.20 * 150 = 30."
                }}
            ]
            """
    elif test_type == "Domain Test (DSA)":
        prompt_template = """Generate {count} Data Structures and Algorithms questions for '{topic}' of '{difficulty}' difficulty for a CSE employability test.
            Each question should have 4 options (A, B, C, D) and include the correct answer letter (e.g., 'A') with detailed explanation.
            Format as a JSON array of objects. Each object must have 'question', 'options' (an array of strings), 'correct_answer' (a single letter 'A','B','C','D'), and 'explanation' keys.
            Focus on practical DSA concepts and implementation.
            Example format:
            [
                {{
                    "question": "Which data structure uses LIFO principle?",
                    "options": ["A) Queue", "B) Stack", "C) Linked List", "D) Array"],
                    "correct_answer": "B",
                    "explanation": "Stack follows the Last-In, First-Out (LIFO) principle, meaning the last element added is the first one to be removed."
                }}
            ]
            """
    else:
        st.error(f"Question generation not implemented for {test_type}.")
        return []

    prompt = PromptTemplate(
        input_variables=["topic", "count", "difficulty"], # Added difficulty
        template=prompt_template
    )

    try:
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(topic=topic, count=count, difficulty=difficulty)
        print(f"DEBUG: Raw AI Response for {test_type} - {topic} ({difficulty}):\n{response[:500]}...")

        # Attempt to parse JSON
        try:
            # Find the first '[' and last ']' to ensure we only parse the JSON array
            json_start = response.find('[')
            json_end = response.rfind(']')
            if json_start != -1 and json_end != -1:
                json_string = response[json_start : json_end + 1]
                questions_data = json.loads(json_string)
                # Validate the structure of each question
                valid_questions = []
                for q_data in questions_data:
                    if all(key in q_data for key in ['question', 'options', 'correct_answer', 'explanation']) and \
                       isinstance(q_data['options'], list) and len(q_data['options']) == 4 and \
                       q_data['correct_answer'] in ['A', 'B', 'C', 'D']:
                        valid_questions.append(q_data)
                if valid_questions:
                    return valid_questions
                else:
                    st.warning("AI generated questions were malformed or empty after validation. Using sample questions.")
                    return create_sample_questions(test_type, topic, count, difficulty)
            else:
                st.warning("AI response did not contain a valid JSON array. Using sample questions.")
                return create_sample_questions(test_type, topic, count, difficulty)
        except json.JSONDecodeError as e:
            st.warning(f"JSON parsing error: {e}. AI response might be malformed. Using sample questions.")
            print(f"DEBUG: JSONDecodeError: {e}, Response causing error: {response}")
            return create_sample_questions(test_type, topic, count, difficulty)
        except Exception as e:
            st.warning(f"An unexpected error occurred during JSON processing: {e}. Using sample questions.")
            print(f"DEBUG: Unexpected error during JSON processing: {e}, Response: {response}")
            return create_sample_questions(test_type, topic, count, difficulty)

    except Exception as e:
        st.error(f"Error generating questions from Groq: {str(e)}. Using sample questions.")
        print(f"DEBUG: Groq API call failed: {e}")
        return create_sample_questions(test_type, topic, count, difficulty)

def create_sample_questions(test_type, topic, count, difficulty="Medium"): # Added difficulty parameter
    """Create sample questions as fallback if AI generation fails or API key is missing.
    Ensures that 'count' questions are returned, even if by repeating existing samples."""
    all_sample_q = []

    if test_type == "English Usage Test":
        if difficulty == "Easy":
            all_sample_q = [
                {"question": "Choose the correct article: __ apple a day keeps the doctor away.", "options": ["A) A", "B) An", "C) The", "D) No article"], "correct_answer": "B", "explanation": "Use 'an' before words that start with a vowel sound."},
                {"question": "Identify the idiom: 'Break a leg'", "options": ["A) To injure oneself", "B) To wish good luck", "C) To stop working", "D) To run fast"], "correct_answer": "B", "explanation": "'Break a leg' is an idiom used to wish someone good luck, especially before a performance."},
                {"question": "Correct the sentence: 'She go to school.'", "options": ["A) She goes to school.", "B) She going to school.", "C) She went to school.", "D) She gone to school."], "correct_answer": "A", "explanation": "For third-person singular subjects (she, he, it) in the present tense, add '-es' or '-s' to the verb."},
            ]
        elif difficulty == "Medium":
            all_sample_q = [
                {"question": "Identify the passive voice: 'The dog chased the cat.'", "options": ["A) The dog chased the cat.", "B) The cat was chased by the dog.", "C) Chasing the cat was the dog.", "D) The cat chasing the dog."], "correct_answer": "B", "explanation": "In passive voice, the subject receives the action. 'The cat' (subject) receives the action of 'was chased'."},
                {"question": "Choose the most appropriate preposition: 'He is good ___ physics.'", "options": ["A) at", "B) in", "C) on", "D) for"], "correct_answer": "A", "explanation": "'Good at' is the correct idiom to express proficiency in a subject or skill."},
                {"question": "Which of these words is a synonym for 'Abundant'?", "options": ["A) Scarce", "B) Plentiful", "C) Rare", "D) Limited"], "correct_answer": "B", "explanation": "'Abundant' means existing or available in large quantities; 'plentiful' has a similar meaning."},
            ]
        else: # Hard or any other difficulty
            all_sample_q = [
                {"question": "Complete the sentence with the correct phrasal verb: 'They decided to ___ the meeting until next week.'", "options": ["A) put off", "B) put on", "C) put up", "D) put down"], "correct_answer": "A", "explanation": "'Put off' means to postpone or delay something."},
                {"question": "Identify the error: 'Despite of the rain, they went for a walk.'", "options": ["A) 'Despite of'", "B) 'the rain'", "C) 'they went'", "D) 'for a walk'"], "correct_answer": "A", "explanation": "The correct phrase is either 'despite the rain' or 'in spite of the rain'. 'Despite of' is incorrect."},
                {"question": "Which word is an antonym for 'Ephemeral'?", "options": ["A) Fleeting", "B) Permanent", "C) Transient", "D) Momentary"], "correct_answer": "B", "explanation": "'Ephemeral' means lasting for a very short time. 'Permanent' is its direct opposite."},
            ]
    elif test_type == "Analytical Reasoning Test":
        if difficulty == "Easy":
            all_sample_q = [
                {"question": "Find the missing number in the series: 2, 4, 6, 8, __", "options": ["A) 9", "B) 10", "C) 12", "D) 14"], "correct_answer": "B", "explanation": "This is an arithmetic progression where each number increases by 2."},
                {"question": "Which of the following is different from the rest?", "options": ["A) Car", "B) Bus", "C) Bicycle", "D) Truck"], "correct_answer": "C", "explanation": "A bicycle is human-powered, while the others are motorized vehicles."},
            ]
        elif difficulty == "Medium":
            all_sample_q = [
                {"question": "If 'SPIN' is coded as '5049', how is 'PINS' coded?", "options": ["A) 4950", "B) 9405", "C) 0495", "D) 0594"], "correct_answer": "A", "explanation": "The digits are directly mapped to the letters: S=5, P=0, I=4, N=9. So, PINS becomes 0495."},
                {"question": "All dogs are mammals. Some mammals are pets. Therefore, some dogs are pets. Is this statement:", "options": ["A) True", "B) False", "C) Cannot be determined", "D) Irrelevant"], "correct_answer": "C", "explanation": "This is an invalid syllogism. The premises don't guarantee that the pets that are mammals are also dogs. It cannot be determined."},
            ]
        else: # Hard or any other difficulty
            all_sample_q = [
                {"question": "A, B, C, D, E are sitting in a row. C is to the immediate left of D. B is to the immediate right of E. E is between A and B. Who is in the middle?", "options": ["A) A", "B) B", "C) C", "D) E"], "correct_answer": "D", "explanation": "The arrangement is A E B C D. So, E is in the middle."},
                {"question": "If 'CLOUD' is coded as 'FNQWG', how is 'SIGHT' coded?", "options": ["A) TJHIU", "B) VKHKW", "C) WLIHV", "D) VLJHW"], "correct_answer": "B", "explanation": "Each letter is shifted by +3 positions: C->F, L->N, O->Q, U->W, D->G. Applying the same to SIGHT: S->V, I->L, G->J, H->K, T->W."},
            ]
    elif test_type == "Quantitative Ability Test":
        if difficulty == "Easy":
            all_sample_q = [
                {"question": "What is 10% of 200?", "options": ["A) 10", "B) 20", "C) 30", "D) 40"], "correct_answer": "B", "explanation": "10% of 200 is (10/100) * 200 = 0.10 * 200 = 20."},
                {"question": "If a car travels at 60 km/h for 2 hours, how far does it travel?", "options": ["A) 30 km", "B) 60 km", "C) 120 km", "D) 180 km"], "correct_answer": "C", "explanation": "Distance = Speed × Time = 60 km/h × 2 h = 120 km."},
            ]
        elif difficulty == "Medium":
            all_sample_q = [
                {"question": "A sum of money doubles itself in 5 years at simple interest. What is the rate of interest per annum?", "options": ["A) 10%", "B) 15%", "C) 20%", "D) 25%"], "correct_answer": "C", "explanation": "If a sum doubles, interest = principal. So, I = P. Using I = PRT/100, P = P * R * 5 / 100 => 1 = 5R/100 => R = 20%."},
                {"question": "If the length of a rectangle is 10 cm and its area is 50 sq cm, what is its width?", "options": ["A) 4 cm", "B) 5 cm", "C) 6 cm", "D) 7 cm"], "correct_answer": "B", "explanation": "Area = Length × Width. So, 50 = 10 × Width => Width = 5 cm."},
            ]
        else: # Hard or any other difficulty
            all_sample_q = [
                {"question": "A mixture contains milk and water in the ratio 5:1. On adding 5 liters of water, the ratio of milk to water becomes 5:2. What is the quantity of milk in the original mixture?", "options": ["A) 20 liters", "B) 25 liters", "C) 30 liters", "D) 35 liters"], "correct_answer": "B", "explanation": "Let milk = 5x, water = x. After adding 5L water: 5x / (x+5) = 5/2. Solving gives x=5. So original milk = 5x = 25 liters."},
                {"question": "If 1/3 of a number is 20, what is 2/5 of that number?", "options": ["A) 12", "B) 24", "C) 36", "D) 48"], "correct_answer": "B", "explanation": "Let the number be N. (1/3)N = 20 => N = 60. Then (2/5)N = (2/5) * 60 = 2 * 12 = 24."},
            ]
    elif test_type == "Domain Test (DSA)":
        if difficulty == "Easy":
            all_sample_q = [
                {"question": "Which data structure uses LIFO principle?", "options": ["A) Queue", "B) Stack", "C) Linked List", "D) Array"], "correct_answer": "B", "explanation": "Stack follows the Last-In, First-Out (LIFO) principle, meaning the last element added is the first one to be removed."},
                {"question": "What is the time complexity to access an element in an array by its index?", "options": ["A) O(1)", "B) O(log n)", "C) O(n)", "D) O(n log n)"], "correct_answer": "A", "explanation": "Array elements can be accessed directly using their index, which takes constant time."},
            ]
        elif difficulty == "Medium":
            all_sample_q = [
                {"question": "Which algorithm is used to find the minimum spanning tree in a graph?", "options": ["A) Dijkstra's Algorithm", "B) Bellman-Ford Algorithm", "C) Prim's or Kruskal's Algorithm", "D) Floyd-Warshall Algorithm"], "correct_answer": "C", "explanation": "Prim's and Kruskal's algorithms are common algorithms used to find a minimum spanning tree in a weighted undirected graph."},
                {"question": "What is the primary disadvantage of using a hash table for data storage?", "options": ["A) Slow insertion", "B) High memory usage", "C) Collision handling overhead", "D) Not suitable for large datasets"], "correct_answer": "C", "explanation": "Hash collisions, where different keys map to the same index, require additional logic (like chaining or open addressing), adding overhead and complexity."},
            ]
        else: # Hard or any other difficulty
            all_sample_q = [
                {"question": "Which sorting algorithm has a worst-case time complexity of O(n log n) and is a comparison sort?", "options": ["A) Quick Sort", "B) Merge Sort", "C) Heap Sort", "D) Both B and C"], "correct_answer": "D", "explanation": "Both Merge Sort and Heap Sort guarantee O(n log n) worst-case time complexity, whereas Quick Sort's worst-case is O(n^2)."},
                {"question": "Which data structure is suitable for implementing a symbol table where operations like search, insert, and delete are frequently performed?", "options": ["A) Array", "B) Linked List", "C) Hash Table or Balanced Binary Search Tree", "D) Queue"], "correct_answer": "C", "explanation": "Hash tables offer average O(1) time for these operations. Balanced BSTs (like AVL trees or Red-Black trees) offer O(log n) worst-case time, both making them suitable."},
            ]
    
    # Ensure we return at least 'count' questions by repeating if necessary
    if not all_sample_q: # Should not happen with current setup, but as a safeguard
        print(f"WARNING: No sample questions defined for {test_type} with difficulty {difficulty}. Returning empty list.")
        return []

    if len(all_sample_q) < count:
        # Repeat samples to reach the desired count
        repeated_samples = (all_sample_q * ((count // len(all_sample_q)) + 1))[:count]
        random.shuffle(repeated_samples) # Shuffle the repeated list
        return repeated_samples
    else:
        return random.sample(all_sample_q, count)


def generate_essay_topic():
    """Generate essay topic using Groq"""
    llm = initialize_groq_client()
    if not llm:
        print("DEBUG: Groq client not initialized for essay topic generation. Using sample.")
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
        Return only the topic title, without any introductory or concluding remarks.
        """
    )

    try:
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run()
        return response.strip().replace('"', '') # Remove potential quotes around the topic
    except Exception as e:
        st.error(f"Error generating essay topic: {str(e)}. Using a sample topic.")
        print(f"DEBUG: Groq API call failed for essay topic: {e}")
        return random.choice([
            "The Impact of Artificial Intelligence on Future Software Development",
            "Cybersecurity Challenges in the Digital Age",
            "The Role of Cloud Computing in Modern Business",
            "Ethical Considerations in Software Engineering",
            "The Future of Remote Work in the Tech Industry"
        ])

def generate_coding_problems():
    """Generate coding problems using Groq"""
    llm = initialize_groq_client()
    if not llm:
        st.warning("Using sample coding problems. Groq API key is missing or invalid.")
        print("DEBUG: Groq client not initialized for coding problem generation. Using sample.")
        return generate_coding_problems_fallback()

    prompt = PromptTemplate(
        input_variables=[],
        template="""Generate 2 coding problems for a CSE employability test.
        Each problem must have: 'title', 'description', 'difficulty' (Easy/Medium/Hard), and 'example' (showing input and expected output).
        Focus on fundamental programming concepts like arrays, strings, loops, and basic algorithms.
        Format as a JSON array of objects. Each object must have these exact keys.
        Example format:
        [
            {{
                "title": "Reverse a String",
                "description": "Write a function that takes a string as input and returns the string reversed.",
                "difficulty": "Easy",
                "example": "Input: 'hello'\nOutput: 'olleh'"
            }},
            {{
                "title": "Find Largest Element in Array",
                "description": "Write a function that finds and returns the largest element in an array of integers.",
                "difficulty": "Easy",
                "example": "Input: [3, 1, 4, 1, 5, 9, 2, 6]\nOutput: 9"
            }}
        ]
        """
    )

    try:
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run()
        print(f"DEBUG: Raw AI Response for Coding Problems:\n{response[:500]}...")

        # Attempt to parse JSON
        try:
            json_start = response.find('[')
            json_end = response.rfind(']')
            if json_start != -1 and json_end != -1:
                json_string = response[json_start : json_end + 1]
                problems = json.loads(json_string)
                # Basic validation for coding problems
                valid_problems = []
                for p_data in problems:
                    if all(key in p_data for key in ['title', 'description', 'difficulty', 'example']):
                        valid_problems.append(p_data)
                if valid_problems:
                    return valid_problems
                else:
                    st.warning("AI generated coding problems were malformed or empty after validation. Using sample problems.")
                    return generate_coding_problems_fallback() # Use a distinct fallback
            else:
                st.warning("AI response did not contain a valid JSON array for coding problems. Using sample problems.")
                return generate_coding_problems_fallback()
        except json.JSONDecodeError as e:
            st.warning(f"JSON parsing error for coding problems: {e}. AI response might be malformed. Using sample problems.")
            print(f"DEBUG: JSONDecodeError for coding problems: {e}, Response causing error: {response}")
            return generate_coding_problems_fallback()
        except Exception as e:
            st.warning(f"An unexpected error occurred during coding problem JSON processing: {e}. Using sample problems.")
            print(f"DEBUG: Unexpected error during coding problem JSON processing: {e}, Response: {response}")
            return generate_coding_problems_fallback()

    except Exception as e:
        st.error(f"Error generating coding problems from Groq: {str(e)}. Using sample problems.")
        print(f"DEBUG: Groq API call failed for coding problems: {e}")
        return generate_coding_problems_fallback()

def generate_coding_problems_fallback():
    """Fallback for coding problems if AI generation fails. Always returns 2 problems."""
    sample_problems = [
        {
            "title": "Two Sum",
            "description": "Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.",
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
    # Ensure exactly 2 problems are returned, picking randomly if more are available, or repeating if fewer.
    if len(sample_problems) >= 2:
        return random.sample(sample_problems, 2)
    else:
        # If somehow fewer than 2 samples, repeat them
        return (sample_problems * 2)[:2]


def main():
    st.markdown('<h1 class="main-header">🎓 CSE Employability Test Preparation</h1>', unsafe_allow_html=True)

    # API Key Input
    if not st.session_state.groq_api_key:
        st.info("Please enter your **Groq API key** to generate AI-powered questions and explanations. You can get one from [Groq Console](https://console.groq.com/keys).")
        api_key_input = st.text_input("Enter Groq API Key:", type="password", key="groq_api_key_input_widget")
        
        # Prioritize env variable, then widget input
        if os.getenv("GROQ_API_KEY"):
            st.session_state.groq_api_key = os.getenv("GROQ_API_KEY")
            st.success("API key loaded from environment variable!")
            st.rerun() # Rerun to apply API key
        elif api_key_input:
            st.session_state.groq_api_key = api_key_input
            st.success("API key saved from input field!")
            st.rerun() # Rerun to apply API key
        else:
            st.warning("You can use the app with sample questions, but AI-generated content requires an API key. Please add it to a `.env` file as `GROQ_API_KEY='your_key'` or enter it above.")

    # Sidebar
    st.sidebar.title("📋 Test Menu")

    if st.sidebar.button("🏠 Dashboard", key="dashboard_btn"):
        reset_session_state_for_dashboard()
        st.session_state.mode = "dashboard"
        st.rerun()

    if st.sidebar.button("💡 Practice Mode", key="practice_mode_btn"):
        reset_session_state_for_dashboard() # Reset everything to start fresh for practice
        st.session_state.mode = "practice"
        st.rerun()

    # Main content rendering based on mode
    if st.session_state.mode == "dashboard":
        show_dashboard()
    elif st.session_state.mode == "test":
        show_test_interface()
    elif st.session_state.mode == "practice":
        show_practice_mode()
    elif st.session_state.mode == "practice_questions": # When actively doing practice questions
        # This will be handled by show_practice_mode, which calls show_mcq_interface
        # But we need to prevent dashboard from showing too
        show_practice_mode()
    elif st.session_state.mode == "results":
        show_results() # Display results directly if redirected from a test completion

def reset_session_state_for_dashboard():
    """Resets all relevant session state variables to default for dashboard view."""
    st.session_state.current_test = None
    st.session_state.questions = []
    st.session_state.current_question = 0
    st.session_state.score = 0
    st.session_state.answers = []
    st.session_state.test_start_time = None
    st.session_state.essay_topic = ""
    st.session_state.coding_problems = []

def show_dashboard():
    """Show the main dashboard"""

    # Progress Overview
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

    # NEW: Global Difficulty Selector for Dashboard Tests
    difficulty_levels = ["Easy", "Medium", "Hard"]
    selected_difficulty_dashboard = st.selectbox("Select **Overall Test Difficulty**:", difficulty_levels, key="dashboard_difficulty_select")

    cols = st.columns(2)
    for i, (test_name, config) in enumerate(TEST_CONFIGS.items()):
        with cols[i % 2]:
            with st.container(border=True): # Using border for better visual separation
                st.markdown(f"""
                <div class="test-card">
                    <h3>{config['icon']} {test_name}</h3>
                    <p><strong>Topics:</strong> {len(config['topics'])} areas</p>
                    <p><strong>Time:</strong> {config['time_limit']} minutes</p>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"Start {test_name}", key=f"start_{test_name}"):
                    # Reset questions, answers, etc., for a new test
                    st.session_state.questions = []
                    st.session_state.answers = []
                    st.session_state.current_question = 0
                    st.session_state.score = 0
                    st.session_state.essay_topic = ""
                    st.session_state.coding_problems = []

                    st.session_state.current_test = test_name
                    st.session_state.test_start_time = datetime.now()
                    st.session_state.mode = "test" # Set mode to test

                    with st.spinner(f"Generating {test_name} questions... This may take a moment."):
                        if test_name == "Written English Test":
                            # Essay topic generation doesn't currently take difficulty, but could be extended
                            st.session_state.essay_topic = generate_essay_topic()
                        elif test_name == "Coding Test":
                            # Coding problems could also be generated with difficulty in mind
                            st.session_state.coding_problems = generate_coding_problems()
                        else:
                            # Generate questions for other tests using the selected dashboard difficulty
                            all_questions = []
                            questions_per_topic = config['question_count'] // len(config['topics'])
                            remaining_questions = config['question_count'] % len(config['topics'])

                            for topic in config['topics']:
                                q_count = questions_per_topic
                                if remaining_questions > 0:
                                    q_count += 1
                                    remaining_questions -= 1
                                # Pass selected_difficulty_dashboard here
                                questions = generate_questions(test_name, topic, q_count, selected_difficulty_dashboard)
                                all_questions.extend(questions)

                            random.shuffle(all_questions) # Shuffle for variety
                            st.session_state.questions = all_questions[:config['question_count']]
                            if not st.session_state.questions:
                                st.error(f"Failed to generate questions for {test_name}. Please check your API key or try again.")
                                st.session_state.mode = "dashboard" # Go back if no questions
                                return

                    st.rerun() # Only rerun once questions are generated/assigned

def show_test_interface():
    """Show the test interface with a real-time updating timer."""
    test_name = st.session_state.current_test
    config = TEST_CONFIGS[test_name]

    st.markdown(f"---")
    st.markdown(f"## {config['icon']} {test_name}")

    # Timer display and logic
    if st.session_state.test_start_time:
        timer_placeholder = st.empty() # Placeholder for the timer display

        # IMPORTANT: This block will cause continuous reruns.
        # It's inside a 'while True' loop to ensure constant updates.
        # This means other interactions might be slightly delayed until the sleep finishes.
        # For a timer, this is often the desired behavior.
        while st.session_state.mode == "test": # Only run timer if in active test mode
            elapsed = datetime.now() - st.session_state.test_start_time
            remaining_seconds = int(timedelta(minutes=config['time_limit']).total_seconds() - elapsed.total_seconds())
            
            if remaining_seconds <= 0:
                timer_placeholder.markdown('<div class="timer">⏰ Time\'s up! Test completed.</div>', unsafe_allow_html=True)
                st.error("Time's up! Your test has been automatically submitted.")
                st.session_state.mode = "results"
                st.rerun() # Rerun to go to results
                return # Exit function after rerunning
            
            minutes, seconds = divmod(remaining_seconds, 60)
            timer_placeholder.markdown(f'<div class="timer">⏱️ Time Remaining: {minutes:02d}:{seconds:02d}</div>', unsafe_allow_html=True)
            
            # This sleep and rerun is crucial for real-time timer updates
            time.sleep(1)
            st.rerun() # Force rerun to update the timer
        
        # If the mode changed (e.g., to 'results' or 'dashboard' from within an MCQ/Essay/Coding submit button)
        # then the while loop would have exited, and the rest of the script (main()) will handle the new mode.
    
    # Test-specific interfaces (these will only run once after a rerun,
    # or if the timer is not active (e.g., in practice mode where timer isn't enforced this way))
    if st.session_state.mode == "test": # Only display test content if still in test mode
        if test_name == "Written English Test":
            show_essay_interface()
        elif test_name == "Coding Test":
            show_coding_interface()
        else: # For MCQ-based tests (English, Analytical, Quantitative, Domain DSA)
            show_mcq_interface()


def show_mcq_interface(is_practice_mode=False):
    """Show multiple choice question interface"""
    if not st.session_state.questions:
        st.error("No questions available. Please go back to the dashboard or select a topic in practice mode.")
        if is_practice_mode:
            if st.button("Back to Practice Selection", key="back_to_practice_from_empty_q"):
                st.session_state.mode = "practice"
                st.rerun()
        else:
            if st.button("Back to Dashboard", key="back_to_dashboard_from_empty_q"):
                st.session_state.mode = "dashboard"
                st.rerun()
        return

    current_q_index = st.session_state.current_question
    total_questions = len(st.session_state.questions)

    if current_q_index >= total_questions:
        if is_practice_mode:
            st.success("You've completed all questions in this practice session!")
            if st.button("End Practice Session", key="end_practice_session"):
                st.session_state.mode = "practice" # Go back to practice selection
                reset_session_state_for_dashboard() # Clear questions/answers for next practice
                st.rerun()
            return
        else:
            st.session_state.mode = "results" # Change mode to results for final test
            st.rerun()
            return

    question_data = st.session_state.questions[current_q_index]

    # Progress bar
    progress = (current_q_index + 1) / total_questions
    st.progress(progress)
    st.markdown(f"**Question {current_q_index + 1} of {total_questions}**")

    # Question
    st.markdown(f'<div class="question-box"><h4>{question_data["question"]}</h4></div>', unsafe_allow_html=True)

    # Options
    # Generate unique keys for radio buttons for each question
    selected_option_value = None
    if f"mcq_q_{current_q_index}_{st.session_state.current_test}_radio" in st.session_state:
        selected_option_value = st.session_state[f"mcq_q_{current_q_index}_{st.session_state.current_test}_radio"]

    selected_option = st.radio(
        "Choose your answer:",
        question_data["options"],
        key=f"mcq_q_{current_q_index}_{st.session_state.current_test}_radio",
        index=question_data["options"].index(selected_option_value) if selected_option_value in question_data["options"] else None
    )
    
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Submit Answer", key=f"submit_mcq_{current_q_index}_{st.session_state.current_test}"):
            # Extract answer letter (A, B, C, D)
            answer_letter = selected_option[0] if selected_option else None # Get the first char (A, B, C, D)
            
            # Find the full text of the user's selected option to store for results review
            user_marked_option_text = selected_option if selected_option else "Not Answered"
            
            if answer_letter is None:
                st.warning("Please select an answer before submitting.")
            else:
                correct = (answer_letter == question_data["correct_answer"])
                
                if correct:
                    st.session_state.score += 1
                    st.success("✅ Correct!")
                else:
                    st.error(f"❌ Incorrect. The correct answer is {question_data['correct_answer']}")
                
                st.info(f"**Explanation:** {question_data['explanation']}")
                
                # Store the answer for review, including the full text of the user's marked option
                st.session_state.answers.append({
                    "question": question_data["question"],
                    "user_answer_letter": answer_letter, # Storing the letter
                    "user_answer_text": user_marked_option_text, # Storing the full text
                    "correct_answer_letter": question_data["correct_answer"],
                    "correct_answer_text": next(opt for opt in question_data["options"] if opt.startswith(question_data["correct_answer"] + ")")), # Get full text of correct answer
                    "is_correct": correct,
                    "explanation": question_data["explanation"]
                })
                
                time.sleep(1)  # Show feedback briefly before moving on
                st.session_state.current_question += 1
                st.rerun()

    with col2:
        if st.button("Skip Question", key=f"skip_mcq_{current_q_index}_{st.session_state.current_test}"):
            # Store skipped answer
            st.session_state.answers.append({
                "question": question_data["question"],
                "user_answer_letter": "Skipped",
                "user_answer_text": "Skipped",
                "correct_answer_letter": question_data["correct_answer"],
                "correct_answer_text": next(opt for opt in question_data["options"] if opt.startswith(question_data["correct_answer"] + ")")), # Get full text of correct answer
                "is_correct": False,
                "explanation": question_data["explanation"]
            })
            st.session_state.current_question += 1
            st.rerun()

def show_essay_interface():
    """Show essay writing interface"""
    if not st.session_state.essay_topic:
        st.error("No essay topic generated. Please go back to the dashboard and try again.")
        if st.button("Back to Dashboard", key="back_to_dashboard_from_empty_essay"):
            st.session_state.mode = "dashboard"
            st.rerun()
        return

    st.markdown(f"### Essay Topic: {st.session_state.essay_topic}")
    st.markdown("**Instructions:** Write a well-structured essay of at least 120 words on the given topic. Focus on clarity, coherence, and correct grammar.")

    essay_text = st.text_area("Your Essay:", height=300, max_chars=2000, key="essay_input")
    word_count = len(essay_text.split()) if essay_text else 0

    st.markdown(f"**Word Count:** {word_count} / 120 (minimum)")

    if st.button("Submit Essay", key="submit_essay_btn"):
        if word_count >= 120:
            st.success("✅ Essay submitted successfully! Review your score and feedback below.")

            # Simple evaluation (can be enhanced with LLM grading)
            score = min(100, (word_count / 120) * 80 + 20) # Base score + word count bonus
            st.session_state.score = score

            st.session_state.answers.append({
                "essay_topic": st.session_state.essay_topic,
                "essay_text": essay_text,
                "word_count": word_count,
                "score_evaluated": score # Use a different key to avoid confusion with main score
            })
            # This is important: set test_start_time to 0 or None to stop the timer
            st.session_state.test_start_time = None
            st.session_state.mode = "results" # Change mode to results
            st.rerun()
        else:
            st.error("❌ Essay must be at least 120 words long to submit.")

def show_coding_interface():
    """Show coding test interface"""
    if not st.session_state.coding_problems:
        st.error("No coding problems available. Please go back to the dashboard.")
        if st.button("Back to Dashboard", key="back_to_dashboard_from_empty_code"):
            st.session_state.mode = "dashboard"
            st.rerun()
        return

    st.markdown("### Coding Problems")
    st.markdown("**Instructions:** Solve the following programming problems. Write clean, efficient code. You can use any programming language you prefer, but focus on the logic.")

    user_solutions = []
    all_solutions_entered = True

    for i, problem in enumerate(st.session_state.coding_problems):
        st.markdown(f"---")
        st.markdown(f"#### Problem {i+1}: {problem['title']} ({problem['difficulty']})")
        st.markdown(f"**Description:** {problem['description']}")
        st.code(problem['example'], language="text") # Use st.code for examples too

        # Use markdown for instructions
        st.markdown(f"**Your Solution (Problem {i+1}):**")
        # Ensure a robust default value for text_area to prevent errors if answers is not yet populated
        current_solution = ""
        # Check if previous solutions exist in session state. This is for persistent display.
        # Note: For coding and essay, st.session_state.answers will typically have just one entry
        # corresponding to the entire submission, not per question.
        if st.session_state.answers and len(st.session_state.answers) > 0 and \
           st.session_state.answers[0].get("type") == "coding_test" and \
           st.session_state.answers[0].get("problems_solved") and \
           i < len(st.session_state.answers[0]["problems_solved"]):
            current_solution = st.session_state.answers[0]["problems_solved"][i]["user_code"]
        
        code = st.text_area(f"Write your code for '{problem['title']}' here:", height=200, key=f"code_{i}_{st.session_state.current_test}", value=current_solution)

        user_solutions.append({"problem_title": problem['title'], "user_code": code})
        if not code.strip():
            all_solutions_entered = False

    st.markdown("---")

    if st.button("Submit All Solutions", key="submit_all_code_btn"):
        if not all_solutions_entered:
            st.warning("Please provide solutions for all problems before submitting.")
        else:
            st.success("✅ All coding solutions submitted! Review your score and feedback below.")
            st.info("💡 In a real test, your code would be run against hidden test cases for evaluation. For this simulation, a placeholder score is provided.")

            # Placeholder score for coding test
            st.session_state.score = 85 # You would replace this with actual evaluation logic (e.g., using an LLM to rate code quality, or running test cases in a backend)
            st.session_state.answers = [{"type": "coding_test", "problems_solved": user_solutions}]
            st.session_state.test_start_time = None # Stop the timer
            st.session_state.mode = "results" # Change mode to results
            st.rerun()


def show_results():
    """Show test results"""
    test_name = st.session_state.current_test
    config = TEST_CONFIGS[test_name]

    st.markdown(f"---")
    st.markdown(f"## 🎯 {test_name} Results")

    if test_name == "Written English Test":
        # Extract the score from the answers list if it's there
        if st.session_state.answers and st.session_state.answers[0].get("score_evaluated") is not None:
            score = st.session_state.answers[0]["score_evaluated"]
        else:
            score = 0 # Default if somehow not set
        st.markdown(f'<div class="score-card"><h2>Score: {score:.1f}%</h2></div>', unsafe_allow_html=True)

        if score >= 80:
            st.success("🌟 **Excellent!** Your essay demonstrates strong writing skills and meets the length requirement.")
        elif score >= 60:
            st.info("👍 **Good work!** Your essay is well-structured. Keep practicing to improve clarity and depth.")
        else:
            st.warning("📚 **Keep practicing!** Focus on meeting the word count, improving structure, and enhancing your arguments.")

        st.markdown("### Your Essay Submission:")
        if st.session_state.answers:
            essay_data = st.session_state.answers[0]
            st.write(f"**Topic:** {essay_data['essay_topic']}")
            st.write(f"**Word Count:** {essay_data['word_count']}")
            st.text_area("Submitted Essay:", value=essay_data['essay_text'], height=250, disabled=True)
        else:
            st.info("No essay was submitted.")


    elif test_name == "Coding Test":
        score = st.session_state.score # This is the placeholder score
        st.markdown(f'<div class="score-card"><h2>Score: {score:.1f}%</h2></div>', unsafe_allow_html=True)

        if score >= 80:
            st.success("🌟 **Excellent coding skills!** Your solutions seem well-thought-out.")
        elif score >= 60:
            st.info("👍 **Good programming fundamentals.** Keep practicing to refine your solutions and handle edge cases.")
        else:
            st.warning("📚 **Keep coding!** Practice more problems to improve your problem-solving and implementation skills.")

        st.markdown("### Your Submitted Solutions:")
        if st.session_state.answers and st.session_state.answers[0].get("problems_solved"):
            for i, solved_problem in enumerate(st.session_state.answers[0]["problems_solved"]):
                st.markdown(f"**Problem {i+1}:** {solved_problem['problem_title']}")
                st.code(solved_problem['user_code'], language="python") # Assume Python for display
        else:
            st.info("No solutions were submitted or recorded.")


    else:
        # MCQ results
        total_questions = len(st.session_state.questions)
        correct_answers = st.session_state.score
        percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0

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

        # Detailed answers
        st.markdown("---")
        st.markdown("### 📝 Detailed Review")
        if st.session_state.answers:
            for i, answer_data in enumerate(st.session_state.answers):
                st.markdown(f"**Q{i+1}:** {answer_data['question']}")
                if answer_data.get("is_correct", False):
                    st.success(f"✅ Your Answer: {answer_data['user_answer_text']} (Correct)")
                elif answer_data.get("user_answer_letter") == "Skipped":
                    st.warning(f"⏭️ You Skipped this question. Correct Answer: {answer_data['correct_answer_text']}")
                else:
                    st.error(f"❌ Your Answer: {answer_data['user_answer_text']} (Incorrect)")
                
                st.markdown(f"**Correct Answer:** {answer_data['correct_answer_text']}")
                st.info(f"**Explanation:** {answer_data['explanation']}")
                st.markdown("---")
        else:
            st.info("No questions were attempted in this session.")

    # Save progress
    st.session_state.progress_data.append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "test_type": test_name,
        "score": st.session_state.score if test_name in ["Written English Test", "Coding Test"] else percentage
    })

    # Action buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Retake Test", key="retake_test_btn"):
            st.session_state.current_test = test_name
            st.session_state.test_start_time = datetime.now()
            st.session_state.score = 0
            st.session_state.answers = []
            st.session_state.current_question = 0
            st.session_state.questions = [] # Clear questions so they can be regenerated if needed
            st.session_state.essay_topic = ""
            st.session_state.coding_problems = []
            st.session_state.mode = "test" # Go back to test mode
            st.rerun()

    with col2:
        if st.button("🏠 Back to Dashboard", key="back_to_dashboard_btn"):
            reset_session_state_for_dashboard()
            st.session_state.mode = "dashboard"
            st.rerun()

def show_practice_mode():
    """Show practice mode interface"""
    st.markdown("---")
    st.markdown("## 🎯 Practice Mode")
    st.markdown("Practice specific topics without time pressure and get instant feedback.")

    # Only show test types that have specific topics (i.e., not essay/coding)
    practice_test_types = {k: v for k, v in TEST_CONFIGS.items() if k not in ["Written English Test", "Coding Test"]}
    
    # If currently in practice_questions mode, don't show select boxes again
    if st.session_state.mode != "practice_questions":
        selected_test_type = st.selectbox("Select Test Type:", list(practice_test_types.keys()), key="practice_test_type_select")

        if selected_test_type:
            config = TEST_CONFIGS[selected_test_type]
            
            # Display available topics
            st.markdown(f"**Available Topics for {selected_test_type}:**")
            st.write(", ".join(config['topics']))

            selected_topic = st.selectbox("Select Topic to Practice:", config['topics'], key="practice_topic_select")
            
            # NEW: Difficulty Level Selector for Practice Mode
            difficulty_levels = ["Easy", "Medium", "Hard"]
            selected_difficulty_practice = st.selectbox("Select Difficulty Level:", difficulty_levels, key="practice_difficulty_select")

            num_questions = st.slider("Number of Questions:", 1, 10, 5, key="practice_num_questions_slider")

            if st.button("Start Practice Session", key="start_practice_session_btn"):
                # Clear previous practice state before generating new questions
                st.session_state.questions = []
                st.session_state.answers = []
                st.session_state.current_question = 0
                st.session_state.score = 0 
                st.session_state.current_test = selected_test_type # Needed for show_mcq_interface context

                with st.spinner(f"Generating practice questions for {selected_topic} ({selected_difficulty_practice})..."):
                    # Pass the selected_difficulty_practice to the generation function
                    st.session_state.questions = generate_questions(selected_test_type, selected_topic, num_questions, selected_difficulty_practice)
                    if not st.session_state.questions:
                        st.error("Could not generate practice questions. Please try a different topic or check your API key.")
                        st.session_state.mode = "practice" # Stay in practice selection
                        return # Do not rerun here; let the outer main loop handle this
                
                st.session_state.mode = "practice_questions" # A new mode for active practice
                st.rerun()
    
    # Only show MCQ interface if actually in practice_questions mode
    if st.session_state.mode == "practice_questions":
        show_mcq_interface(is_practice_mode=True)


if __name__ == "__main__":
    main()
