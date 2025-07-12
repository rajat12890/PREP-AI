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

def generate_questions(test_type, topic, count=5):
    """Generate practice questions using Groq"""
    llm = initialize_groq_client()
    if not llm:
        st.warning(f"Using sample questions for {test_type} - {topic}. Groq API key is missing or invalid.")
        return create_sample_questions(test_type, topic, count)

    if test_type == "English Usage Test":
        prompt_template = """Generate {count} multiple choice questions for '{topic}' for a CSE employability test.
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
        prompt_template = """Generate {count} analytical reasoning questions for '{topic}' for a CSE employability test.
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
        prompt_template = """Generate {count} quantitative ability questions for '{topic}' for a CSE employability test.
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
        prompt_template = """Generate {count} Data Structures and Algorithms questions for '{topic}' for a CSE employability test.
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
        input_variables=["topic", "count"],
        template=prompt_template
    )

    try:
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(topic=topic, count=count)
        # print(f"DEBUG: Raw AI Response for {test_type} - {topic}:\n{response[:500]}...") # Print first 500 chars for debug

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
                    return create_sample_questions(test_type, topic, count)
            else:
                st.warning("AI response did not contain a valid JSON array. Using sample questions.")
                return create_sample_questions(test_type, topic, count)
        except json.JSONDecodeError as e:
            st.warning(f"JSON parsing error: {e}. AI response might be malformed. Using sample questions.")
            # print(f"DEBUG: JSONDecodeError: {e}, Response causing error: {response}") # More detailed error
            return create_sample_questions(test_type, topic, count)
        except Exception as e:
            st.warning(f"An unexpected error occurred during JSON processing: {e}. Using sample questions.")
            # print(f"DEBUG: Unexpected error during JSON processing: {e}, Response: {response}")
            return create_sample_questions(test_type, topic, count)

    except Exception as e:
        st.error(f"Error generating questions from Groq: {str(e)}. Using sample questions.")
        # print(f"DEBUG: Groq API call failed: {e}")
        return create_sample_questions(test_type, topic, count)

def create_sample_questions(test_type, topic, count):
    """Create sample questions as fallback if AI generation fails or API key is missing.
    Ensures that 'count' questions are returned, even if by repeating existing samples."""
    all_sample_q = []

    if test_type == "English Usage Test":
        all_sample_q = [
            {"question": "Choose the correct article: __ apple a day keeps the doctor away.", "options": ["A) A", "B) An", "C) The", "D) No article"], "correct_answer": "B", "explanation": "Use 'an' before words that start with a vowel sound."},
            {"question": "Identify the idiom: 'Bite the bullet'", "options": ["A) To eat quickly", "B) To face a difficult situation with courage", "C) To chew on metal", "D) To harm someone"], "correct_answer": "B", "explanation": "To 'bite the bullet' means to endure a difficult or unpleasant situation."},
            {"question": "Correct the sentence: 'He don't know nothing.'", "options": ["A) He doesn't know nothing.", "B) He don't know anything.", "C) He doesn't know anything.", "D) He knows nothing."], "correct_answer": "C", "explanation": "Avoid double negatives. 'He doesn't know anything' or 'He knows nothing' are correct."},
            {"question": "What is the antonym of 'Efficient'?", "options": ["A) Productive", "B) Capable", "C) Inefficient", "D) Effective"], "correct_answer": "C", "explanation": "Efficient means performing or functioning in the best possible manner with the least waste of time and effort; Inefficient is its opposite."},
            {"question": "Choose the correctly spelled word:", "options": ["A) Occassion", "B) Occasion", "C) Ocasion", "D) Occassionn"], "correct_answer": "B", "explanation": "The correct spelling is 'occasion'."}
        ]
    elif test_type == "Analytical Reasoning Test":
        all_sample_q = [
            {"question": "Find the missing number in the series: 2, 4, 8, 16, __", "options": ["A) 24", "B) 32", "C) 48", "D) 64"], "correct_answer": "B", "explanation": "This is a geometric progression where each number is doubled (2*2=4, 4*2=8, 8*2=16, 16*2=32)."},
            {"question": "Which of the following is different from the rest?", "options": ["A) Apple", "B) Banana", "C) Potato", "D) Grape"], "correct_answer": "C", "explanation": "Potato is a vegetable, while the others are fruits."},
            {"question": "If 'BAT' is coded as '2120', how is 'CAT' coded?", "options": ["A) 3120", "B) 231", "C) 321", "D) 123"], "correct_answer": "A", "explanation": "Each letter is assigned its alphabetical position (A=1, B=2, C=3, etc.). So, C=3, A=1, T=20, making CAT = 3120."},
            {"question": "All cats are animals. Some animals are black. Therefore, some cats are black. Is this statement:", "options": ["A) True", "B) False", "C) Cannot be determined", "D) Irrelevant"], "correct_answer": "C", "explanation": "This is an example of an invalid syllogism. Just because some animals are black doesn't mean those specific black animals are cats. We cannot determine if some cats are black from the given premises."},
            {"question": "A is taller than B. B is taller than C. D is taller than A. Who is the tallest?", "options": ["A) A", "B) B", "C) C", "D) D"], "correct_answer": "D", "explanation": "The order of height is C < B < A < D, so D is the tallest."}
        ]
    elif test_type == "Quantitative Ability Test":
        all_sample_q = [
            {"question": "A shopkeeper sells an item for $120, making a profit of 20%. What was the cost price?", "options": ["A) $96", "B) $100", "C) $110", "D) $144"], "correct_answer": "B", "explanation": "If selling price is $120 and profit is 20%, then $120 = Cost Price * (1 + 0.20) = 1.2 * Cost Price. So, Cost Price = 120 / 1.2 = $100."},
            {"question": "If 5 workers can complete a task in 10 days, how many days will 10 workers take to complete the same task?", "options": ["A) 2 days", "B) 5 days", "C) 10 days", "D) 20 days"], "correct_answer": "B", "explanation": "Total work = Workers × Days = 5 × 10 = 50 units. For 10 workers, Days = Total Work / Workers = 50 / 10 = 5 days."},
            {"question": "What is the area of a circle with radius 7 cm? (Use π = 22/7)", "options": ["A) 44 sq cm", "B) 154 sq cm", "C) 22 sq cm", "D) 77 sq cm"], "correct_answer": "B", "explanation": "Area of circle = πr² = (22/7) * 7 * 7 = 22 * 7 = 154 sq cm."},
            {"question": "If x + 5 = 12, what is the value of x?", "options": ["A) 5", "B) 7", "C) 12", "D) 17"], "correct_answer": "B", "explanation": "Subtract 5 from both sides: x = 12 - 5 = 7."},
            {"question": "What is the average of 10, 20, and 30?", "options": ["A) 15", "B) 20", "C) 25", "D) 60"], "correct_answer": "B", "explanation": "Average = (Sum of numbers) / (Count of numbers) = (10 + 20 + 30) / 3 = 60 / 3 = 20."}
        ]
    elif test_type == "Domain Test (DSA)":
        all_sample_q = [
            {"question": "Which data structure is best for implementing a 'undo' feature in an editor?", "options": ["A) Queue", "B) Array", "C) Stack", "D) Hash Table"], "correct_answer": "C", "explanation": "A stack is suitable for 'undo' operations because it follows LIFO (Last-In, First-Out), meaning the last element added is the first to be undone."},
            {"question": "What is the worst-case time complexity of inserting an element into a sorted linked list?", "options": ["A) O(1)", "B) O(log n)", "C) O(n)", "D) O(n log n)"], "correct_answer": "C", "explanation": "In the worst case, you might need to traverse the entire list to find the correct insertion point."},
            {"question": "Which algorithm finds the shortest path in a graph with non-negative edge weights?", "options": ["A) DFS", "B) BFS", "C) Dijkstra's Algorithm", "D) Prim's Algorithm"], "correct_answer": "C", "explanation": "Dijkstra's Algorithm is specifically designed for finding the shortest paths from a single source vertex to all other vertices in a graph with non-negative edge weights."},
            {"question": "What is a 'hash collision' in a hash table?", "options": ["A) When two different keys map to the same index", "B) When two identical keys map to different indices", "C) When the hash table is full", "D) When an element is deleted"], "correct_answer": "A", "explanation": "A hash collision occurs when the hash function generates the same index for two or more different keys."},
            {"question": "Which sorting algorithm has the best average-case time complexity?", "options": ["A) Bubble Sort", "B) Selection Sort", "C) Quick Sort", "D) Insertion Sort"], "correct_answer": "C", "explanation": "Quick Sort typically has an average-case time complexity of O(n log n), which is generally considered the best among comparison-based sorts."}
        ]
    
    # Ensure we return at least 'count' questions by repeating if necessary
    if not all_sample_q: # Should not happen with current setup, but as a safeguard
        # If no samples are defined at all, return a very basic fallback
        return [{"question": f"Sample Question {i+1} for {test_type} {topic}", "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"], "correct_answer": "A", "explanation": "This is a fallback explanation."} for i in range(count)]

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
        # print("DEBUG: Groq client not initialized for essay topic generation. Using sample.")
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
        # print(f"DEBUG: Groq API call failed for essay topic: {e}")
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
        # print("DEBUG: Groq client not initialized for coding problem generation. Using sample.")
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
        # print(f"DEBUG: Raw AI Response for Coding Problems:\n{response[:500]}...") # Print first 500 chars for debug

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
            # print(f"DEBUG: JSONDecodeError for coding problems: {e}, Response causing error: {response}")
            return generate_coding_problems_fallback()
        except Exception as e:
            st.warning(f"An unexpected error occurred during coding problem JSON processing: {e}. Using sample problems.")
            # print(f"DEBUG: Unexpected error during coding problem JSON processing: {e}, Response: {response}")
            return generate_coding_problems_fallback()

    except Exception as e:
        st.error(f"Error generating coding problems from Groq: {str(e)}. Using sample problems.")
        # print(f"DEBUG: Groq API call failed for coding problems: {e}")
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

    # API Key Input (only show if not in test/practice_questions mode)
    if not st.session_state.groq_api_key and st.session_state.mode in ["dashboard", "practice"]:
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
    # Moved the timer logic out of show_test_interface and into main to control reruns better
    if st.session_state.mode == "test":
        test_name = st.session_state.current_test
        config = TEST_CONFIGS[test_name]
        
        st.markdown(f"---")
        st.markdown(f"## {config['icon']} {test_name}")

        # Placeholder for timer
        timer_placeholder = st.empty()

        # Update timer display
        if st.session_state.test_start_time:
            elapsed = datetime.now() - st.session_state.test_start_time
            remaining_seconds = int(timedelta(minutes=config['time_limit']).total_seconds() - elapsed.total_seconds())

            if remaining_seconds <= 0:
                timer_placeholder.markdown('<div class="timer">⏰ Time\'s up! Test completed.</div>', unsafe_allow_html=True)
                st.error("Time's up! Your test has been automatically submitted.")
                st.session_state.mode = "results"
                st.rerun()
                return # Exit main to go to results
            
            minutes, seconds = divmod(remaining_seconds, 60)
            timer_placeholder.markdown(f'<div class="timer">⏱️ Time Remaining: {minutes:02d}:{seconds:02d}</div>', unsafe_allow_html=True)
        
        # Render the specific test interface below the timer
        if test_name == "Written English Test":
            show_essay_interface()
        elif test_name == "Coding Test":
            show_coding_interface()
        else: # For MCQ-based tests
            show_mcq_interface()
        
        # This is where the magic happens for real-time updates within the "test" mode
        # Only rerun if still in "test" mode and timer is active
        if st.session_state.mode == "test" and st.session_state.test_start_time is not None:
            time.sleep(1)
            st.rerun() # Force rerun to update the timer and test content


    elif st.session_state.mode == "dashboard":
        show_dashboard()
    elif st.session_state.mode == "practice":
        show_practice_mode()
    elif st.session_state.mode == "practice_questions":
        show_practice_mode() # This will internally call show_mcq_interface(is_practice_mode=True)
    elif st.session_state.mode == "results":
        show_results()


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

    cols = st.columns(2)
    for i, (test_name, config) in enumerate(TEST_CONFIGS.items()):
        with cols[i % 2]:
            with st.container(border=True): # Using border for better visual separation
                st.markdown(f"""
                <div class="test-card">
                    <h3>{config['icon']} {test_name}</h3>
                    <p><strong>Topics:</strong> {len(config['topics'])} areas</p>
                    <p><strong>Questions:</strong> {config['question_count']}</p>
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
                            st.session_state.essay_topic = generate_essay_topic()
                        elif test_name == "Coding Test":
                            st.session_state.coding_problems = generate_coding_problems()
                        else:
                            # Generate questions for other tests
                            all_questions = []
                            # Distribute questions evenly among topics, then add any remainder to random topics
                            questions_per_topic = config['question_count'] // len(config['topics'])
                            remaining_questions = config['question_count'] % len(config['topics'])

                            for topic in config['topics']:
                                q_count = questions_per_topic
                                if remaining_questions > 0:
                                    q_count += 1
                                    remaining_questions -= 1
                                questions = generate_questions(test_name, topic, q_count)
                                all_questions.extend(questions)

                            random.shuffle(all_questions) # Shuffle for variety
                            st.session_state.questions = all_questions[:config['question_count']]
                            if not st.session_state.questions:
                                st.error(f"Failed to generate questions for {test_name}. Please check your API key or try again.")
                                st.session_state.mode = "dashboard" # Go back if no questions
                                # Do not rerun here; let the outer main loop handle the mode change.
                                return

                    st.rerun() # Only rerun once questions are generated/assigned

# show_test_interface function has been refactored and its content moved to main()
# Leaving this as a placeholder, but its logic is now primarily in main()
# def show_test_interface():
#     pass

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
    selected_option = st.radio("Choose your answer:", question_data["options"], key=f"mcq_q_{current_q_index}_{st.session_state.current_test}")
    
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Submit Answer", key=f"submit_mcq_{current_q_index}_{st.session_state.current_test}"):
            # Extract answer letter (A, B, C, D)
            answer_letter = selected_option[0] if selected_option else None # Get the first char (A, B, C, D)
            
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
                
                # Store the answer for review
                st.session_state.answers.append({
                    "question": question_data["question"],
                    "user_answer": answer_letter,
                    "correct_answer": question_data["correct_answer"],
                    "is_correct": correct,
                    "explanation": question_data["explanation"]
                })
                
                time.sleep(1)  # Show feedback briefly before moving on
                st.session_state.current_question += 1
                st.rerun()

    with col2:
        if st.button("Skip Question", key=f"skip_mcq_{current_q_index}_{st.session_state.current_test}"):
            st.session_state.answers.append({
                "question": question_data["question"],
                "user_answer": "Skipped",
                "correct_answer": question_data["correct_answer"],
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
        if i < len(st.session_state.answers) and isinstance(st.session_state.answers[i], dict) and "user_code" in st.session_state.answers[i]:
            current_solution = st.session_state.answers[i]["user_code"]
        
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
                    st.success(f"✅ Your Answer: {answer_data['user_answer']} (Correct)")
                elif answer_data.get("user_answer") == "Skipped":
                    st.warning(f"⏭️ You Skipped this question. Correct: {answer_data['correct_answer']}")
                else:
                    st.error(f"❌ Your Answer: {answer_data['user_answer']} (Incorrect), Correct: {answer_data['correct_answer']}")
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
            num_questions = st.slider("Number of Questions:", 1, 10, 5, key="practice_num_questions_slider")

            if st.button("Start Practice Session", key="start_practice_session_btn"):
                # Clear previous practice state before generating new questions
                st.session_state.questions = []
                st.session_state.answers = []
                st.session_state.current_question = 0
                st.session_state.score = 0 
                st.session_state.current_test = selected_test_type # Needed for show_mcq_interface context

                with st.spinner(f"Generating practice questions for {selected_topic}..."):
                    st.session_state.questions = generate_questions(selected_test_type, selected_topic, num_questions)
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