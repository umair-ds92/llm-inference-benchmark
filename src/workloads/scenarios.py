"""
Workload Scenarios for LLM Benchmarking

Generates realistic prompts for different use cases:
- Q&A: Short factual questions
- Summarization: Long documents to summarize
- Code Generation: Programming tasks
- Conversation: Multi-turn dialogues
"""

from typing import List
import random


def generate_qa_prompts(num_samples: int = 100) -> List[str]:
    """
    Generate question-answering prompts.
    
    Args:
        num_samples: Number of prompts to generate
        
    Returns:
        List of Q&A prompts
    """
    questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "When was the Declaration of Independence signed?",
        "What is the largest planet in our solar system?",
        "Who painted the Mona Lisa?",
        "What is the chemical symbol for gold?",
        "How many continents are there?",
        "What is the boiling point of water in Celsius?",
        "Who invented the telephone?",
        "What is the currency of Japan?",
        "When did World War II end?",
        "What is the square root of 144?",
        "Who was the first president of the United States?",
        "What is photosynthesis?",
        "How many bones are in the human body?",
        "What is the capital of Australia?",
        "Who wrote The Great Gatsby?",
        "What is the formula for water?",
        "When was the internet invented?",
        "What is the smallest country in the world?",
        "Who discovered penicillin?",
        "What is the speed of sound?",
        "How many planets are in our solar system?",
        "What is the capital of Canada?",
        "Who painted Starry Night?",
        "What is Newton's first law of motion?",
        "When was the United Nations founded?",
        "What is the largest ocean on Earth?",
        "Who wrote To Kill a Mockingbird?",
    ]
    
    prompts = []
    for i in range(num_samples):
        prompts.append(questions[i % len(questions)])
    
    return prompts


def generate_summarization_prompts(num_samples: int = 100) -> List[str]:
    """
    Generate document summarization prompts.
    
    Args:
        num_samples: Number of prompts to generate
        
    Returns:
        List of summarization prompts
    """
    articles = [
        """Summarize the following article:

The global technology sector experienced significant shifts in 2024, with artificial intelligence leading the charge in innovation and investment. Major tech companies reported record spending on AI infrastructure, totaling over $200 billion in combined capital expenditures. This unprecedented investment wave has sparked debates about sustainability and returns on investment.

Meanwhile, regulatory bodies worldwide have begun implementing stricter guidelines for AI development and deployment. The European Union's AI Act, which came into effect in mid-2024, established a risk-based framework for AI systems. Similar legislation is being considered in the United States, China, and other major economies.

The semiconductor industry continues to face challenges with supply chain disruptions, though recent investments in domestic manufacturing capacity show promise. New fabrication plants in Arizona, Texas, and Ohio are expected to begin production within the next two years.

Please provide a concise summary.""",

        """Summarize this business report:

Q4 2024 Financial Performance

Revenue: The company reported total revenue of $12.4 billion for Q4 2024, representing an 18% year-over-year increase. Cloud services grew 34%, enterprise software grew 22%, and consulting services grew 12%.

Profitability: Operating income reached $3.2 billion with a 25.8% operating margin, up from 23.1% last year. Net income totaled $2.6 billion, or $4.23 per share.

Strategic Initiatives: Major accomplishments include the acquisition of CloudTech Solutions for $2.1 billion, expansion of data center capacity, and launch of next-generation AI platform.

Please summarize the key financial highlights.""",

        """Summarize this research abstract:

Climate change poses significant threats to global food security. This study examines potential impacts on crop yields across different regions and proposes adaptive strategies. We analyzed climate data from 1990-2023 and projected future scenarios.

Results indicate that without adaptive measures, global crop yields could decline by 15-30% by 2050. However, implementation of adaptive strategies could mitigate yield losses by up to 60%.

The study reveals significant regional variations, with tropical and subtropical regions facing the most severe impacts. Policy recommendations include increased investment in agricultural research and climate-smart farming practices.

Provide a summary of main findings.""",
    ]
    
    prompts = []
    for i in range(num_samples):
        prompts.append(articles[i % len(articles)])
    
    return prompts


def generate_code_generation_prompts(num_samples: int = 100) -> List[str]:
    """
    Generate code generation prompts.
    
    Args:
        num_samples: Number of prompts to generate
        
    Returns:
        List of code generation prompts
    """
    tasks = [
        "Write a Python function that calculates the factorial of a number.",
        "Create a JavaScript function to validate an email address.",
        "Write a SQL query to find the top 10 customers by total purchase amount.",
        "Implement a binary search algorithm in Python.",
        "Write a Python function to reverse a linked list.",
        "Create a REST API endpoint in Python Flask for user authentication.",
        "Write a function to detect palindromes in a string.",
        "Implement quicksort algorithm in Python.",
        "Write a Python function to calculate Fibonacci numbers using memoization.",
        "Create a function to merge two sorted arrays.",
        "Write a Python function to validate a binary search tree.",
        "Implement a function to find all permutations of a string.",
        "Write a SQL query to find duplicate records in a table.",
        "Create a Python function to convert Roman numerals to integers.",
        "Implement a depth-first search algorithm for a graph.",
        "Write a function to check if two strings are anagrams.",
        "Create a Python class for a simple bank account.",
        "Write a function to find the longest common subsequence.",
        "Implement bubble sort algorithm in Python.",
        "Create a function to check if a number is prime.",
    ]
    
    prompts = []
    for i in range(num_samples):
        prompts.append(tasks[i % len(tasks)])
    
    return prompts


def generate_conversation_prompts(num_samples: int = 50) -> List[str]:
    """
    Generate conversational prompts with context.
    
    Args:
        num_samples: Number of prompts to generate
        
    Returns:
        List of conversation prompts
    """
    conversations = [
        """Continue this conversation:

User: I'm thinking about switching careers to data science.
Assistant: That's interesting! What's motivating this change?
User: I've always been fascinated by data and want to make more analytical impact.
Assistant:""",

        """Continue this conversation:

User: Can you explain how neural networks work?
Assistant: Neural networks are computing systems inspired by biological neural networks.
User: What are the different layers for?
Assistant:""",

        """Continue this conversation:

User: I need help planning a trip to Japan.
Assistant: I'd be happy to help! How many days are you planning?
User: About 10 days, visiting Tokyo and Kyoto.
Assistant:""",

        """Continue this conversation:

User: My Python code keeps giving me an IndexError.
Assistant: Let me help you debug. Can you share the error message?
User: It says "list index out of range" on line 15.
Assistant:""",

        """Continue this conversation:

User: What's the difference between machine learning and deep learning?
Assistant: Machine learning is broader. Deep learning uses neural networks with multiple layers.
User: So deep learning is always better?
Assistant:""",
    ]
    
    prompts = []
    for i in range(num_samples):
        prompts.append(conversations[i % len(conversations)])
    
    return prompts