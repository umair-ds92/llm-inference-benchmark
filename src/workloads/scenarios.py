"""Generate simple Q&A prompts"""
from typing import List

def generate_qa_prompts(num_samples: int = 10) -> List[str]:
    """Generate simple factual questions"""
    templates = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "When was the Declaration of Independence signed?",
        "What is the largest planet in our solar system?",
        "Who painted the Mona Lisa?",
        "What is the chemical symbol for gold?",
        "How many continents are there?",
        "What is the boiling point of water?",
        "Who invented the telephone?",
    ]
    
    # Cycle through templates
    prompts = []
    for i in range(num_samples):
        prompts.append(templates[i % len(templates)])
    
    return prompts