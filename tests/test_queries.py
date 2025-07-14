"""
Test queries for evaluating RAG system performance
"""
from typing import List

TEST_QUERIES = [
    {
        "query": "What's the vacation policy for employees with 3 years of experience?",
        "expected_topics": ["vacation", "PTO", "20 days"],
        "difficulty": "easy"
    },
    {
        "query": "How do I book the Golden Gate conference room?",
        "expected_topics": ["Golden Gate", "conference room", "Google Calendar"],
        "difficulty": "easy"
    },
    {
        "query": "What are the API rate limits for enterprise customers?",
        "expected_topics": ["API", "rate limits", "10,000 requests/hour"],
        "difficulty": "medium"
    },
    {
        "query": "What's the process for reporting a security incident?",
        "expected_topics": ["security incident", "contain threat", "notify security team"],
        "difficulty": "medium"
    },
    {
        "query": "What benefits are available and what does the company pay for health insurance?",
        "expected_topics": ["health insurance", "90%", "Blue Cross"],
        "difficulty": "hard"
    },
    {
        "query": "What's the address of the London office?",
        "expected_topics": ["London", "office", "address"],
        "difficulty": "easy",
        "source_type": "pdf"
    },
    {
        "query": "What authentication methods are supported by the API?",
        "expected_topics": ["API", "authentication", "OAuth", "API key"],
        "difficulty": "medium",
        "source_type": "pdf"
    },
    {
        "query": "Who is the Engineering Manager and what's their email?",
        "expected_topics": ["Mike Chen", "Engineering Manager", "mike.chen@techcorp.com"],
        "difficulty": "easy",
        "source_type": "csv"
    },
    {
        "query": "Which employees work remotely?",
        "expected_topics": ["Remote", "Alex Kumar", "Ryan Thompson"],
        "difficulty": "medium",
        "source_type": "csv"
    }
]

def evaluate_response(query: str, response: str, expected_topics: List[str]) -> float:
    """
    Simple evaluation function to check if response contains expected topics
    
    Returns:
        Score between 0 and 1 based on topic coverage
    """
    # TODO: Implement evaluation logic
    pass

def run_evaluation():
    """Run evaluation on all test queries"""
    # TODO: Implement full evaluation suite
    pass