"""Test configuration and fixtures."""

import os
import pytest
import tempfile
import shutil

@pytest.fixture(scope="session")
def test_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def test_data():
    """Create test dataset for training and inference."""
    return """
Question: What is 2+2?
Answer: Let me solve this step by step:
1) 2+2 is a basic addition problem
2) The sum of 2 and 2 is 4
Therefore, 2+2 = 4

Question: Write a Python function to add two numbers.
Answer: Here's a simple Python function to add two numbers:
def add_numbers(a, b):
    return a + b

Question: Explain what is machine learning?
Answer: Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.
"""

@pytest.fixture(scope="session")
def test_dataset(test_dir, test_data):
    """Create a test dataset file."""
    dataset_path = os.path.join(test_dir, "test_data.txt")
    with open(dataset_path, "w") as f:
        f.write(test_data)
    return dataset_path
