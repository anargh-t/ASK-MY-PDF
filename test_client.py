"""
Simple test client for AskMyPDF API
"""
import requests
import os

BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "your_api_key_here")


def test_health():
    """Test the health check endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_process_pdf(pdf_path: str):
    """Test the PDF processing endpoint"""
    print(f"Processing PDF: {pdf_path}")
    with open(pdf_path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/process_pdf",
            files={"file": f},
            data={"chunk_size": 1000, "overlap": 200}
        )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_query(question: str, top_k: int = 5):
    """Test the query endpoint"""
    print(f"Querying: {question}")
    response = requests.post(
        f"{BASE_URL}/query",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"question": question, "top_k": top_k}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("AskMyPDF API Test Client")
    print("=" * 50)
    print()
    
    # Test health check
    test_health()
    
    # Uncomment and modify these to test with your PDF
    # test_process_pdf("sample_document.pdf")
    # test_query("What is the main topic of this document?")
    
    print("Testing complete!")

