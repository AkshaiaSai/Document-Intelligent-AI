#!/usr/bin/env python3
"""Test script to verify Gemini API connectivity."""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ GOOGLE_API_KEY not found in .env file")
    exit(1)

print(f"✅ API Key found: {api_key[:10]}...")

# Configure Gemini
try:
    genai.configure(api_key=api_key)
    print("✅ Gemini configured successfully")
except Exception as e:
    print(f"❌ Error configuring Gemini: {e}")
    exit(1)

# Test embedding generation
try:
    print("\n🔍 Testing embedding generation...")
    result = genai.embed_content(
        model="models/text-embedding-004",
        content="This is a test",
        task_type="RETRIEVAL_DOCUMENT"
    )
    print(f"✅ Embedding generated successfully! Dimension: {len(result['embedding'])}")
except Exception as e:
    print(f"❌ Error generating embedding: {e}")
    print("\n💡 Troubleshooting:")
    print("   1. Check your internet connection")
    print("   2. Try: ping generativelanguage.googleapis.com")
    print("   3. Check if you're behind a firewall/proxy")
    print("   4. Verify API key is valid at https://makersuite.google.com/app/apikey")
    exit(1)

# Test LLM generation
try:
    print("\n🤖 Testing LLM generation...")
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Say hello")
    print(f"✅ LLM response: {response.text[:50]}...")
except Exception as e:
    print(f"❌ Error with LLM: {e}")
    exit(1)

print("\n🎉 All tests passed! Your Gemini API is working correctly.")
