#!/bin/bash

echo "Setting up AI Voice Assistant with Gemini..."

# Activate virtual environment
echo "Activating virtual environment..."
source .ai/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt')"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "Please edit .env file and add your Gemini API key"
    echo "Get your API key from: https://makersuite.google.com/app/apikey"
fi

echo "Setup complete!"
echo "To run the voice assistant:"
echo "1. Edit .env file and add your Gemini API key"
echo "2. Run: python app.py"