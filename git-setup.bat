@echo off
echo Setting up Git repository...

REM Initialize git repository
git init

REM Add all files
git add .

REM Create initial commit
git commit -m "Initial commit: Course Q&A RAG Chatbot with hybrid retrieval and multilingual support"

echo.
echo Git repository initialized successfully!
echo.
echo Next steps:
echo 1. Create a new repository on GitHub
echo 2. Run: git remote add origin https://github.com/YOUR_USERNAME/rag-chatbot.git
echo 3. Run: git branch -M main
echo 4. Run: git push -u origin main
echo.
pause