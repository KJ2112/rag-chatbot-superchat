 @echo off
 echo 🚀 Setting up CodeWhisperer (Gemini)...

 python -m venv venv
 call venv\Scripts\activate

 pip install -r requirements.txt

 if not exist data mkdir data
 if not exist chroma_data mkdir chroma_data

 if not exist .env (
   copy env.example .env >nul
   echo ✅ Created .env - add your GOOGLE_API_KEY
 ) else (
   echo ✅ .env already exists
 )

 echo.
 echo 📝 Next steps:
 echo 1. Edit .env with your GOOGLE_API_KEY (get from aistudio.google.com)
 echo 2. Run: streamlit run app.py
 echo.
 pause

