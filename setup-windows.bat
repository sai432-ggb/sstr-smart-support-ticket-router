@echo off
echo Setting up Smart Support Ticket Router for Windows...

REM Activate virtual environment
call venv\Scripts\activate

REM Install packages
pip install --upgrade pip
pip install scikit-learn numpy pandas scipy fastapi uvicorn[standard] pydantic nltk textblob python-dotenv pyyaml requests joblib pytest statsmodels matplotlib seaborn tqdm click loguru sqlalchemy httpx pytest-asyncio pytest-cov python-multipart black ipykernel

REM Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

REM Create directories
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist data\features mkdir data\features
if not exist models\saved_models mkdir models\saved_models
if not exist models\metrics mkdir models\metrics
if not exist logs mkdir logs

REM Copy environment file
if not exist .env copy .env.example .env

echo.
echo Setup complete! Now run:
echo   python scripts\generate_data.py
echo   python scripts\train_models.py
echo   python -m uvicorn src.api.main:app --reload