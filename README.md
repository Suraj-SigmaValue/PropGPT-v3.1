# PropGPT - Django + React Migration

This project has been migrated from Streamlit to a Django REST API backend with React + Tailwind frontend.

## Architecture

- **Backend**: Django REST Framework API (Python)
- **Frontend**: React with Vite + Tailwind CSS (JavaScript)
- **LLM Integration**: OpenAI GPT-4o-mini / Google Gemini
- **Vector Store**: FAISS + BM25 hybrid retrieval
- **Caching**: Semantic response caching

## Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Copy `.env.example` to `.env` and configure:
```bash
copy .env.example .env
```

Edit `.env` and add your API keys:
```
OPENAI_API_KEY=sk-your-key-here
GOOGLE_API_KEY=your-google-key-here
USE_LLM=openai
DJANGO_SECRET_KEY=your-secret-key-here
```

6. Run migrations:
```bash
python manage.py migrate
```

7. Start the Django development server:
```bash
python manage.py runserver
```

The backend API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```

The default configuration points to `http://localhost:8000/api`

4. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Usage

1. Ensure both backend and frontend servers are running
2. Open `http://localhost:5173` in your browser
3. Select a comparison type (Location, City, or Project)
4. Choose items to compare (up to 5)
5. Select analysis categories
6. Ask questions about your selected items
7. View AI-generated analysis with metrics
8. Provide feedback using thumbs up/down buttons

## API Endpoints

- `GET /api/health/` - Health check
- `GET /api/comparison-items/` - Get available items for comparison type
- `POST /api/query/` - Submit analysis query
- `POST /api/feedback/` - Submit feedback (thumbs up/down)
- `GET /api/cache/stats/` - Get cache statistics
- `DELETE /api/cache/clear/` - Clear cache

## Features

- ✅ Multi-level analysis (Location, City, Project)
- ✅ LLM-powered query intelligence (OpenAI/Gemini)
- ✅ Hybrid retrieval (FAISS + BM25)
- ✅ Semantic response caching
- ✅ Human-in-the-loop feedback system
- ✅ Real-time response streaming
- ✅ Token usage tracking
- ✅ Dark theme UI matching original design

## Project Structure

```
PropGPT-v3.1/
├── backend/
│   ├── propgpt_api/          # Django project
│   ├── api/                  # Main API app
│   │   ├── services/         # Business logic
│   │   ├── views.py          # API endpoints
│   │   ├── serializers.py    # Request/response validation
│   │   └── urls.py           # URL routing
│   ├── agents.py             # LLM agents
│   ├── config.py             # Configuration
│   ├── mapping.py            # Data mappings
│   ├── prompts.py            # LLM prompts
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── services/         # API client
│   │   ├── App.jsx           # Main component
│   │   └── index.css         # Tailwind styles
│   ├── package.json          # Node dependencies
│   └── tailwind.config.js    # Tailwind configuration
├── Pune_Grand_Summary.xlsx   # Data file
└── README.md                 # This file
```

## Development

### Backend Development

- API views are in `backend/api/views.py`
- Business logic is in `backend/api/services/`
- Add new endpoints in `backend/api/urls.py`

### Frontend Development

- Main app logic is in `frontend/src/App.jsx`
- API calls are in `frontend/src/services/api.js`
- Styles are in `frontend/src/index.css`

## Troubleshooting

### Backend Issues

- **Import errors**: Ensure virtual environment is activated
- **Missing API keys**: Check `.env` file configuration
- **Data file not found**: Ensure `Pune_Grand_Summary.xlsx` is in project root

### Frontend Issues

- **API connection errors**: Verify backend is running on port 8000
- **CORS errors**: Check `CORS_ALLOWED_ORIGINS` in backend settings
- **Build errors**: Clear node_modules and reinstall: `rm -rf node_modules && npm install`

## License

Proprietary - SigmaValue

## Support

For issues or questions, contact the development team.
