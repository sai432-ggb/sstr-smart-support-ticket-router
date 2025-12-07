"""
FastAPI Application for Smart Support Ticket Router
Production-ready REST API with authentication, monitoring, and documentation
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.classifier import TicketClassifier
from src.models.priority_detector import PriorityDetector
from src.models.forecaster import TicketForecaster

# Initialize FastAPI app
app = FastAPI(
    title="Smart Support Ticket Router API",
    description="AI-powered ticket classification, priority detection, and demand forecasting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global variables for models (loaded at startup)
classifier = None
priority_detector = None
forecaster = None

# Rate limiting storage (simple in-memory, use Redis for production)
request_counts = {}


# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class TicketInput(BaseModel):
    """Input schema for ticket classification"""
    subject: str = Field(..., min_length=5, max_length=500,
                        description="Ticket subject")
    description: str = Field(..., min_length=10, max_length=10000,
                            description="Ticket description")
    customer_email: Optional[str] = Field(None, description="Customer email")
    customer_tier: Optional[str] = Field("free", description="Customer tier")
    
    @validator('customer_tier')
    def validate_tier(cls, v):
        valid_tiers = ['free', 'business', 'enterprise']
        if v not in valid_tiers:
            raise ValueError(f'customer_tier must be one of {valid_tiers}')
        return v

    class Config:
        schema_extra = {
            "example": {
                "subject": "Cannot login to my account",
                "description": "I've been trying to log in for the past hour but keep getting error 401. This is blocking my entire team from working.",
                "customer_email": "user@company.com",
                "customer_tier": "business"
            }
        }


class ClassificationResponse(BaseModel):
    """Response schema for classification"""
    ticket_id: str
    category: str
    priority: str
    confidence: float
    predicted_resolution_time: str
    suggested_team: str
    explanation: str
    top_predictions: List[Dict[str, Any]]
    processing_time_ms: float


class ForecastRequest(BaseModel):
    """Request schema for forecasting"""
    days: int = Field(7, ge=1, le=30, description="Number of days to forecast")
    category: Optional[str] = Field(None, description="Specific category to forecast")
    
    class Config:
        schema_extra = {
            "example": {
                "days": 7,
                "category": "Technical Support"
            }
        }


class ForecastResponse(BaseModel):
    """Response schema for forecast"""
    forecast_period_days: int
    category: Optional[str]
    forecasts: List[Dict[str, Any]]
    summary: Dict[str, Any]
    generated_at: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    api_version: str


# ============================================================================
# Authentication & Rate Limiting
# ============================================================================

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token (simplified - use proper auth in production)"""
    # In production, validate against database or JWT
    valid_tokens = os.getenv("API_TOKENS", "test-token-123").split(",")
    
    if credentials.credentials not in valid_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials


async def rate_limit(request: Request):
    """Simple rate limiting (use Redis for production)"""
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean old entries
    cutoff_time = current_time - 60  # 1-minute window
    request_counts[client_ip] = [
        ts for ts in request_counts.get(client_ip, []) 
        if ts > cutoff_time
    ]
    
    # Check limit
    if len(request_counts.get(client_ip, [])) >= 100:  # 100 requests per minute
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Add current request
    request_counts.setdefault(client_ip, []).append(current_time)


# ============================================================================
# Startup & Shutdown Events
# ============================================================================

@app.on_event("startup")
async def load_models():
    """Load ML models on startup"""
    global classifier, priority_detector, forecaster
    
    print("🚀 Loading models...")
    
    try:
        # Load classifier
        classifier = TicketClassifier()
        if os.path.exists('models/saved_models/classifier_v1.pkl'):
            classifier.load(
                'models/saved_models/classifier_v1.pkl',
                'models/saved_models/vectorizer_v1.pkl'
            )
            print("✓ Classifier loaded")
        else:
            print("⚠️ Classifier model not found - training required")
        
        # Load priority detector
        priority_detector = PriorityDetector()
        print("✓ Priority detector initialized")
        
        # Load forecaster
        forecaster = TicketForecaster()
        if os.path.exists('models/saved_models/forecaster_v1.pkl'):
            forecaster.load('models/saved_models/forecaster_v1.pkl')
            print("✓ Forecaster loaded")
        else:
            print("⚠️ Forecaster model not found - training required")
        
        print("✅ All models loaded successfully\n")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Shutting down API...")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Smart Support Ticket Router API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded={
            "classifier": classifier is not None and classifier.model is not None,
            "priority_detector": priority_detector is not None,
            "forecaster": forecaster is not None and forecaster.model is not None
        },
        api_version="1.0.0"
    )


@app.post("/api/v1/predict/classify", response_model=ClassificationResponse)
async def classify_ticket(
    ticket: TicketInput,
    request: Request,
    token: str = Depends(verify_token)
):
    """
    Classify a support ticket
    
    Analyzes the ticket and predicts:
    - Category (Technical Support, Billing, etc.)
    - Priority (Critical, High, Medium, Low)
    - Suggested team and resolution time
    """
    await rate_limit(request)
    
    start_time = time.time()
    
    try:
        if classifier is None or classifier.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Classification model not loaded"
            )
        
        # Classify ticket
        classification = classifier.predict_single(
            ticket.subject,
            ticket.description
        )
        
        # Detect priority
        priority_result = priority_detector.predict_priority(
            ticket.subject,
            ticket.description,
            ticket.customer_tier
        )
        
        # Generate ticket ID
        ticket_id = f"TKT-{int(time.time() * 1000)}"
        
        # Map category to team and resolution time
        team_mapping = {
            "Technical Support": "Backend Team",
            "Billing & Payments": "Finance Team",
            "Product Inquiry": "Sales Team",
            "Account Management": "Customer Success",
            "Feature Request": "Product Team",
            "Bug Report": "Engineering Team",
            "General Inquiry": "Support Team"
        }
        
        resolution_time_mapping = {
            "critical": "1-2 hours",
            "high": "4-6 hours",
            "medium": "12-24 hours",
            "low": "24-48 hours"
        }
        
        # Generate explanation
        explanation = priority_detector.explain_priority(
            ticket.subject,
            ticket.description,
            ticket.customer_tier
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return ClassificationResponse(
            ticket_id=ticket_id,
            category=classification['category'],
            priority=priority_result['priority'],
            confidence=classification['confidence'],
            predicted_resolution_time=resolution_time_mapping[priority_result['priority']],
            suggested_team=team_mapping.get(classification['category'], "Support Team"),
            explanation=explanation,
            top_predictions=classification['top_predictions'],
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification error: {str(e)}"
        )


@app.post("/api/v1/predict/priority", response_model=Dict[str, Any])
async def detect_priority(
    ticket: TicketInput,
    request: Request,
    token: str = Depends(verify_token)
):
    """
    Detect ticket priority only
    
    Analyzes sentiment, urgency keywords, and patterns
    to determine priority level
    """
    await rate_limit(request)
    
    try:
        if priority_detector is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Priority detector not loaded"
            )
        
        result = priority_detector.predict_priority(
            ticket.subject,
            ticket.description,
            ticket.customer_tier
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Priority detection error: {str(e)}"
        )


@app.post("/api/v1/forecast/demand", response_model=ForecastResponse)
async def forecast_demand(
    forecast_req: ForecastRequest,
    request: Request,
    token: str = Depends(verify_token)
):
    """
    Forecast ticket demand
    
    Predicts ticket volume for resource planning
    """
    await rate_limit(request)
    
    try:
        if forecaster is None or forecaster.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Forecaster model not loaded"
            )
        
        # Generate forecast
        forecast_df = forecaster.forecast(periods=forecast_req.days)
        
        # Convert to list of dicts
        forecasts = forecast_df.to_dict('records')
        for f in forecasts:
            f['date'] = f['date'].isoformat()
        
        # Calculate summary statistics
        summary = {
            "total_forecasted": float(forecast_df['forecast'].sum()),
            "daily_average": float(forecast_df['forecast'].mean()),
            "peak_day": forecast_df.loc[forecast_df['forecast'].idxmax(), 'date'].isoformat(),
            "peak_volume": float(forecast_df['forecast'].max()),
            "confidence_interval_width": float(
                (forecast_df['upper_bound'] - forecast_df['lower_bound']).mean()
            )
        }
        
        return ForecastResponse(
            forecast_period_days=forecast_req.days,
            category=forecast_req.category,
            forecasts=forecasts,
            summary=summary,
            generated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecasting error: {str(e)}"
        )


@app.get("/api/v1/categories", response_model=List[str])
async def get_categories(token: str = Depends(verify_token)):
    """Get list of supported ticket categories"""
    if classifier is None or classifier.classes_ is None:
        return [
            "Technical Support",
            "Billing & Payments",
            "Product Inquiry",
            "Account Management",
            "Feature Request",
            "Bug Report",
            "General Inquiry"
        ]
    return classifier.classes_.tolist()


@app.get("/api/v1/stats", response_model=Dict[str, Any])
async def get_statistics(token: str = Depends(verify_token)):
    """Get API statistics"""
    return {
        "total_requests": sum(len(v) for v in request_counts.values()),
        "active_clients": len(request_counts),
        "models_loaded": {
            "classifier": classifier is not None and classifier.model is not None,
            "priority_detector": priority_detector is not None,
            "forecaster": forecaster is not None and forecaster.model is not None
        },
        "uptime_seconds": int(time.time() - app.state.start_time) if hasattr(app.state, 'start_time') else 0
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# Middleware
# ============================================================================

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
    return response


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Store start time
    app.state.start_time = time.time()
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )