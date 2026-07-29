from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    city: str = ""
    family_size: Optional[int] = None
    commute_km: Optional[float] = None
    long_trip_frequency: str = ""
    has_home_charger: Optional[bool] = None
    preferred_type: str = ""
    preferred_energy: str = ""
    preferred_brands: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    intent_level: str = "了解中"


class RecommendRequest(BaseModel):
    query: str
    profile: UserProfile = Field(default_factory=UserProfile)
    top_k: int = 5
    use_deep_search: bool = False
    thread_id: str = ""
    candidate_pool_strategy: str = "auto"


class VehicleScore(BaseModel):
    id: int
    brand: str
    model: str
    vehicle_type: str
    energy_type: str
    price_min: int
    price_max: int
    cltc_range: int
    seats: int
    score: float
    budget_score: float
    range_score: float
    space_score: float
    charging_score: float
    smart_score: float
    safety_score: float
    scenario_score: float
    reasons: List[str]
    cautions: List[str]
    highlights: str = ""
    weaknesses: str = ""
    source_type: str = "local"
    source_url: str = ""
    source_title: str = ""


class RecommendResponse(BaseModel):
    profile: Dict[str, Any]
    recommendations: List[VehicleScore]
    answer: str
    agent_trace: List[Dict[str, Any]]
    skill_trace: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]


class CompareRequest(BaseModel):
    models: List[str]
    profile: UserProfile = Field(default_factory=UserProfile)


class ChatRequest(BaseModel):
    query: str
    top_k: int = 6
    use_web_search: bool = True
    history: List[Dict[str, str]] = Field(default_factory=list)


class DeepSearchRequest(BaseModel):
    query: str
    top_k: int = 6


class LeadCreate(BaseModel):
    name: str = "匿名客户"
    phone_masked: str = ""
    profile: UserProfile = Field(default_factory=UserProfile)
    recommended_models: List[str] = Field(default_factory=list)
    next_action: str = ""


class RecommendationFeedbackCreate(BaseModel):
    query: str = ""
    model_name: str = ""
    rating: str = "neutral"
    reason: str = ""
    candidate_pool: str = "unknown"
    scenario_tags: List[str] = Field(default_factory=list)
    profile: Dict[str, Any] = Field(default_factory=dict)
    recommendation: Dict[str, Any] = Field(default_factory=dict)
