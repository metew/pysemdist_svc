from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import Optional, Dict
from .. import __version__
from ..config import settings
from ..repository import (
    fetch_country_petitions,
    fetch_city_petitions,
    fetch_dm_petitions,
    fetch_topic_petitions,
)
from ..io_utils import save_results, maybe_upload_s3

app = FastAPI(title="pysemdist API", version=__version__)

class Health(BaseModel):
    status: str = "ok"

@app.get("/healthz", response_model=Health)
def healthz():
    return Health()

class DateRange(BaseModel):
    start_date: str = Field(..., description="Inclusive start date, YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="Optional inclusive end date, YYYY-MM-DD")

class CountryReq(DateRange):
    country_code: str = Field(default_factory=lambda: settings.default_country_code)
    limit: int = 500

class CityReq(DateRange):
    country_code: str = Field(default_factory=lambda: settings.default_country_code)
    city: str
    limit: int = 500

class DMReq(DateRange):
    country_code: str = Field(default_factory=lambda: settings.default_country_code)
    dm_id: int
    limit: int = 500

class TopicReq(DateRange):
    country_code: str = Field(default_factory=lambda: settings.default_country_code)
    topic_id: int
    limit: int = 500

class ExtractResponse(BaseModel):
    count: int
    sample: list[dict]
    output_paths: Dict[str, str]
    s3_locations: Dict[str, str] | dict

@app.post("/extract/country", response_model=ExtractResponse)
async def extract_country(req: CountryReq):
    df = await run_in_threadpool(fetch_country_petitions, req.start_date, req.end_date, req.country_code, req.limit)
    group = f"country_{req.country_code.lower()}"
    paths = save_results(df, group=group)
    s3_locs = maybe_upload_s3(paths)
    return ExtractResponse(count=len(df), sample=df.head(5).to_dict(orient="records"), output_paths=paths, s3_locations=s3_locs)

@app.post("/extract/city", response_model=ExtractResponse)
async def extract_city(req: CityReq):
    df = await run_in_threadpool(fetch_city_petitions, req.start_date, req.end_date, req.country_code, req.city, req.limit)
    group = f"city_{req.city.lower().replace(' ', '')}"
    paths = save_results(df, group=group)
    s3_locs = maybe_upload_s3(paths)
    return ExtractResponse(count=len(df), sample=df.head(5).to_dict(orient="records"), output_paths=paths, s3_locations=s3_locs)

@app.post("/extract/dm", response_model=ExtractResponse)
async def extract_dm(req: DMReq):
    df = await run_in_threadpool(fetch_dm_petitions, req.start_date, req.end_date, req.country_code, req.dm_id, req.limit)
    group = f"dm_{req.dm_id}"
    paths = save_results(df, group=group)
    s3_locs = maybe_upload_s3(paths)
    return ExtractResponse(count=len(df), sample=df.head(5).to_dict(orient="records"), output_paths=paths, s3_locations=s3_locs)

@app.post("/extract/topic", response_model=ExtractResponse)
async def extract_topic(req: TopicReq):
    df = await run_in_threadpool(fetch_topic_petitions, req.start_date, req.end_date, req.country_code, req.topic_id, req.limit)
    group = f"topic_{req.topic_id}"
    paths = save_results(df, group=group)
    s3_locs = maybe_upload_s3(paths)
    return ExtractResponse(count=len(df), sample=df.head(5).to_dict(orient="records"), output_paths=paths, s3_locations=s3_locs)

from ..goals_service import build_goal_sets
from ..summarizer import summarize_goal
from ..repository import (
    fetch_country_petitions,
    fetch_city_petitions,
    fetch_dm_petitions,
    fetch_topic_petitions,
)
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Optional, List

class ClusterParams(BaseModel):
    start_date: str
    end_date: Optional[str] = None
    country_code: Optional[str] = None
    city: Optional[str] = None
    dm_id: Optional[int] = None
    topic_id: Optional[int] = None
    limit: int = 500
    metric: str = "euclidean"          # HDBSCAN metric approximation (cosine via normalized vectors)
    min_cluster_size: int = 8
    min_samples: Optional[int] = None
    model_name: str = "all-MiniLM-L6-v2"

class GoalSet(BaseModel):
    goal_id: str
    goal_description: str
    total_petitions: int
    petition_ids: List[int]

@app.post("/cluster/goals", response_model=List[GoalSet])
async def cluster_goals(params: ClusterParams):
    # Gather data based on provided filters (AND/OR semantics: union unique ids)
    import pandas as pd
    frames = []
    if params.country_code:
        frames.append(await run_in_threadpool(fetch_country_petitions, params.start_date, params.end_date, params.country_code, params.limit))
    if params.city and params.country_code:
        frames.append(await run_in_threadpool(fetch_city_petitions, params.start_date, params.end_date, params.country_code, params.city, params.limit))
    if params.dm_id and params.country_code:
        frames.append(await run_in_threadpool(fetch_dm_petitions, params.start_date, params.end_date, params.country_code, params.dm_id, params.limit))
    if params.topic_id and params.country_code:
        frames.append(await run_in_threadpool(fetch_topic_petitions, params.start_date, params.end_date, params.country_code, params.topic_id, params.limit))
    if not frames and params.country_code:
        frames.append(await run_in_threadpool(fetch_country_petitions, params.start_date, params.end_date, params.country_code, params.limit))

    if not frames:
        return []

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
    # Build goals
    goals = await run_in_threadpool(build_goal_sets, df,)
    # Summarize each goal using a few petition texts as context
    # Build a map id->text
    text_map = dict(zip(df["id"].tolist(), df["text"].astype(str).tolist()))
    results = []
    for g in goals:
        texts = [text_map[i] for i in g["petition_ids"][:5] if i in text_map]
        desc = await run_in_threadpool(summarize_goal, texts)
        results.append(GoalSet(goal_id=g["goal_id"], goal_description=desc, total_petitions=g["total_petitions"], petition_ids=g["petition_ids"]))
    return results
