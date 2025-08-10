from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import os
import re
import json
import google.generativeai as genai
from serpapi import GoogleSearch
from dotenv import load_dotenv


load_dotenv()

# --- MODELS ---
class VideoIdeaRequest(BaseModel):
    video_idea: str
    category_id: str = "71"  # Default to Food & Drink

class KeywordResponse(BaseModel):
    keyword: str
    original_idea: str

class TrendsResponse(BaseModel):
    keyword: str
    category_trends: Dict[str, List[str]]
    global_trends: List[str]

class ViralAngleRequest(BaseModel):
    original_idea: str
    category_trends: Dict[str, List[str]]
    global_trends: List[str]

class ViralAngleResponse(BaseModel):
    viral_angle: str
    selected_trends: List[str]

class ScriptRequest(BaseModel):
    viral_angle: str

class ScriptResponse(BaseModel):
    script: str

class FullPipelineResponse(BaseModel):
    original_idea: str
    keyword: str
    viral_angle: str
    selected_trends: List[str]
    script: str
    trends_data: Dict

# --- CONFIGURATION ---
UTILITY_MODEL = "gemini-2.5-flash-lite"
CREATIVE_MODEL = "gemini-2.5-flash-lite"

app = FastAPI(title="Viral Script Generator API", version="1.0.0")

class TrendInfusedScriptService:
    def __init__(self):
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        if not serpapi_key:
            raise ValueError("SERPAPI_API_KEY environment variable is required")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
            
        self.serpapi_api_key = serpapi_key
        genai.configure(api_key=gemini_key)
        self.utility_model = genai.GenerativeModel(model_name=UTILITY_MODEL)
        self.creative_model = genai.GenerativeModel(model_name=CREATIVE_MODEL)

    def extract_keyword(self, video_idea: str) -> str:
        """Extract base keyword from video idea"""
        prompt = f"""
        Analyze the following video idea and extract the single most important noun or short noun phrase (2-3 words max) that represents the core subject. This keyword will be used to search Google Trends.

        Your response MUST be only the keyword phrase itself, in lowercase, with no explanation, punctuation, or quotation marks.

        - Idea: "A fun 8s TikTok about why iced coffee is better than hot coffee"
        - Your Response: iced coffee

        - Idea: "Make a tutorial on how to bake sourdough bread"
        - Your Response: sourdough bread

        - Idea: "A review of the new Tesla Cybertruck"
        - Your Response: tesla cybertruck

        - Idea: "{video_idea}"
        - Your Response:
        """
        
        try:
            response = self.utility_model.generate_content(prompt)
            keyword = response.text.strip().lower()
            if not keyword or len(keyword) > 50:
                raise ValueError("Invalid keyword returned")
            return keyword
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to extract keyword: {str(e)}")

    def _parse_related_topics_csv(self, csv_data: list) -> dict:
        """Parse CSV data from SerpAPI trends"""
        if not csv_data:
            return {"top": [], "rising": []}
        
        trends = {"top": [], "rising": []}
        current_section = None
        
        for row in csv_data:
            if not row:
                continue
            if row.upper() == "TOP":
                current_section = "top"
                continue
            if row.upper() == "RISING":
                current_section = "rising"
                continue
            if ":" in row or not current_section:
                continue
            
            try:
                topic = re.sub(r',(\+?\d+%?|Breakout)$', '', row).strip()
                if topic:
                    trends[current_section].append(topic)
            except (IndexError, ValueError):
                continue
        
        return trends

    def fetch_trends(self, keyword: str, category_id: str) -> tuple:
        """Fetch trends from Google Trends"""
        try:
            # Category-specific trends
            params_cat = {
                "engine": "google_trends",
                "q": keyword,
                "cat": category_id,
                "data_type": "RELATED_TOPICS",
                "date": "today 1-m",
                "gprop": "youtube",
                "csv": "true",
                "api_key": self.serpapi_api_key
            }
            
            search_cat = GoogleSearch(params_cat)
            results_cat = search_cat.get_dict()
            
            category_trends = {"top": [], "rising": []}
            if "error" not in results_cat:
                category_trends = self._parse_related_topics_csv(results_cat.get("csv", []))

            # Global trends
            params_global = {
                "engine": "google_trends_trending_now",
                "api_key": self.serpapi_api_key
            }
            
            search_global = GoogleSearch(params_global)
            results_global = search_global.get_dict()
            
            global_trends = []
            if "error" not in results_global:
                trending_searches = []
                for key, value in results_global.items():
                    if (isinstance(value, list) and len(value) > 0 and 
                        isinstance(value[0], dict) and 'title' in value[0]):
                        trending_searches = value
                        break
                
                if trending_searches:
                    global_trends = [
                        t.get('title', {}).get('query') 
                        for t in trending_searches 
                        if t.get('title', {}).get('query')
                    ]

            return category_trends, global_trends

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch trends: {str(e)}")

    def generate_viral_angle(self, original_idea: str, category_trends: dict, global_trends: list) -> tuple:
        """Generate viral angle using AI"""
        # Merge trends
        merged, seen = [], set()
        for section in [category_trends.get("rising", []), category_trends.get("top", []), global_trends]:
            for trend in section:
                if trend.lower() not in seen:
                    merged.append(trend)
                    seen.add(trend.lower())
        
        trend_candidates = merged[:25]
        
        if not trend_candidates:
            return original_idea, []

        trend_list_str = "\n".join(f'- "{trend}"' for trend in trend_candidates)
        prompt = f"""
        You are a world-class social media strategist and viral trend analyst. Your task is to analyze a list of raw trend candidates and an original video idea, then formulate a single, concise 'Viral Angle' prompt for a scriptwriter AI.

        **Original Idea:** "{original_idea}"

        **Trend Candidates:**
        {trend_list_str}

        **Your Process:**
        1.  **Analyze & Filter:** Review all trend candidates. You MUST DISCARD trends that are:
            - Generic or redundant (e.g., 'Coffee', 'Cup' if the topic is coffee).
            - Brand-unsafe, political, religious, or overly niche/strange.
        2.  **Select for Virality:** From the safe and interesting trends, identify the 1-3 trends with the HIGHEST potential to make the video viral, engaging, and unique.
        3.  **Formulate Viral Angle:** Combine the original idea with your chosen viral trends into a single, creative instruction sentence. This sentence is the final output.
        4.  **Output Format:** Your final response MUST be a single JSON object containing two keys: "selected_trends" (a list of the exact trend names you chose) and "viral_angle" (the instruction sentence you formulated). Do not add any other text or markdown.

        **EXAMPLE:**
        - Original Idea: "Make a fun 8s TikTok about why iced coffee is better than hot coffee"
        - Trend Candidates: ["Coffee", "Starbucks", "Dalgona coffee", "Autumn", "Cozy Coffee Shop"]
        - Your Response:
        {{
            "selected_trends": ["Dalgona coffee", "Cozy Coffee Shop"],
            "viral_angle": "Create a fun 8s TikTok showing iced coffee is superior to hot coffee by contrasting a boring hot coffee with a trendy, aesthetic 'Dalgona coffee' in a 'Cozy Coffee Shop' setting."
        }}
        """
        
        try:
            response = self.utility_model.generate_content(prompt)
            cleaned_text = re.sub(r'```json\n?|```', '', response.text.strip())
            result_json = json.loads(cleaned_text)
            
            viral_angle = result_json.get("viral_angle")
            selected_trends = result_json.get("selected_trends", [])
            
            if not viral_angle or not isinstance(selected_trends, list):
                raise ValueError("AI response missing required keys")
                
            return viral_angle, selected_trends
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate viral angle: {str(e)}")

    def generate_script(self, viral_angle: str) -> str:
        """Generate script from viral angle"""
        prompt = f"""
        You are an expert short-form video scriptwriter for platforms like TikTok and Instagram Reels.
        Your task is to take a specific 'Viral Angle' and turn it into a complete, ready-to-shoot script concept.

        **Viral Angle / Core Instruction:**
        "{viral_angle}"

        **Your Output:**
        - Create a short, concise script (for an 8-15 second video).
        - Include scene descriptions, dialogue/VO, and suggested on-screen text/hashtags.
        - The tone should be authentic, punchy, and highly shareable.
        """
        
        try:
            response = self.creative_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate script: {str(e)}")

# Initialize service
service = TrendInfusedScriptService()

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {"message": "Viral Script Generator API", "version": "1.0.0"}

@app.post("/extract-keyword", response_model=KeywordResponse)
async def extract_keyword(request: VideoIdeaRequest):
    """Extract keyword from video idea"""
    keyword = service.extract_keyword(request.video_idea)
    return KeywordResponse(keyword=keyword, original_idea=request.video_idea)

@app.post("/fetch-trends", response_model=TrendsResponse)
async def fetch_trends(request: dict):
    """Fetch trends based on keyword and category"""
    keyword = request.get("keyword")
    category_id = request.get("category_id", "71")
    
    if not keyword:
        raise HTTPException(status_code=400, detail="Keyword is required")
    
    category_trends, global_trends = service.fetch_trends(keyword, category_id)
    
    return TrendsResponse(
        keyword=keyword,
        category_trends=category_trends,
        global_trends=global_trends
    )

@app.post("/generate-viral-angle", response_model=ViralAngleResponse)
async def generate_viral_angle(request: ViralAngleRequest):
    """Generate viral angle from trends and original idea"""
    viral_angle, selected_trends = service.generate_viral_angle(
        request.original_idea,
        request.category_trends,
        request.global_trends
    )
    
    return ViralAngleResponse(
        viral_angle=viral_angle,
        selected_trends=selected_trends
    )

@app.post("/generate-script", response_model=ScriptResponse)
async def generate_script(request: ScriptRequest):
    """Generate script from viral angle"""
    script = service.generate_script(request.viral_angle)
    return ScriptResponse(script=script)

@app.post("/full-pipeline", response_model=FullPipelineResponse)
async def full_pipeline(request: VideoIdeaRequest):
    """Run the complete pipeline in one call"""
    try:
        # Step 1: Extract keyword
        keyword = service.extract_keyword(request.video_idea)
        
        # Step 2: Fetch trends
        category_trends, global_trends = service.fetch_trends(keyword, request.category_id)
        
        # Step 3: Generate viral angle
        viral_angle, selected_trends = service.generate_viral_angle(
            request.video_idea, category_trends, global_trends
        )
        
        # Step 4: Generate script
        script = service.generate_script(viral_angle)
        
        return FullPipelineResponse(
            original_idea=request.video_idea,
            keyword=keyword,
            viral_angle=viral_angle,
            selected_trends=selected_trends,
            script=script,
            trends_data={
                "category_trends": category_trends,
                "global_trends": global_trends
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)