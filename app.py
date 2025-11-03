from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import json
import re
import os

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# === Pydantic Schema ===
class LessonFeedback(BaseModel):
    topic: str
    numerical_grade: int
    letter_grade: str
    score_content: int
    score_organization: int
    score_mechanics: int
    calculation: str
    strengths: list[str]
    weaknesses: list[str]
    improvement_suggestions: list[str]
    mechanics_issues: list[str]

# === LLM setup ===
model = "gemini-2.5-flash"
llm = ChatGoogleGenerativeAI(model=model)
parser = PydanticOutputParser(pydantic_object=LessonFeedback)

# === Prompt with strict JSON enforcement ===
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert educator and evaluator. Evaluate a lesson explanation.

Rubric (scores 0-100 before weighting):
- Content: 80%
- Organization: 15%
- Mechanics: 5% (spelling/grammar)

Requirements:
1. Provide ONLY a JSON object with the following keys exactly:
   topic, numerical_grade, letter_grade, score_content, score_organization, 
   score_mechanics, calculation, strengths, weaknesses, improvement_suggestions, mechanics_issues
2. Numeric grade = round(0.8*score_content + 0.15*score_organization + 0.05*score_mechanics)
3. Include a short 'calculation' string showing weighted calculation
4. Up to 5 mechanics issues (spelling/grammar), empty list if none
5. Emphasize content: strengths, weaknesses, and at least 3 improvement suggestions must focus on conceptual clarity and organization

ALWAYS respond with VALID JSON only. Do NOT include explanations, markdown fences, or extra text.
"""),
    ("human", "{query}")
]).partial(format_instructions=parser.get_format_instructions())

# === Adapter function to map unexpected keys ===
def adapt_feedback(raw_dict):
    """Map model output to the expected LessonFeedback schema."""
    return {
        "topic": raw_dict.get("topic", "Unknown Topic"),
        "numerical_grade": raw_dict.get("numerical_grade") or raw_dict.get("overall_score", 0),
        "letter_grade": raw_dict.get("letter_grade", "N/A"),
        "score_content": raw_dict.get("score_content") or raw_dict.get("content_score", 0),
        "score_organization": raw_dict.get("score_organization") or raw_dict.get("organization_score", 0),
        "score_mechanics": raw_dict.get("score_mechanics") or raw_dict.get("mechanics_score", 0),
        "calculation": raw_dict.get("calculation") or raw_dict.get("score_breakdown", ""),
        "strengths": raw_dict.get("strengths", []),
        "weaknesses": raw_dict.get("weaknesses", []),
        "improvement_suggestions": raw_dict.get("improvement_suggestions", []),
        "mechanics_issues": raw_dict.get("mechanics_issues", []),
    }

# === Helper function to extract JSON from raw LLM response ===
def extract_json(text: str) -> str:
    """Extract JSON object from text, even if wrapped in markdown fences."""
    text = text.strip()
    # regex to find {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else "{}"

# === Routes ===
@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/evaluate", response_class=HTMLResponse)
async def evaluate_lesson(request: Request, topic: str = Form(...), explanation: str = Form(...)):
    if not explanation.strip():
        return templates.TemplateResponse("index.html", {"request": request, "error": "Explanation cannot be empty."})

    query = f"Topic: {topic}\n\nUser's explanation:\n{explanation}\n\nPlease evaluate according to the rubric."
    raw_response = llm.invoke(prompt.format_messages(query=query))

    # --- SAFELY EXTRACT RAW TEXT ---
    raw_text = getattr(raw_response, "content", raw_response)
    if not raw_text or not raw_text.strip():
        return templates.TemplateResponse("index.html", {"request": request, "error": "Empty response from LLM."})

    # --- Extract JSON safely ---
    json_text = extract_json(raw_text)

    # --- Parse JSON safely ---
    try:
        raw_dict = json.loads(json_text)
    except json.JSONDecodeError:
        raw_dict = {}

    # --- Adapt to schema ---
    adapted_dict = adapt_feedback(raw_dict)
    feedback = LessonFeedback(**adapted_dict)

    return templates.TemplateResponse("index.html", {"request": request, "feedback": feedback})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
