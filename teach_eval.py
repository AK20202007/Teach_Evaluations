from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Load environment variables (for API keys)
load_dotenv()

# Select model
model = "gemini-2.5-flash"

# Structured output expected from the model
class LessonFeedback(BaseModel):
    topic: str
    numerical_grade: int
    letter_grade: str
    # component breakdown: values 0-100 for each component before weighting
    score_content: int
    score_organization: int
    score_mechanics: int
    calculation: str  # short text showing weighted calculation
    strengths: list[str]
    weaknesses: list[str]
    improvement_suggestions: list[str]
    mechanics_issues: list[str]  # up to 5 specific spelling/grammar issues

# Initialize LLM and parser
llm = ChatGoogleGenerativeAI(model=model)
parser = PydanticOutputParser(pydantic_object=LessonFeedback)

# Prompt: clearly state rubric and require calculation + examples of mechanics issues
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert educator and evaluator. The user will provide a lesson explanation on a topic.

Evaluate using this component rubric (all component scores are expressed on a 0-100 scale *before* weighting):
- Content (weight 80%): accuracy, depth of understanding, meaningful examples, correct use of concepts.
- Organization (weight 15%): logical flow, structure, clear progression, signposting.
- Mechanics (weight 5%): spelling, grammar, punctuation. Mechanics should only slightly influence the final grade.
  - Mechanics deductions must be small: mechanics can change the final numeric grade by at most +/- 7 points total.
  - If mechanics seriously interfere with comprehension, note that explicitly and it may justify a larger deduction, but explain why.

Requirements for your response:
1. Provide all fields exactly in the Pydantic schema that follows format_instructions. Do not output any other text.
2. For numeric grade: compute a weighted total = round(0.80*score_content + 0.15*score_organization + 0.05*score_mechanics).
3. Include a short 'calculation' string showing the three component scores and the weighted computation (example: "0.8*86 + 0.15*78 + 0.05*92 = 85").
4. Provide up to 5 specific mechanics issues (spelling/grammar) with short excerpts or corrections in 'mechanics_issues'. If none, return an empty list.
5. Emphasize content: if two explanations only differ by minor spelling mistakes, the numerical grade should primarily reflect content differences and should not be identical only because the model ignored mechanics — your calculation must show the small mechanics effect.
6. Return strengths, weaknesses, and at least 3 concrete improvement suggestions focused on conceptual clarity and organization (not only grammar).

Respond only in the structured format that the parser expects.
            """,
        ),
        ("human", "{query}")
    ]
).partial(format_instructions=parser.get_format_instructions())

# Interactive section
def get_explanation_from_user():
    print("Welcome to the content-focused Lesson Evaluator.")
    topic = input("Enter the topic you'll explain: ").strip()
    print("\nBegin explaining your topic. Type 'DONE' on a new line when you are finished.\n")
    lines = []
    while True:
        try:
            text = input()
        except EOFError:
            break
        if text.strip().lower() == "done":
            break
        lines.append(text)
    explanation = "\n".join(lines)
    return topic, explanation

def grade_explanation(topic: str, explanation: str):
    query = f"Topic: {topic}\n\nUser's explanation:\n{explanation}\n\nPlease evaluate according to the rubric."
    # invoke the model with the prompt
    raw_response = llm.invoke(prompt.format_messages(query=query))
    return raw_response

def display_feedback(raw_response):
    try:
        # raw_response.content is expected to contain the model's text; adjust if different in your SDK
        feedback = parser.parse(raw_response.content)
        print("===== Lesson Evaluation =====")
        print(f"Topic: {feedback.topic}")
        print(f"Numerical Grade: {feedback.numerical_grade}/100")
        print(f"Letter Grade: {feedback.letter_grade}")
        print(f"Calculation: {feedback.calculation}")
        print("\nComponent scores (0-100 before weighting):")
        print(f" - Content: {feedback.score_content}")
        print(f" - Organization: {feedback.score_organization}")
        print(f" - Mechanics: {feedback.score_mechanics}")
        print("\nStrengths:")
        for s in feedback.strengths:
            print(f" - {s}")
        print("\nWeaknesses:")
        for w in feedback.weaknesses:
            print(f" - {w}")
        print("\nSuggestions for Improvement (conceptual/structural):")
        for i in feedback.improvement_suggestions:
            print(f" - {i}")
        print("\nSpecific mechanics issues (up to 5):")
        if feedback.mechanics_issues:
            for m in feedback.mechanics_issues:
                print(f" - {m}")
        else:
            print(" - None noted")
        print("===============================")
    except Exception as e:
        print("Error parsing response:", e)
        # helpful debug output
        try:
            print("Raw response content:", raw_response.content)
        except Exception:
            print("Raw response:", raw_response)

def main():
    topic, explanation = get_explanation_from_user()
    if not explanation.strip():
        print("No explanation provided. Exiting.")
        return
    raw = grade_explanation(topic, explanation)
    display_feedback(raw)

if __name__ == "__main__":
    main()
