from google import genai
import os
import json

# 1. Connect to Gemini
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# 2. System instruction — sets the model's role/behavior
SYSTEM_PROMPT = """You are a data extraction assistant.
You only output valid JSON, with no extra text, no markdown formatting, no explanation, no code fences."""

# 3. Few-shot examples — teaches the model the exact output format by showing examples
FEW_SHOT_PROMPT = """
Example 1:
Review: "The delivery was 3 days late and the box was crushed."
Output: {{"sentiment": "negative", "key_issues": ["late delivery", "damaged packaging"], "urgency": "medium"}}

Example 2:
Review: "Absolutely love this! Works perfectly, fast shipping."
Output: {{"sentiment": "positive", "key_issues": [], "urgency": "low"}}

Now extract from this review:
Review: "{review}"
Output:
"""


def extract_review(review_text: str) -> dict:
    prompt = FEW_SHOT_PROMPT.format(review=review_text)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={"system_instruction": SYSTEM_PROMPT}
    )

    raw_text = response.text.strip()

    # Models sometimes wrap JSON in ```json fences even when told not to.
    # Strip that off before parsing.
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        print("Model did not return valid JSON. Raw output was:")
        print(raw_text)
        return None


def load_sample_reviews(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    sample_reviews = load_sample_reviews("examples/sample_reviews.txt")

    for review in sample_reviews:
        print(f"\nReview: {review}")
        result = extract_review(review)
        print("Extracted:", result)
