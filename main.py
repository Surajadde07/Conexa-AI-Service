from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline
import ollama
import whisper
import tempfile
import shutil
import os

app = FastAPI()

# =========================
# LOAD MODELS ON STARTUP
# =========================

model = whisper.load_model("base")

print("Whisper model loaded...")

summarizer = pipeline(
    "text2text-generation",
    model="google/flan-t5-large"
)

print("Summarizer model loaded...")

sentiment_analyzer = pipeline(
    "sentiment-analysis"
)

print("Sentiment model loaded...")



class InsightsRequest(BaseModel):
    transcript: str


# =========================
# REQUEST MODEL
# =========================

class TranscriptRequest(BaseModel):
    transcript: str

class ScoreRequest(BaseModel):
    transcript: str

# =========================
# HEALTH CHECK
# =========================

@app.get("/")
def home():
    return {
        "message": "Convexa AI FastAPI Running"
    }


# =========================
# TRANSCRIBE AUDIO
# =========================

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):

    temp_file_path = None

    try:

        suffix = os.path.splitext(file.filename)[1]

        with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
        ) as temp_file:

            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        result = model.transcribe(temp_file_path)

        transcript = result["text"].strip()

        return {
            "transcript": transcript
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

    finally:

        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/analyze")
async def analyze_call(request: TranscriptRequest):

    try:

        transcript = request.transcript[:4000]

        # =========================
        # SENTIMENT (Separate Model)
        # =========================

        sentiment_result = sentiment_analyzer(
            transcript[:512]
        )

        sentiment = sentiment_result[0]["label"]

        # =========================
        # LLM ANALYSIS
        # =========================

        prompt = f"""
You are an expert Call Center Quality Assurance Analyst.

Analyze the customer support conversation.

Return ONLY valid JSON.

Required JSON format:

{{
  "summary": "Business summary",

  "insights": "Customer Intent: ...\\nMain Issue: ...\\nCustomer Concern: ...\\nOutcome: ...\\nAgent Performance: ...",

  "overallScore": 0,
  "communication": 0,
  "problemResolution": 0,
  "professionalism": 0,
  "customerSatisfaction": 0,

  "strengths": [
    "strength 1",
    "strength 2"
  ],

  "improvements": [
    "improvement 1",
    "improvement 2"
  ]
}}

Rules:

SUMMARY:
- Mention customer issue.
- Mention agent response.
- Mention final outcome.
- Keep professional.

INSIGHTS:
Return exactly:

Customer Intent:
Main Issue:
Customer Concern:
Outcome:
Agent Performance:

QUALITY SCORES:
- Scores between 0 and 100.
- Be realistic.

STRENGTHS:
- Mention what the agent did well.

IMPROVEMENTS:
- Mention what the agent could improve.

Return ONLY valid JSON.

Transcript:

{transcript}
"""

        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = response["message"]["content"]

        content = content.replace(
            "```json",
            ""
        )

        content = content.replace(
            "```",
            ""
        )

        content = content.strip()

        import json

        try:

            result = json.loads(content)

            # Inject sentiment from dedicated model
            result["sentiment"] = sentiment

            return result

        except Exception:

            return {
                "summary": "Summary unavailable",

                "sentiment": sentiment,

                "insights": "Analysis unavailable",

                "overallScore": 75,
                "communication": 75,
                "problemResolution": 75,
                "professionalism": 75,
                "customerSatisfaction": 75,

                "strengths": [
                    "Conversation completed"
                ],

                "improvements": [
                    "Further analysis unavailable"
                ]
            }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )

# # =========================
# # GENERATE SUMMARY
# # =========================

# @app.post("/summary")
# async def generate_summary(request: TranscriptRequest):

#     try:

#         transcript = request.transcript.strip()

#         if not transcript:
#             return JSONResponse(
#                 status_code=400,
#                 content={"error": "Transcript is required"}
#             )

#         # FLAN-T5 doesn't handle huge inputs well
#         transcript = transcript[:2500]

#         prompt = f"""
# You are an AI call center analyst.

# Analyze this customer support call and provide a concise business summary.

# Return:
# Customer Issue:
# Agent Response:
# Outcome:

# Transcript:
# {transcript}
# """

#         result = summarizer(
#             prompt,
#             max_new_tokens=200,
#             do_sample=False
#         )

#         summary = result[0]["generated_text"].strip()

#         return {
#             "summary": summary
#         }

#     except Exception as e:

#         return JSONResponse(
#             status_code=500,
#             content={"error": str(e)}
#         )


# @app.post("/sentiment")
# async def analyze_sentiment(request: TranscriptRequest):

#     try:

#         transcript = request.transcript.strip()

#         if not transcript:
#             return JSONResponse(
#                 status_code=400,
#                 content={"error": "Transcript is required"}
#             )

#         transcript = transcript[:512]

#         result = sentiment_analyzer(transcript)

#         return {
#             "sentiment": result[0]["label"],
#             "confidence": round(result[0]["score"] * 100, 2)
#         }

#     except Exception as e:

#         return JSONResponse(
#             status_code=500,
#             content={"error": str(e)}
#         )

# @app.post("/insights")
# async def generate_insights(request: TranscriptRequest):

#     try:

#         transcript = request.transcript[:3000]

#         prompt = f"""
# You are an expert call center analyst.

# Analyze the conversation.

# Return in this format:

# Customer Intent:
# Main Issue:
# Customer Concern:
# Outcome:
# Agent Performance:

# Transcript:
# {transcript}
# """

#         response = ollama.chat(
#             model="qwen2.5:3b",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ]
#         )

#         insights = response["message"]["content"]

#         return {
#             "insights": insights
#         }

#     except Exception as e:

#         return JSONResponse(
#             status_code=500,
#             content={"error": str(e)}
#         )

# @app.post("/quality-score")
# async def quality_score(request: ScoreRequest):

#     try:

#         transcript = request.transcript[:3000]

#         prompt = f"""
# You are a senior Call Center QA Analyst.

# Analyze the conversation and score it.

# Rules:
# - Scores must be between 0 and 100
# - Give realistic scores
# - Do NOT return explanations outside JSON
# - Return ONLY valid JSON

# Example:

# {{
#     "overallScore": 85,
#     "communication": 88,
#     "problemResolution": 82,
#     "professionalism": 90,
#     "customerSatisfaction": 80,
#     "strengths": [
#         "Agent communicated clearly",
#         "Agent provided pricing information"
#     ],
#     "improvements": [
#         "Could confirm customer understanding",
#         "Could provide additional alternatives"
#     ]
# }}

# Conversation:

# {transcript}
# """

#         response = ollama.chat(
#             model="qwen2.5:3b",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ]
#         )

#         content = response["message"]["content"]

#         # Remove markdown code blocks if model returns them
#         content = content.replace("```json", "")
#         content = content.replace("```", "")
#         content = content.strip()

#         import json

#         try:
#             result = json.loads(content)

#         except Exception:

#             result = {
#                 "overallScore": 75,
#                 "communication": 75,
#                 "problemResolution": 75,
#                 "professionalism": 75,
#                 "customerSatisfaction": 75,
#                 "strengths": [
#                     "Conversation completed successfully"
#                 ],
#                 "improvements": [
#                     "Further analysis unavailable"
#                 ]
#             }

#         return result

#     except Exception as e:

#         return JSONResponse(
#             status_code=500,
#             content={
#                 "error": str(e)
#             }
#         )
        
@app.post("/keywords")
async def extract_keywords(request: TranscriptRequest):

    try:

        transcript = request.transcript[:3000]

        prompt = f"""
You are an expert Conversation Intelligence AI.

Analyze the conversation transcript and extract the most important business-related keywords or key phrases.

Rules:
- Return ONLY valid JSON.
- Do NOT return markdown.
- Do NOT return explanations.
- Output format:

{{
  "keywords": [
    "keyword1",
    "keyword2"
  ]
}}

- Return between 5 and 10 keywords.
- Prefer specific phrases over generic words.
- Extract only information that actually appears in the conversation.
- Do not invent keywords.

Focus on identifying:
- Products
- Services
- Customer requests
- Customer issues
- Complaints
- Promotions
- Discounts
- Pricing
- Refunds
- Replacements
- Deliveries
- Billing topics
- Subscription plans
- Technical issues
- Product names
- Vehicle names
- Company names
- Plan names
- Policy names
- Version numbers
- Important business entities

Avoid generic words such as:
- Customer
- Agent
- Call
- Support
- Help
- Issue
- Service
- Product
- Order

Do NOT include:
- Person names
- Phone numbers
- Email addresses
- Physical addresses
- ZIP codes
- Customer IDs
- Account numbers
- Credit card details
- Verification information

Good examples:
[
  "Map Update",
  "2009 Nissan Altima",
  "Version 7.7",
  "$50 Discount",
  "Premium Subscription",
  "Refund Request",
  "Delivery Delay",
  "Billing Issue",
  "Replacement Order"
]

Bad examples:
[
  "John Smith",
  "555-123-4567",
  "123 Main Street",
  "Customer Number 15243",
  "Support Agent"
]

Transcript:
{transcript}
"""

        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        import json

        keywords = json.loads(
            response["message"]["content"]
        )

        return keywords

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )