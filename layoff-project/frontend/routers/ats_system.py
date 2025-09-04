# from dotenv import load_dotenv
# import streamlit as st
# import os
# import json
# import PyPDF2 as pdf
# import google.generativeai as genai
# import json
# import re

# # ---------- Setup ----------
# load_dotenv()
# API_KEY = os.getenv("API_GOOGLE_KEY")  # keep your env var name
# if not API_KEY:
#     st.warning("Missing API_GOOGLE_KEY in your environment or Streamlit secrets.")
# genai.configure(api_key=API_KEY)


# # ---------- Helpers ----------
# def get_gemini_response(prompt: str) -> str:
#     model = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-1.5-flash" for speed
#     resp = model.generate_content(prompt)
#     return resp.text


# def input_pdf_text(uploaded_file) -> str:
#     reader = pdf.PdfReader(uploaded_file)
#     text = ""
#     # ✅ iterate over pages (your earlier error came from calling reader like a function)
#     for page in reader.pages:
#         text += page.extract_text() or ""
#     return text


# def must_inputs(uploaded_file, jd: str):
#     if not uploaded_file:
#         st.error("Please upload your resume (PDF).")
#         return False, ""
#     if not jd.strip():
#         st.error("Please paste a job description.")
#         return False, ""
#     text = input_pdf_text(uploaded_file)
#     if not text.strip():
#         st.error("Could not extract text from the PDF (maybe it’s scanned).")
#         return False, ""
#     return True, text


# def extract_json_block(s: str) -> str:
#     """Pull a JSON object out of an LLM reply (handles ```json ... ``` too)."""
#     if not s:
#         return ""
#     m = re.search(r"```json\s*(\{.*?\})\s*```", s, re.DOTALL | re.IGNORECASE)
#     if m:
#         return m.group(1)
#     start, end = s.find("{"), s.rfind("}")
#     return s[start : end + 1] if start != -1 and end != -1 and end > start else s


# # ---------- Prompts (your 3 flows) ----------
# # 1) Tell me about the resume
# INPUT_PROMPT1 = """
# You are an experienced Technical HR Manager. Review the resume against the job description and provide:
# - Whether the profile aligns with the role
# - Key strengths
# - Key weaknesses/gaps
# - Actionable improvements (brief bullets)

# RESUME:
# {text}

# JOB DESCRIPTION:
# {jd}
# """

# # 2) How can I improve my skills
# # (You can keep this separate for a different tone/focus)
# INPUT_PROMPT2 = """
# You are a skilled career coach. Given the resume and job description, provide specific, actionable suggestions to strengthen the candidate:
# - Top skills to learn or improve
# - Tools/technologies to add
# - Concrete experiences/projects to build
# - Keywords to incorporate naturally (not keyword stuffing)

# RESUME:
# {text}

# JOB DESCRIPTION:
# {jd}
# """

# # 3) Percentage match (JSON)
# INPUT_PROMPT3 = """
# You are an ATS (Applicant Tracking System). Evaluate the resume vs the job description.

# Return ONLY valid JSON (no code fences, no extra text) with exactly these keys:
# {{"JD Match":"<percent>%","MissingKeywords":["<keyword1>","<keyword2>"],"FinalThoughts":""}}

# Rules for MissingKeywords:
# - It MUST be a JSON array of strings.
# - Each item is a full keyword/phrase (e.g., "React", "Go", "Fastify", "TypeScript").
# - Do NOT return a single concatenated string. Do NOT return character arrays.

# RESUME:
# {text}

# JOB DESCRIPTION:
# {jd}
# """

# # ---------- UI ----------


# def show():
#     st.set_page_config(page_title="ATS Resume Expert")
#     st.header("ATS Tracking System")
#     st.text("Improve Your Resume for ATS")

#     jd = st.text_area("Paste the Job Description")
#     uploaded_file = st.file_uploader(
#         "Upload Your Resume (PDF)", type="pdf", help="Please upload a PDF"
#     )

#     if uploaded_file is not None:
#         st.success("PDF Uploaded Successfully")

#     col1, col2, col3 = st.columns(3)
#     submit1 = col1.button("Tell Me About the Resume")
#     submit2 = col2.button("How Can I Improve My Skills")
#     submit3 = col3.button("Percentage Match")

#     # ---------- Actions ----------
#     if submit1:
#         ok, text = must_inputs(uploaded_file, jd)
#         if ok:
#             with st.spinner("Analyzing resume..."):
#                 prompt = INPUT_PROMPT1.format(text=text, jd=jd)
#                 response = get_gemini_response(prompt)
#             st.subheader("Review")
#             st.write(response)

#     elif submit2:
#         ok, text = must_inputs(uploaded_file, jd)
#         if ok:
#             with st.spinner("Generating improvement suggestions..."):
#                 prompt = INPUT_PROMPT2.format(text=text, jd=jd)
#                 response = get_gemini_response(prompt)
#             st.subheader("Improvement Suggestions")
#             st.write(response)

#     elif submit3:
#         ok, text = must_inputs(uploaded_file, jd)
#         if ok:
#             with st.spinner("Computing ATS match..."):
#                 prompt = INPUT_PROMPT3.format(text=text, jd=jd)
#                 try:
#                     response = get_gemini_response(prompt)
#                 except Exception as e:
#                     st.error(f"Error calling Gemini: {e}")
#                     st.stop()

#             st.subheader("ATS Match")
#             try:
#                 raw = extract_json_block(response.strip())
#                 data = json.loads(raw)

#                 colA, colB = st.columns(2)
#                 with colA:
#                     st.metric("JD Match", data.get("JD Match", "N/A"))
#                 with colB:
#                     st.metric(
#                         "Missing Keywords", str(len(data.get("MissingKeywords", [])))
#                     )

#                 if data.get("MissingKeywords"):
#                     st.markdown(
#                         "**Missing Keywords:** " + ", ".join(data["MissingKeywords"])
#                     )
#                 else:
#                     st.info("No missing keywords detected.")

#                 if data.get("FinalThoughts"):
#                     st.markdown("**Final Thoughts:** " + data["FinalThoughts"])

#                 with st.expander("Raw JSON"):
#                     st.json(data)

#             except Exception:
#                 st.write(response)
# app_langgraph.py
import os
import json
from typing import Literal, TypedDict, List, Any

import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

# LangGraph
from langgraph.graph import StateGraph, START, END

# ---------------- Setup ----------------
load_dotenv()
API_KEY = os.getenv("API_GOOGLE_KEY")
if not API_KEY:
    st.warning("Missing API_GOOGLE_KEY in your environment (or Streamlit secrets).")


def make_llm(model_name: str, temperature: float):
    return ChatGoogleGenerativeAI(
        model=model_name, temperature=temperature, google_api_key=API_KEY
    )


# ---------------- PDF Helper ----------------
def read_pdf_text(uploaded_file) -> str:
    try:
        reader = PdfReader(uploaded_file)
        parts = []
        for p in reader.pages:
            t = p.extract_text() or ""
            if t.strip():
                parts.append(t)
        return "\n".join(parts).strip()
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""


# ---------------- Parsers / Schemas ----------------
class ATSMatch(BaseModel):
    JD_Match: str = Field(description='Percentage string like "82%"')
    MissingKeywords: List[str] = Field(
        description="Distinct, full keywords/phrases (e.g., React, Go, Fastify, TypeScript)."
    )
    FinalThoughts: str


match_parser = PydanticOutputParser(pydantic_object=ATSMatch)

# ---------------- Prompts ----------------
TELL_PROMPT = PromptTemplate.from_template(
    """
You are an experienced Technical HR Manager. Review the resume against the job description and provide:
- Whether the profile aligns with the role
- Key strengths
- Key weaknesses/gaps
- 3-6 actionable improvements (brief bullets)

Be concise and concrete.

RESUME:
{resume}

JOB DESCRIPTION:
{jd}
"""
)

IMPROVE_P   ROMPT = PromptTemplate.from_template(
    """
You are a skilled career coach. Given the resume and job description, provide specific, actionable suggestions to strengthen the candidate:
- Top skills to learn or improve
- Tools/technologies to add
- Concrete experiences/projects to build
- Keywords to incorporate naturally (avoid keyword stuffing)

Keep it practical and step-by-step.

RESUME:
{resume}

JOB DESCRIPTION:
{jd}
"""
)

MATCH_PROMPT = PromptTemplate(
    template="""
You are an ATS (Applicant Tracking System). Evaluate the resume vs the job description.

Return ONLY valid JSON following these rules:
- Use keys exactly as: JD_Match, MissingKeywords, FinalThoughts
- JD_Match must be a percentage string like "82%"
- MissingKeywords must be an array of distinct, full keywords/phrases (e.g., "React", "Go", "Fastify", "TypeScript").
  Do NOT output a single concatenated string. Do NOT output character arrays.
- FinalThoughts is a short paragraph.

{format_instructions}

RESUME:
{resume}

JOB DESCRIPTION:
{jd}
""",
    input_variables=["resume", "jd"],
    partial_variables={"format_instructions": match_parser.get_format_instructions()},
)


# ---------------- Chains (created per-run with current LLM) ----------------
def build_chains(llm):
    tell_chain = TELL_PROMPT | llm | StrOutputParser()
    improve_chain = IMPROVE_PROMPT | llm | StrOutputParser()
    match_chain = MATCH_PROMPT | llm | match_parser
    return tell_chain, improve_chain, match_chain


# ---------------- LangGraph State ----------------
class GraphState(TypedDict):
    resume: str
    jd: str
    task: Literal["tell", "improve", "match"]
    result: Any
    model_name: str
    temperature: float


# ---------------- LangGraph Nodes ----------------
def router(state: GraphState):
    """Route to the proper node based on 'task'."""
    return state["task"]


def tell_node(state: GraphState):
    llm = make_llm(state["model_name"], state["temperature"])
    tell_chain, _, _ = build_chains(llm)
    out = tell_chain.invoke({"resume": state["resume"], "jd": state["jd"]})
    return {"result": out}


def improve_node(state: GraphState):
    llm = make_llm(state["model_name"], state["temperature"])
    _, improve_chain, _ = build_chains(llm)
    out = improve_chain.invoke({"resume": state["resume"], "jd": state["jd"]})
    return {"result": out}


def match_node(state: GraphState):
    llm = make_llm(state["model_name"], state["temperature"])
    _, _, match_chain = build_chains(llm)
    result: ATSMatch = match_chain.invoke(
        {"resume": state["resume"], "jd": state["jd"]}
    )
    # post-process for safety (dedupe/sort)
    cleaned = sorted(
        list({(kw or "").strip(): None for kw in (result.MissingKeywords or [])}.keys())
    )
    result.MissingKeywords = [k for k in cleaned if k]
    return {"result": result}


# Build the graph
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("tell", tell_node)
    g.add_node("improve", improve_node)
    g.add_node("match", match_node)
    # Use conditional edges from START based on the router's return value
    g.add_conditional_edges(
        START,
        router,
        {
            "tell": "tell",
            "improve": "improve",
            "match": "match",
        },
    )
    g.add_edge("tell", END)
    g.add_edge("improve", END)
    g.add_edge("match", END)
    return g.compile()


# ---------------- Streamlit UI ----------------
def show():
    st.set_page_config(page_title="ATS Resume Agent (LangGraph)", page_icon="🕸️")
    st.title("🕸️ ATS Resume Agent — LangGraph + Gemini")
    st.caption("Agentic workflow that routes to the right analysis node.")

    with st.sidebar:
        st.subheader("Model Settings")
        model_choice = st.selectbox(
            "Model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

        st.markdown("---")
        st.write("**How it works**")
        st.write("The graph routes your request to one of three nodes:")
        st.write("- **tell** → HR-style review")
        st.write("- **improve** → career-coach plan")
        st.write("- **match** → ATS JSON + metrics")

    jd = st.text_area("📋 Paste the Job Description", height=220)
    uploaded = st.file_uploader("📄 Upload Your Resume (PDF)", type=["pdf"])

    if uploaded is not None:
        st.success("PDF uploaded successfully.")

    col1, col2, col3 = st.columns(3)
    b1 = col1.button("Tell Me About the Resume")
    b2 = col2.button("How Can I Improve My Skills")
    b3 = col3.button("Percentage Match")

    if b1 or b2 or b3:
        if not uploaded:
            st.error("Please upload your resume (PDF).")
            st.stop()
        if not jd or not jd.strip():
            st.error("Please paste a job description.")
            st.stop()

        resume_text = read_pdf_text(uploaded)
        if not resume_text:
            st.error("Could not extract text from the PDF (maybe it’s scanned).")
            st.stop()

        task: Literal["tell", "improve", "match"] = (
            "tell" if b1 else ("improve" if b2 else "match")
        )
        graph = build_graph()

        with st.spinner("Running agent…"):
            final_state = graph.invoke(
                {
                    "resume": resume_text,
                    "jd": jd,
                    "task": task,
                    "result": None,
                    "model_name": model_choice,
                    "temperature": temperature,
                }
            )
            res = final_state["result"]

        if task == "tell":
            st.subheader("Review")
            st.write(res)
        elif task == "improve":
            st.subheader("Improvement Suggestions")
            st.write(res)
        else:
            st.subheader("ATS Match")
            st.metric("JD Match", res.JD_Match)
            st.metric("Missing Keywords", str(len(res.MissingKeywords)))
            if res.MissingKeywords:
                st.markdown("**Missing Keywords:** " + ", ".join(res.MissingKeywords))
            else:
                st.info("No missing keywords detected.")
            if res.FinalThoughts:
                st.markdown("**Final Thoughts:** " + res.FinalThoughts)
            with st.expander("Raw JSON"):
                st.json(json.loads(res.model_dump_json()))

    st.markdown("---")
    st.caption("Built with 🕸️ LangGraph + Gemini • © Your Name")


# if __name__ == "__main__":
#     main()
