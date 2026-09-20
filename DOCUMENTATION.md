# MediFlow AI — Comprehensive System Documentation & Technical Specification

> **Project Name:** MediFlow AI  
> **Repository:** [https://github.com/TALAVIYAJAY/mediflow-ai](https://github.com/TALAVIYAJAY/mediflow-ai)  
> **Author:** Jay Talaviya  
> **Architecture:** Two-Tier Agentic Ecosystem (n8n Frontline Agent + FastAPI/LangGraph Deep Reasoning Engine)  
> **Date:** September 2026  

---

## Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [The Complete Technology Stack](#3-the-complete-technology-stack)
4. [Component Deep Dive](#4-component-deep-dive)
   - [4.1 Tier 1: Frontline Omnichannel AI Agent (n8n)](#41-tier-1-frontline-omnichannel-ai-agent-n8n)
   - [4.2 Tier 2: Cognitive Reasoning & Clinical State Machine (LangGraph)](#42-tier-2-cognitive-reasoning--clinical-state-machine-langgraph)
   - [4.3 Tier 3: Unified Data & Vector Lake (PostgreSQL + pgvector)](#43-tier-3-unified-data--vector-lake-postgresql--pgvector)
   - [4.4 Tier 4: Robotic Process Automation (Playwright RPA)](#44-tier-4-robotic-process-automation-playwright-rpa)
   - [4.5 Tier 5: Evaluation & Quality Assurance (Ragas & Pytest)](#45-tier-5-evaluation--quality-assurance-ragas--pytest)
5. [Database Architecture & Data Schemas](#5-database-architecture--data-schemas)
   - [5.1 Relational Schema (PostgreSQL)](#51-relational-schema-postgresql)
   - [5.2 Vector Schema (pgvector)](#52-vector-schema-pgvector)
   - [5.3 Pydantic Data Contracts](#53-pydantic-data-contracts)
6. [LangGraph State Machine Specification](#6-langgraph-state-machine-specification)
7. [n8n Workflow Specification](#7-n8n-workflow-specification)
8. [Automated Evaluation & Benchmarking Strategy](#8-automated-evaluation--benchmarking-strategy)
9. [Step-by-Step Implementation Roadmap](#9-step-by-step-implementation-roadmap)
10. [Local Development & Environment Setup](#10-local-development--environment-setup)

---

## 1. Executive Summary & Problem Statement

### The Problem in Modern Healthcare Operations
1. **Clinical Triage Bottlenecks:** Patients flooding hospital communication channels (WhatsApp, phone, web) often do not know whether their symptoms require emergency intervention or routine outpatient care. Poor triage leads to delayed emergency responses and overloaded doctors.
2. **Messy Diagnostic Reports:** Patients arrive with multi-page, scanned, or photographed lab test reports (blood tests, pathology, radiology). Doctors spend critical minutes manually skimming tables to spot abnormal biomarkers and historical comparisons.
3. **The Mediclaim Pre-Authorization Nightmare:** Inpatient admissions and surgeries require insurance pre-authorization. Over 30% of claims are rejected or delayed due to minor policy exclusions, missing ICD-10 codes, or failure to demonstrate medical necessity under dense 100-page policy manuals.
4. **Disjointed Systems:** Hospital Electronic Health Records (EHR), calendar schedulers, insurance web portals, and messaging apps exist in completely separate silos.

### The MediFlow AI Solution
MediFlow AI introduces an **autonomous, safety-first, multi-tier agentic ecosystem** that bridges these silos:
* **Instant Omnichannel Triage:** An n8n-powered conversational agent with active memory assesses patient symptoms and triggers life-saving emergency red flags when critical indicators appear.
* **Multi-Modal Clinical Vision:** Google Gemini 2.0 Flash parses messy lab reports into normalized, typed Pydantic models.
* **Self-Correcting Policy RAG:** PostgreSQL with `pgvector` indexes insurance coverage policies and clinical guidelines. A cyclical LangGraph agent critiques its own pre-authorization recommendations before issuing decisions.
* **Automated RPA Execution:** Playwright submits verified pre-authorization claims directly to mock insurance portals without manual data entry.

---

## 2. End-to-End System Architecture

MediFlow AI uses a **Two-Tier Agent Architecture**:
* **Tier 1 (Frontline):** An event-driven, conversational AI Agent built in **n8n** that interfaces with patients via WhatsApp, Webhooks, and Voice.
* **Tier 2 (Deep Brain):** A high-performance **FastAPI + LangGraph** microservice running clinical reasoning, multi-modal report parsing, vector search in **PostgreSQL (`pgvector`)**, and self-correction loops.

```mermaid
flowchart TD
    subgraph Client_Channels ["Omnichannel Ingestion Layer"]
        P1[Patient: Web Chat / Mobile]
        P2[Patient: WhatsApp Message / Audio]
        P3[Patient / Doctor: Uploads Lab Report PDF/Scan]
    end

    subgraph Tier_1 ["Tier 1: Frontline Conversational Agent (n8n)"]
        N8N_TRIG[n8n Webhook / Chat Trigger]
        N8N_AGENT[n8n AI Agent Node: LangChain Orchestrator]
        N8N_MEM[(n8n Window Buffer Memory)]
        N8N_MODEL[Google Gemini Chat Model: gemini-2.0-flash]
        N8N_TOOL1[Tool: Deep Clinical Audit & RAG]
        N8N_TOOL2[Tool: Calendar Schedulers]
        N8N_TOOL3[Tool: Insurance Claim Dispatcher]
        
        N8N_TRIG --> N8N_AGENT
        N8N_AGENT --- N8N_MEM
        N8N_AGENT --- N8N_MODEL
        N8N_AGENT --> N8N_TOOL1
        N8N_AGENT --> N8N_TOOL2
        N8N_AGENT --> N8N_TOOL3
    end

    subgraph Tier_2 ["Tier 2: Cognitive Reasoning Engine (FastAPI + LangGraph)"]
        FASTAPI[FastAPI Async Gateway]
        LG_ENGINE[LangGraph Clinical State Machine]
        
        subgraph LangGraph_Nodes ["LangGraph Execution Graph"]
            N_TRIAGE[1. Triage & Emergency Guardrail Node]
            N_VISION[2. Gemini Vision Lab Report Parser]
            N_RAG[3. pgvector Hybrid Policy Retrieval]
            N_DECIDE[4. Pre-Auth Decision Generator]
            N_REFLECT{5. Adversarial Self-Critique Loop}
        end
        
        FASTAPI --> LG_ENGINE
        LG_ENGINE --> N_TRIAGE --> N_VISION --> N_RAG --> N_DECIDE --> N_REFLECT
        N_REFLECT -- "Discrepancy / Hallucination Found" --> N_RAG
        N_REFLECT -- "Verified & Grounded" --> OUTPUT[Final Clinical Audit JSON]
    end

    subgraph Tier_3 ["Tier 3: Unified Data & Knowledge Store (PostgreSQL)"]
        PG_REL[(Relational Tables: Patients, Doctors, Claims)]
        PG_VEC[(pgvector Embeddings: Insurance Policies & Medical Guidelines)]
    end

    subgraph Tier_4 ["Tier 4: Robotic Process Automation & Execution"]
        RPA_BOT[Playwright Headless Browser: Insurance Portal Submission]
        GCAL[Google Calendar / Doctor Roster]
        NOTIF[WhatsApp / Email Dispatcher]
    end

    Client_Channels --> N8N_TRIG
    N8N_TOOL1 -->|HTTP POST| FASTAPI
    OUTPUT --> FASTAPI -->|Structured Payload| N8N_TOOL1
    N_RAG <--> PG_VEC
    OUTPUT --> PG_REL
    N8N_TOOL3 --> RPA_BOT
    N8N_TOOL2 --> GCAL
    N8N_AGENT --> NOTIF
```

---

## 3. The Complete Technology Stack

| Layer | Technology | Version / Model | Justification & Purpose |
| :--- | :--- | :--- | :--- |
| **Frontline Agent** | **n8n** | Self-Hosted / Latest | Visual orchestrator, webhook ingestion, chat memory management, and multi-tool routing. |
| **Agentic Framework** | **LangGraph** | `>= 0.2.0` | Stateful multi-step graph execution, cyclical self-correction, conditional branching, and human-in-the-loop support. |
| **LLM & Vision** | **Google Gemini** | `gemini-2.0-flash` | Ultra-fast, multi-modal token processing with generous free-tier limits (1,500 req/day). |
| **Embeddings** | **Google AI** | `text-embedding-004` | 768-dimensional dense semantic vector representations for clinical and policy text. |
| **Primary Database** | **PostgreSQL** | `15+` / `16+` | ACID-compliant enterprise storage for patient records, appointments, and audit logs. |
| **Vector Engine** | **`pgvector`** | `0.7+` | Native PostgreSQL extension for cosine, L2, and inner product similarity search over high-dimensional vectors. |
| **Backend API** | **FastAPI** | `>= 0.110.0` | High-concurrency asynchronous REST endpoints, automatic OpenAPI/Swagger docs, and streaming support. |
| **Data Validation** | **Pydantic** | `v2.x` | Strict type validation, zero-hallucination structured outputs, and runtime schema enforcement. |
| **Browser RPA** | **Playwright** | `>= 1.40.0` | Headless browser automation for filing claims in mock insurance web portals. |
| **AI Evaluation** | **Ragas & Pytest** | Latest | Mathematical benchmarking of RAG Faithfulness, Answer Relevancy, and Context Precision in CI/CD. |
| **Code Quality** | **Ruff & Pytest** | Latest | High-speed linting, formatting, and unit testing pipeline. |

---

## 4. Component Deep Dive

### 4.1 Tier 1: Frontline Omnichannel AI Agent (n8n)
The frontline agent acts as the primary contact point for users:
* **Conversational Memory:** Configured with the **Window Buffer Memory** node (tracking conversations via a unique `sessionId` mapped to the patient's phone number or web chat session).
* **Tone & Persona:** Empathetic, medically cautious, strictly compliant with HIPAA/telemedicine disclaimers.
* **Tool Calling Capabilities:**
  1. `triage_patient`: Assesses symptoms and returns danger-level indicators.
  2. `analyze_medical_report`: Hands off uploaded PDFs/images to the FastAPI backend.
  3. `check_insurance_preauth`: Checks whether a suggested surgery or procedure is covered under the patient's policy.
  4. `trigger_insurance_portal_rpa`: Dispatches Playwright to execute external form filing.
  5. `book_appointment`: Connects to Google Calendar or PostgreSQL appointment rosters.

### 4.2 Tier 2: Cognitive Reasoning & Clinical State Machine (LangGraph)
Complex medical decisions cannot be entrusted to a single prompt. LangGraph executes a deterministic state machine:
* **State Schema:** Carries patient symptoms, extracted biomarkers, raw document text, retrieved policy clauses, self-critique scores, and the final decision payload.
* **Emergency Guardrail (Node 1):** Scans inputs for critical medical red flags (e.g., severe chest pain radiating to arm, acute shortness of breath, sudden facial numbness). If detected, it immediately bypasses all other nodes and returns an emergency escalation payload.
* **Multi-Modal Extractor (Node 2):** Ingests PDF or image bytes, sending them to Gemini Vision with a Pydantic schema to extract structured lab metrics (e.g., Hemoglobin, Fasting Glucose, Serum Creatinine).
* **Policy RAG Retriever (Node 3):** Performs hybrid search (BM25 keyword + `pgvector` semantic similarity) over policy clauses matching the extracted diagnosis.
* **Self-Correction & Reflection Node (Node 4):** Evaluates whether the generated pre-authorization approval satisfies every retrieved policy requirement. If discrepancies are found (e.g., 2-year waiting period for pre-existing hypertension not fulfilled), it loops back with critique notes to adjust the decision.

### 4.3 Tier 3: Unified Data & Vector Lake (PostgreSQL + pgvector)
Instead of running a separate vector database (like Pinecone or ChromaDB), MediFlow AI centralizes all storage in PostgreSQL:
* **Relational Storage:** Standard relational tables for patients, medical staff, appointment bookings, and claim histories.
* **Vector Storage:** Tables with `vector(768)` columns to store chunked embeddings of insurance policy PDFs (e.g., Star Health, HDFC Ergo, Max Bupa) and clinical treatment protocols.
* **Hybrid Search Query:** Combining full-text SQL search (`tsvector`) and vector distance (`<=>` cosine operator) ensures exact matching of medical codes (ICD-10/CPT) while maintaining semantic understanding.

### 4.4 Tier 4: Robotic Process Automation (Playwright RPA)
Most insurance companies and government health schemes lack modern REST APIs. They rely on legacy web portals:
* When the LangGraph agent confirms that a claim satisfies all clinical and policy rules, it produces a standardized `InsuranceClaimPayload`.
* A background Playwright script launches a headless Chromium instance, navigates to the mock insurance portal, authenticates, fills in the patient ID, diagnosis codes, estimated costs, and attached lab reports, and captures the generated submission confirmation number.

### 4.5 Tier 5: Evaluation & Quality Assurance (Ragas & Pytest)
To guarantee production-grade reliability:
* **Faithfulness Evaluation:** Measures whether the agent's explanation is derived strictly from the retrieved policy documents (zero hallucination).
* **Context Precision:** Measures whether the retrieved clauses were relevant to the medical diagnosis.
* **CI/CD Automation:** Runs automated test cases in GitHub Actions on every `git push`.

---

## 5. Database Architecture & Data Schemas

### 5.1 Relational Schema (PostgreSQL)

```sql
-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Patients Table
CREATE TABLE patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    date_of_birth DATE NOT NULL,
    phone_number VARCHAR(20) UNIQUE NOT NULL,
    email VARCHAR(150),
    insurance_policy_number VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Doctors Table
CREATE TABLE doctors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name VARCHAR(150) NOT NULL,
    specialization VARCHAR(100) NOT NULL,
    email VARCHAR(150) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE
);

-- Appointments Table
CREATE TABLE appointments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    doctor_id UUID REFERENCES doctors(id),
    appointment_time TIMESTAMP WITH TIME ZONE NOT NULL,
    status VARCHAR(50) DEFAULT 'SCHEDULED', -- SCHEDULED, COMPLETED, CANCELLED
    triage_urgency VARCHAR(30) DEFAULT 'ROUTINE', -- ROUTINE, URGENT, EMERGENCY
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Insurance Claims / Pre-Authorizations Table
CREATE TABLE insurance_claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id),
    policy_number VARCHAR(50) NOT NULL,
    diagnosis_code VARCHAR(30) NOT NULL, -- e.g. ICD-10 K80.2
    procedure_name VARCHAR(200) NOT NULL,
    estimated_amount NUMERIC(10, 2) NOT NULL,
    status VARCHAR(50) NOT NULL, -- PRE_AUTH_APPROVED, PRE_AUTH_REJECTED, MANUAL_REVIEW
    ai_audit_notes TEXT,
    portal_reference_number VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### 5.2 Vector Schema (pgvector)

```sql
-- Insurance Policy Knowledge Base
CREATE TABLE insurance_policy_knowledge (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    insurer_name VARCHAR(100) NOT NULL,
    policy_name VARCHAR(150) NOT NULL,
    clause_title VARCHAR(200) NOT NULL,
    clause_text TEXT NOT NULL,
    category VARCHAR(100), -- SURGERY, PRE_EXISTING, EXCLUSION, MATERNITY
    embedding vector(768), -- Google text-embedding-004 produces 768 dimensions
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- HNSW Index for lightning-fast approximate nearest neighbor vector search
CREATE INDEX ON insurance_policy_knowledge 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Clinical Guidelines Knowledge Base
CREATE TABLE clinical_guidelines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    disease_category VARCHAR(100) NOT NULL,
    symptom_keywords TEXT[] NOT NULL,
    recommended_tests TEXT[] NOT NULL,
    emergency_indicators TEXT[] NOT NULL,
    guideline_summary TEXT NOT NULL,
    embedding vector(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ON clinical_guidelines 
USING hnsw (embedding vector_cosine_ops);
```

### 5.3 Pydantic Data Contracts

#### Lab Report Extraction Schema
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class BiomarkerResult(BaseModel):
    name: str = Field(description="Name of the biomarker, e.g., Hemoglobin, Fasting Blood Sugar")
    measured_value: float = Field(description="The numeric value recorded on the report")
    unit: str = Field(description="Measurement unit, e.g., mg/dL, g/dL, %")
    reference_range_low: Optional[float] = Field(None, description="Lower bound of normal range")
    reference_range_high: Optional[float] = Field(None, description="Upper bound of normal range")
    is_abnormal: bool = Field(description="True if value is outside reference boundaries")

class ClinicalReportExtraction(BaseModel):
    patient_name: Optional[str]
    report_date: Optional[str]
    lab_name: Optional[str]
    biomarkers: List[BiomarkerResult]
    primary_abnormalities: List[str]
    clinical_summary: str
```

#### Pre-Authorization Decision Schema
```python
class PreAuthDecision(BaseModel):
    is_approved: bool = Field(description="Whether the pre-authorization is approved by the AI auditor")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0")
    icd10_code: str = Field(description="Primary ICD-10 diagnosis code")
    cpt_code: Optional[str] = Field(None, description="CPT procedure code")
    covered_clauses: List[str] = Field(description="Exact policy clauses supporting coverage")
    exclusion_risks: List[str] = Field(description="Potential exclusion risks detected")
    reasoning: str = Field(description="Comprehensive clinical-legal rationale")
    requires_human_signoff: bool = Field(description="Flagged true if high-risk surgery or low confidence")
```

---

## 6. LangGraph State Machine Specification

```mermaid
stateDiagram-v2
    [*] --> IngestPayload
    IngestPayload --> EmergencyGuardrailCheck

    state EmergencyGuardrailCheck <<choice>>
    EmergencyGuardrailCheck --> TriggerEmergencyProtocol: Red Flag Detected (e.g. Chest Pain, Stroke)
    EmergencyGuardrailCheck --> MultiModalParsing: Routine / Non-Emergency

    MultiModalParsing --> HybridPolicyRetrieval: Biomarkers Extracted
    HybridPolicyRetrieval --> GenerateDraftDecision: Clauses Fetched from pgvector
    GenerateDraftDecision --> AdversarialReflection: Audit Decision against Policy

    state AdversarialReflection <<choice>>
    AdversarialReflection --> HybridPolicyRetrieval: Policy Gap / Loophole Found (Count < 2)
    AdversarialReflection --> FinalizeAuditPayload: Verified, Grounded & Consistent

    TriggerEmergencyProtocol --> [*]
    FinalizeAuditPayload --> [*]
```

### LangGraph Agent State
```python
from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    session_id: str
    patient_id: Optional[str]
    raw_symptoms: Optional[str]
    document_bytes: Optional[bytes]
    is_emergency: bool
    emergency_message: Optional[str]
    extracted_biomarkers: Optional[Dict[str, Any]]
    retrieved_policies: List[Dict[str, Any]]
    draft_decision: Optional[Dict[str, Any]]
    critique_feedback: Optional[str]
    reflection_iteration: int
    final_decision: Optional[Dict[str, Any]]
```

---

## 7. n8n Workflow Specification

The frontline n8n workflow contains the following configured nodes:
1. **Chat / Webhook Trigger:**
   * Receives incoming messages, session IDs, and uploaded media attachments (PDFs/Images).
2. **AI Agent Node:**
   * System Prompt: *"You are MediFlow's Clinical AI Assistant. You assist patients with symptom triage, lab report explanations, and insurance coverage checks. Always prioritize patient safety. Never provide final medication prescriptions. Hand off deep medical checks to your attached tools."*
3. **Window Buffer Memory Node:**
   * Context Window Length: `8` messages.
   * Session Key: `{{ $json.sessionId }}`.
4. **Google Gemini Model Node:**
   * Model: `gemini-2.0-flash`.
   * Temperature: `0.2` (Low temperature to prevent medical hallucinations).
5. **Custom HTTP Tool 1 (`clinical_audit_tool`):**
   * Endpoint: `POST http://localhost:8000/api/v1/agent/audit`
   * Sends patient details, symptoms, and report file.
6. **Custom HTTP Tool 2 (`insurance_rpa_tool`):**
   * Endpoint: `POST http://localhost:8000/api/v1/rpa/submit-claim`
   * Triggers Playwright headless claim submission.
7. **Google Calendar Tool:**
   * Dynamically books available slots for recommended doctor specializations.

---

## 8. Automated Evaluation & Benchmarking Strategy

To ensure zero hallucination, MediFlow AI integrates **Ragas** metrics into the testing framework:

```mermaid
graph LR
    SUBMIT[Test Claim Inputs] --> PIPELINE[MediFlow LangGraph Engine]
    PIPELINE --> GENERATED[Generated Pre-Auth Output]
    
    subgraph Ragas_Metrics ["Ragas Automated Evaluation"]
        M1[Faithfulness: Is response grounded in retrieved policy?]
        M2[Answer Relevance: Does it address patient condition?]
        M3[Context Recall: Were all critical exclusion clauses retrieved?]
    end
    
    GENERATED --> Ragas_Metrics
    Ragas_Metrics --> PASS_FAIL{Score >= 0.85?}
    PASS_FAIL -- Yes --> BUILD_PASS[CI/CD Build Success]
    PASS_FAIL -- No --> BUILD_FAIL[CI/CD Build Alert]
```

### Benchmark Metric Targets
* **Faithfulness Score:** $\ge 0.90$ (Ensures the model never invents insurance policy coverage).
* **Answer Relevancy:** $\ge 0.85$ (Ensures clinical explanations directly address the patient's symptoms).
* **Context Precision:** $\ge 0.85$ (Ensures the vector search retrieves the exact relevant clauses).

---

## 9. Step-by-Step Implementation Roadmap

```mermaid
gantt
    title MediFlow AI Implementation Roadmap
    dateFormat  YYYY-MM-DD
    section Phase 1: Environment & DB
    Virtual Env & Pyproject Setup     :active, p1_1, 2026-09-21, 1d
    PostgreSQL & pgvector Config       :p1_2, after p1_1, 1d
    Seed Knowledge Base & Policies    :p1_3, after p1_2, 1d
    section Phase 2: AI Core
    Gemini Multi-Modal Parser         :p2_1, after p1_3, 2d
    pgvector Hybrid Retrieval Engine  :p2_2, after p2_1, 2d
    LangGraph State Machine & Guardrail:p2_3, after p2_2, 3d
    section Phase 3: APIs & RPA
    FastAPI Endpoints & Contracts     :p3_1, after p2_3, 2d
    Playwright Insurance Portal Bot   :p3_2, after p3_1, 2d
    section Phase 4: Orchestration & Tests
    n8n Workflow & Tool Assembly      :p4_1, after p3_2, 2d
    Ragas Evaluation & Pytest Suite   :p4_2, after p4_1, 2d
    section Phase 5: Demo & Delivery
    Interactive Streamlit/Web Sandbox :p5_1, after p4_2, 2d
    GitHub CI/CD Actions & Final Docs :p5_2, after p5_1, 1d
```

---

## 10. Local Development & Environment Setup

### 10.1 Prerequisites
* Python 3.10, 3.11, or 3.12
* PostgreSQL 15+ with pgAdmin
* Node.js v18+ (for local n8n)
* Google Gemini API Key (Free tier from Google AI Studio)

### 10.2 Environment Configuration (`.env`)
```bash
# Google Gemini API
GOOGLE_API_KEY="your_free_gemini_api_key_here"

# Database Configuration (PostgreSQL + pgvector)
DATABASE_URL="postgresql://postgres:your_password@localhost:5432/mediflow_db"

# Server Settings
APP_ENV="development"
PORT=8000
HOST="0.0.0.0"

# n8n Webhook Secret (for securing communication)
N8N_WEBHOOK_SECRET="mediflow_secure_webhook_token_2026"
```

---

*This document serves as the foundational architectural contract for the MediFlow AI project.*
