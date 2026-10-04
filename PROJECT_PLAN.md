# Lecture Drift Detector: Project Plan

## 1. Problem Statement
Across a semester-long course, instructors frequently explain the same concept multiple times, often in increasing depth. A concept introduced as a simplification in an early lecture is commonly refined, extended, or occasionally contradicted in later lectures as the course builds toward a fuller treatment of the topic.

Existing lecture-search and lecture-Q&A tools treat every lecture as an independent, isolated document. They retrieve the single most relevant chunk of text and answer from it, with no awareness of where that chunk sits in the course’s conceptual timeline. This means a student can be given an early, simplified, or later-revised explanation with no indication that the idea evolved.

**This project builds a system that tracks how the explanation of each concept changes across an entire lecture series, classifies the nature of that change, and visualizes it as a per-concept timeline — surfacing refinements, corrections, and genuine contradictions that a single-document RAG system would miss entirely.**

Critically, a large share of the meaning in a lecture is visual, not spoken — diagrams, equations, board work, and slide content. This project extracts visual content from the lecture video, using scene-change detection to capture frames only when the visual content actually changes, and fuses this with the spoken transcript so that "drift" can be detected even when it happens visually rather than verbally.

## 2. What the System Does
1. Transcribes a series of lectures from audio.
2. Segments each transcript into topically coherent claim-level chunks.
3. Detects visual scene changes in the lecture video and extracts a frame at each change point (not a fixed time interval).
4. Uses a vision-language model to describe the visual content of each captured frame (slide text, diagrams, equations, board work) and aligns it with the corresponding transcript segment by timestamp.
5. Tags each (now multimodal) segment with the concept(s) it discusses.
6. Clusters segments that discuss the same concept across different lectures, even when worded differently.
7. Classifies the relationship between consecutive segments in a concept cluster as Consistent, Refinement, Correction, or Contradiction.
8. Renders this as a timeline per concept, so a student can see how an idea evolved across the course — including cases where the visual explanation changed even if the spoken words were similar.
9. Validates the whole pipeline against a hand-labeled evaluation set, reporting precision/recall/F1 per relationship class.

## 3. System Pipeline

**Step 1 — Speech-to-Text (ASR)**
Lecture audio is transcribed into timestamped text.

**Step 2 — Topic-Aware Segmentation**
Transcripts are split at points where consecutive sentence embeddings show a significant drop in similarity (a proxy for topic shift), combined with pause-length cues from ASR timestamps.

**Step 3 — Visual Scene-Change Detection**
The lecture video is scanned frame-by-frame and consecutive frames are compared for meaningful visual difference. A frame is only captured as a “keyframe” when the visual content actually changes by more than a set threshold.

**Step 4 — Visual Content Extraction**
Each captured keyframe is passed to a vision-language model, which produces a semantic description of its content (understanding of diagrams, equations, and layout).

**Step 5 — Multimodal Alignment**
Each keyframe’s timestamp is matched to the nearest overlapping transcript segment, and the visual description is appended to that segment. 

**Step 6 — Concept Tagging**
Each multimodal segment is passed to an LLM to extract the 1–3 core concepts it discusses.

**Step 7 — Cross-Lecture Concept Clustering**
Segments discussing the same underlying concept are grouped into a single cluster using embedding similarity and density-based clustering.

**Step 8 — Pairwise Relationship Classification**
Within each concept cluster, consecutive segments are compared by an LLM using a few-shot prompt and classified into: Consistent, Refinement, Correction, or Contradiction.

**Step 9 — Timeline Construction**
For each concept, the classified sequence of segments is arranged chronologically into a visual timeline showing how the explanation evolved.

**Step 10 — Evaluation**
A hand-labeled benchmark of segment pairs is used to compute precision, recall, and F1 per relationship class.

## 4. Technology Stack
* **ASR (speech-to-text):** Whisper / faster-whisper (Runs locally, no API cost, built for long-form audio)
* **Visual scene-change detection:** OpenCV + scikit-image (SSIM / frame-diffing), ~1 fps sampling (Avoids redundant frames, cheap)
* **Visual content understanding:** Vision-language model (GPT-4o-mini vision / Claude Haiku vision) (Required for diagrams/flowcharts, cheaper variant for volume)
* **Multimodal alignment:** Timestamp matching (Python)
* **Embeddings:** sentence-transformers (all-MiniLM-L6-v2) (Fast, local, "good enough" for similarity tasks)
* **Segmentation:** Cosine similarity + pause-length cues
* **Clustering:** HDBSCAN (Handles variable number of clusters per course and noise gracefully)
* **Concept tagging & classification:** LLM API (GPT-4o-mini / Claude Haiku), few-shot prompting
* **Storage:** SQLite (Relational querying without server setup)
* **Backend:** Python (FastAPI optional)
* **Frontend:** Streamlit → React + timeline library (stretch) (Streamlit for fast pipeline validation)
* **Evaluation:** scikit-learn (classification_report, confusion matrix), pandas, matplotlib

## 5. Evaluation Methodology
1. Manually label ~100 segment pairs drawn from real lecture transcripts.
2. Run the automated pipeline on the same pairs.
3. Compare pipeline output to ground-truth labels using precision, recall, and F1 per class, plus a confusion matrix.
4. Report results honestly, including expected weaknesses in rare classes (e.g., Contradiction).

## 6. Realistic Build Timeline (~3 Weeks)
* **Week 1:**
  * Whisper transcription + topic-shift segmentation on 3–4 sample lectures.
  * Scene-change detection on corresponding videos; extract and review keyframes.
* **Week 1–2:**
  * Vision-language model description of keyframes; align with transcript segments.
* **Week 2:**
  * Concept tagging + cross-lecture clustering on multimodal segments.
  * Pairwise relationship classification via LLM few-shot prompting; prompt iteration.
* **Week 2–3:**
  * Hand-label evaluation set; compute metrics; iterate on weak points.
* **Week 3:**
  * Build timeline visualization with keyframe thumbnails (Streamlit); write up results.

## 7. Key Talking Points for Interviews
* **Why “drift” and not just “contradiction”:** Most changes are refinements. A spectrum (Consistent → Refinement → Correction → Contradiction) is more accurate.
* **Why clustering before comparison:** Comparing every segment to every other segment is combinatorially expensive. Clustering by concept first reduces the comparison space.
* **Why scene-change detection over fixed-interval screenshots:** Capturing on a fixed clock misses short-lived slides and wastes compute on redundant static slides.
* **Why a VLM over OCR for visual content:** Lecture visuals are often diagrammatic where meaning depends on spatial structure, not just text.
