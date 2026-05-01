import ast
import json
import re

import httpx

from services.config import settings


class LLMService:
    def __init__(self):
        self.provider = settings.llm_provider.lower()
        self.use_groq = self.provider == "groq" and bool(settings.groq_api_key)

    async def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        if self.use_groq:
            return await self._groq_chat(system_prompt, user_prompt, json_mode=True)
        return self._mock_json(user_prompt)

    async def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        if self.use_groq:
            return await self._groq_chat(system_prompt, user_prompt, json_mode=False)
        return self._mock_text(user_prompt)

    def heuristic_extract(self, clinical_note: str) -> dict:
        return self._build_mock_extraction(clinical_note)

    async def _groq_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool,
    ) -> str:
        payload: dict[str, object] = {
            "model": settings.groq_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                f"{settings.groq_base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if response.status_code == 400 and json_mode:
                payload.pop("response_format", None)
                response = await client.post(
                    f"{settings.groq_base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.groq_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    def _mock_json(self, user_prompt: str) -> str:
        if "possible_conditions" in user_prompt:
            extraction = self._extract_dict_from_prompt(user_prompt)
            return json.dumps(self._build_mock_recommendations(extraction))

        clinical_note = self._extract_note_from_prompt(user_prompt)
        return json.dumps(self._build_mock_extraction(clinical_note))

    def _extract_note_from_prompt(self, user_prompt: str) -> str:
        for label in ("Clinical note:", "Cleaned note:"):
            if label in user_prompt:
                return user_prompt.split(label, 1)[1].strip()
        return user_prompt

    def _extract_dict_from_prompt(self, user_prompt: str) -> dict:
        marker = "Extracted information:"
        if marker not in user_prompt:
            return {}
        raw = user_prompt.split(marker, 1)[1].strip()
        for stop_marker in ("\n\nRecommendations:", "\n\nSummary:"):
            if stop_marker in raw:
                raw = raw.split(stop_marker, 1)[0].strip()
        try:
            parsed = ast.literal_eval(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (SyntaxError, ValueError):
            return {}

    def _build_mock_extraction(self, clinical_note: str) -> dict:
        note = clinical_note.lower()
        symptoms = self._extract_symptoms(clinical_note)
        diagnoses = self._extract_diagnoses(clinical_note)
        medications = self._extract_medications(clinical_note)
        allergies = self._extract_allergies(clinical_note)
        relevant_negatives = self._extract_relevant_negatives(clinical_note)
        vital_signs = self._extract_vital_signs(clinical_note)

        if "no prior conditions" in note and not any(
            item.lower() == "no prior conditions" for item in relevant_negatives
        ):
            relevant_negatives.append("no prior conditions")

        return {
            "symptoms": symptoms,
            "diagnoses": diagnoses,
            "medications": medications,
            "vital_signs": vital_signs,
            "allergies": allergies,
            "relevant_negatives": relevant_negatives,
        }

    def _extract_symptoms(self, clinical_note: str) -> list[str]:
        note = clinical_note.lower()
        negative_phrases = [
            item.lower() for item in self._extract_relevant_negatives(clinical_note)
        ]
        symptom_catalog = [
            "abdominal pain",
            "back pain",
            "blood in stool",
            "blurred vision",
            "chest pain",
            "chills",
            "confusion",
            "constipation",
            "cough",
            "dehydration",
            "diarrhea",
            "dizziness",
            "dysuria",
            "ear pain",
            "fatigue",
            "fever",
            "headache",
            "hematuria",
            "joint pain",
            "loss of appetite",
            "loss of smell",
            "loss of taste",
            "nausea",
            "numbness",
            "palpitations",
            "rash",
            "runny nose",
            "seizure",
            "shortness of breath",
            "sore throat",
            "sweating",
            "swelling",
            "syncope",
            "vomiting",
            "weakness",
            "wheezing",
        ]
        symptoms: list[str] = []
        for symptom in symptom_catalog:
            if (
                symptom in note
                and not self._is_negated(note, symptom)
                and not any(symptom in phrase for phrase in negative_phrases)
            ):
                symptoms.append(symptom)

        complaint_patterns = [
            r"(?:reports?|reported|complains? of|c/o|presents? with|experiencing|has|having)\s+([^.;\n]+)",
            r"(?:chief complaint|cc)\s*[:\-]\s*([^.;\n]+)",
        ]
        for pattern in complaint_patterns:
            for match in re.finditer(pattern, clinical_note, re.I):
                for candidate in self._split_clinical_list(match.group(1)):
                    cleaned = self._clean_candidate(candidate)
                    if self._looks_like_symptom(cleaned) and cleaned.lower() not in symptoms:
                        symptoms.append(cleaned)

        return self._dedupe_symptoms(symptoms)[:12]

    def _extract_vital_signs(self, clinical_note: str) -> list[dict[str, str | None]]:
        vital_patterns = [
            ("temperature", r"(?:temp(?:erature)?\s*)?(\d{2,3}(?:\.\d+)?)\s*(?:degrees?|deg|[^\w\s])?\s*([fc])\b"),
            ("blood pressure", r"(?:bp|blood pressure)\s*[:\-]?\s*(\d{2,3}/\d{2,3})"),
            ("heart rate", r"(?:hr|heart rate|pulse)\s*[:\-]?\s*(\d{2,3})\s*(?:bpm)?"),
            ("respiratory rate", r"(?:rr|respiratory rate)\s*[:\-]?\s*(\d{1,2})"),
            ("oxygen saturation", r"(?:spo2|o2 sat|oxygen saturation)\s*[:\-]?\s*(\d{2,3})\s*%?"),
            ("glucose", r"(?:glucose|blood sugar|rbs|fbs)\s*[:\-]?\s*(\d{2,3})\s*(?:mg/dl)?"),
        ]
        vitals: list[dict[str, str | None]] = []
        seen: set[tuple[str, str]] = set()
        for name, pattern in vital_patterns:
            for match in re.finditer(pattern, clinical_note, re.I):
                if name == "temperature":
                    value = f"{match.group(1)} {match.group(2).upper()}"
                    unit = match.group(2).upper()
                elif name == "oxygen saturation":
                    value = f"{match.group(1)}%"
                    unit = "%"
                elif name == "heart rate":
                    value = match.group(1)
                    unit = "bpm"
                elif name == "glucose":
                    value = match.group(1)
                    unit = "mg/dL"
                else:
                    value = match.group(1)
                    unit = None
                key = (name, value)
                if key not in seen:
                    seen.add(key)
                    vitals.append(
                        {
                            "name": name,
                            "value": value,
                            "unit": unit,
                            "raw_text": match.group(0),
                        }
                    )
        return vitals

    def _extract_medications(self, clinical_note: str) -> list[str]:
        medication_patterns = [
            r"\b(?:taking|takes|on|started on|prescribed|given)\b\s+([^.;\n]+)",
            r"\b(?:medications?|meds)\b\s*[:\-]\s*([^.;\n]+)",
        ]
        medications: list[str] = []
        for pattern in medication_patterns:
            for match in re.finditer(pattern, clinical_note, re.I):
                for candidate in self._split_clinical_list(match.group(1)):
                    cleaned = self._clean_candidate(candidate)
                    if cleaned and cleaned.lower() not in {"no medications", "none"}:
                        medications.append(cleaned)
        return self._dedupe(medications)[:10]

    def _extract_diagnoses(self, clinical_note: str) -> list[str]:
        note = clinical_note.lower()
        diagnosis_patterns = [
            r"(?:diagnosed with|history of|hx of|pmh of|past medical history of)\s+([^.;\n]+)",
            r"(?:diagnosis|dx)\s*[:\-]\s*([^.;\n]+)",
        ]
        diagnoses: list[str] = []
        for pattern in diagnosis_patterns:
            for match in re.finditer(pattern, clinical_note, re.I):
                for candidate in self._split_clinical_list(match.group(1)):
                    cleaned = self._clean_candidate(candidate)
                    if cleaned and not cleaned.lower().startswith("no "):
                        diagnoses.append(cleaned)
        diagnosis_catalog = [
            "asthma",
            "copd",
            "diabetes",
            "hypertension",
            "heart failure",
            "coronary artery disease",
            "chronic kidney disease",
            "migraine",
            "anemia",
            "hypothyroidism",
            "hyperthyroidism",
            "pregnancy",
        ]
        for diagnosis in diagnosis_catalog:
            if diagnosis in note and not self._is_negated(note, diagnosis):
                diagnoses.append(diagnosis)
        return self._dedupe(diagnoses)[:10]

    def _extract_allergies(self, clinical_note: str) -> list[str]:
        allergies: list[str] = []
        for match in re.finditer(r"(?:allerg(?:y|ies)|allergic to)\s*[:\-]?\s*([^.;\n]+)", clinical_note, re.I):
            for candidate in self._split_clinical_list(match.group(1)):
                cleaned = self._clean_candidate(candidate)
                if cleaned and cleaned.lower() not in {"none", "nkda", "no known drug allergies"}:
                    allergies.append(cleaned)
        return self._dedupe(allergies)

    def _extract_relevant_negatives(self, clinical_note: str) -> list[str]:
        negatives: list[str] = []
        for match in re.finditer(r"\b(?:denies|no|without)\s+([^.;\n]+)", clinical_note, re.I):
            phrase = self._clean_candidate(match.group(0))
            if phrase:
                split_phrases = self._split_negative_phrase(phrase)
                negatives.extend(split_phrases or [phrase])
        return self._dedupe(negatives)[:10]

    def _split_negative_phrase(self, phrase: str) -> list[str]:
        match = re.match(r"^(denies|no|without)\s+(.+)$", phrase, re.I)
        if not match:
            return []
        prefix = match.group(1)
        body = match.group(2)
        parts = [
            self._clean_candidate(part)
            for part in re.split(r",|\band\b", body, flags=re.I)
        ]
        parts = [part for part in parts if part]
        if len(parts) <= 1:
            return [phrase]
        return [f"{prefix} {part}" for part in parts]

    def _build_mock_recommendations(self, extraction: dict) -> dict:
        symptoms = [str(item).lower() for item in extraction.get("symptoms", [])]
        diagnoses = [str(item).lower() for item in extraction.get("diagnoses", [])]
        symptom_text = " ".join(symptoms + diagnoses)

        possible_conditions: list[str] = []
        recommended_tests: list[str] = []
        medications: list[str] = []
        follow_ups: list[str] = []

        if any(term in symptom_text for term in ["fever", "cough", "sore throat", "runny nose", "wheezing", "shortness of breath"]):
            possible_conditions.extend([
                "viral upper respiratory infection",
                "influenza-like illness",
                "COVID-19 or other respiratory infection",
            ])
            recommended_tests.extend([
                "COVID-19 antigen or PCR test",
                "influenza test if locally circulating",
                "chest X-ray if hypoxia, chest pain, abnormal lung exam, or persistent fever",
            ])
            medications.extend(["supportive care with hydration and rest", "antipyretic if clinically appropriate"])

        if any(term in symptom_text for term in ["chest pain", "palpitations", "syncope"]):
            possible_conditions.extend(["acute coronary syndrome consideration", "arrhythmia", "anxiety or non-cardiac chest pain"])
            recommended_tests.extend(["ECG", "troponin if ACS is possible", "basic metabolic panel and CBC"])
            follow_ups.append("urgent evaluation if chest pain is active, severe, exertional, or associated with dyspnea or syncope")

        if any(term in symptom_text for term in ["abdominal pain", "vomiting", "nausea", "diarrhea", "constipation", "blood in stool"]):
            possible_conditions.extend(["gastroenteritis", "gastritis or peptic disease", "acute abdominal pathology depending on exam"])
            recommended_tests.extend(["CBC and electrolytes if dehydration or persistent symptoms", "liver enzymes and lipase for upper abdominal pain", "stool testing if severe or bloody diarrhea"])
            medications.extend(["oral rehydration if tolerated", "antiemetic or antidiarrheal only if clinically appropriate"])

        if any(term in symptom_text for term in ["headache", "dizziness", "weakness", "numbness", "confusion", "seizure", "blurred vision"]):
            possible_conditions.extend(["migraine or tension headache", "vestibular syndrome", "neurologic event requiring exclusion if focal deficits are present"])
            recommended_tests.extend(["focused neurologic exam", "blood glucose", "neuroimaging if red flags or focal neurologic findings"])
            follow_ups.append("urgent evaluation for sudden severe headache, new focal weakness, seizure, confusion, or vision loss")

        if any(term in symptom_text for term in ["dysuria", "hematuria", "urinary", "flank pain"]):
            possible_conditions.extend(["urinary tract infection", "pyelonephritis consideration", "renal stone consideration"])
            recommended_tests.extend(["urinalysis", "urine culture when indicated", "renal imaging if severe flank pain or obstruction concern"])

        if any(term in symptom_text for term in ["rash", "swelling", "itching"]):
            possible_conditions.extend(["allergic reaction", "viral exanthem", "dermatitis or cellulitis depending on exam"])
            recommended_tests.extend(["skin exam with distribution documentation", "CBC or inflammatory markers if cellulitis/systemic illness suspected"])
            follow_ups.append("urgent evaluation for facial swelling, breathing difficulty, mucosal involvement, or rapidly spreading rash")

        if not possible_conditions:
            possible_conditions.append("insufficient information for a narrow differential")
            recommended_tests.extend(["focused history and physical exam", "targeted labs or imaging based on clinician assessment"])

        follow_ups.append("clinician should review recommendations, assess red flags, allergies, contraindications, and local protocols")

        return {
            "possible_conditions": self._dedupe(possible_conditions)[:8],
            "recommended_tests": self._dedupe(recommended_tests)[:8],
            "medications": self._dedupe(medications)[:6],
            "follow_ups": self._dedupe(follow_ups)[:6],
            "safety_notes": [
                "AI recommendations require clinician review and are not a final diagnosis.",
                "Escalate care for severe, worsening, or life-threatening symptoms.",
            ],
        }

    def _split_clinical_list(self, value: str) -> list[str]:
        value = re.sub(
            r"\b(?:for|since)\s+(?:\d+\s+(?:day|days|week|weeks|month|months|year|years)|yesterday|today|last night|this morning)\b",
            "",
            value,
            flags=re.I,
        )
        return [item.strip() for item in re.split(r",|;|\band\b|\bwith\b", value, flags=re.I)]

    def _clean_candidate(self, value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip(" ,:-")
        cleaned = re.sub(r"\b(?:mild|moderate|severe)\s+", lambda m: m.group(0).strip() + " ", cleaned, flags=re.I)
        return cleaned[:80]

    def _looks_like_symptom(self, value: str) -> bool:
        lowered = value.lower()
        blocked = ["temperature", "bp", "blood pressure", "medication", "allergy", "diagnosed", "history", "prior condition"]
        if not lowered or any(term in lowered for term in blocked):
            return False
        return 1 <= len(lowered.split()) <= 7

    def _is_negated(self, note: str, term: str) -> bool:
        index = note.find(term)
        if index < 0:
            return False
        prefix = note[max(0, index - 24):index]
        return bool(re.search(r"\b(no|denies|without|negative for)\s+$", prefix))

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            key = value.lower().strip()
            if key and key not in seen:
                seen.add(key)
                result.append(value)
        return result

    def _dedupe_symptoms(self, symptoms: list[str]) -> list[str]:
        deduped = self._dedupe(symptoms)
        result: list[str] = []
        for symptom in deduped:
            lowered = symptom.lower()
            has_more_specific = any(
                lowered != other.lower()
                and lowered in other.lower().split()
                and len(other.split()) > len(symptom.split())
                for other in deduped
            )
            if not has_more_specific:
                result.append(symptom)
        return result

    def _mock_text(self, user_prompt: str) -> str:
        extraction = self._extract_dict_from_prompt(user_prompt)
        symptoms = extraction.get("symptoms", []) if extraction else []
        vitals = extraction.get("vital_signs", []) if extraction else []
        negatives = extraction.get("relevant_negatives", []) if extraction else []
        symptom_text = ", ".join(symptoms) if symptoms else "the documented symptoms"
        vital_text = ", ".join(
            f"{item.get('name')}: {item.get('value')}"
            for item in vitals
            if isinstance(item, dict)
        )
        negative_text = ", ".join(negatives) if negatives else "none documented"

        if "Generate a structured readable clinical report" in user_prompt:
            recommendations = self._extract_recommendations_from_prompt(user_prompt)
            conditions = ", ".join(recommendations.get("possible_conditions", [])) or "requires clinician assessment"
            tests = "\n".join(f"- {item}" for item in recommendations.get("recommended_tests", [])) or "- Targeted testing based on exam"
            meds = "\n".join(f"- {item}" for item in recommendations.get("medications", [])) or "- Medication plan requires clinician review"
            follow_ups = "\n".join(f"- {item}" for item in recommendations.get("follow_ups", [])) or "- Follow up based on symptom severity"
            return (
                "Clinical Workflow Assistant Report\n\n"
                "Summary\n"
                f"Patient presents with {symptom_text}. "
                f"Vitals documented: {vital_text or 'none documented'}. "
                f"Relevant negatives: {negative_text}.\n\n"
                "Extracted Information\n"
                f"- Symptoms: {symptom_text}\n"
                f"- Vital signs: {vital_text or 'none documented'}\n"
                f"- Relevant negatives: {negative_text}\n\n"
                "Assessment Considerations\n"
                f"- {conditions}\n\n"
                "Recommended Next Steps\n"
                f"{tests}\n"
                f"{meds}\n"
                f"{follow_ups}\n\n"
                "Safety Notes\n"
                "Recommendations require licensed clinician review and do not replace diagnosis or local care protocols."
            )
        return (
            f"Patient presents with {symptom_text}. "
            f"Vitals documented: {vital_text or 'none documented'}. "
            f"Relevant negatives: {negative_text}."
        )

    def _extract_recommendations_from_prompt(self, user_prompt: str) -> dict:
        marker = "Recommendations:"
        if marker not in user_prompt:
            return {}
        raw = user_prompt.split(marker, 1)[1].strip()
        try:
            parsed = ast.literal_eval(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (SyntaxError, ValueError):
            return {}
