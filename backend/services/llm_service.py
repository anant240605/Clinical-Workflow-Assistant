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

    def heuristic_recommendations(self, extraction: dict) -> dict:
        return self._build_mock_recommendations(extraction)

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
            "ankle pain",
            "back pain",
            "blood in stool",
            "blurred vision",
            "body ache",
            "body aches",
            "burning urination",
            "chest pain",
            "chills",
            "cold",
            "confusion",
            "constipation",
            "cough",
            "dehydration",
            "diarrhea",
            "difficulty in understanding commands",
            "difficulty understanding commands",
            "dyspnea",
            "dyspnea on exertion",
            "dizziness",
            "dysuria",
            "ear pain",
            "difficulty bearing weight",
            "facial droop",
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
            "pain",
            "palpitations",
            "right-sided motor weakness",
            "right-sided weakness",
            "rash",
            "rhinorrhea",
            "runny nose",
            "seizure",
            "shortness of breath",
            "slurred speech",
            "sore throat",
            "sweating",
            "swelling",
            "syncope",
            "tenderness",
            "twisting injury",
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
            r"(?:reports?|reported|complains? of|c/o|presents? with|presenting with|experiencing|has|having)\s+([^.;\n]+)",
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
            r"\bon\s+(anticoagulant therapy|warfarin|apixaban|rivaroxaban|dabigatran|edoxaban|heparin|enoxaparin)\b",
            r"\b(?:taking|takes|started on|start|continue|prescribed|prescribe|given|administered|advised)\b\s+([^.;\n]+)",
            r"\b(?:medications?|meds|rx|prescription|treatment|plan)\b\s*[:\-]\s*([^.;\n]+)",
            r"\b(?:tab|tablet|cap|capsule|syrup|inj|injection|cream|ointment|drops|spray)\.?\s+([^.;\n]+)",
        ]
        medications: list[str] = []
        for pattern in medication_patterns:
            for match in re.finditer(pattern, clinical_note, re.I):
                prefix = clinical_note[max(0, match.start() - 16):match.start()].lower()
                if re.search(r"\b(no|denies|without)\s+$", prefix):
                    continue
                medication_text = match.group(0) if pattern.startswith(r"\b(?:tab") else match.group(1)
                for candidate in self._split_medication_list(medication_text):
                    cleaned = self._clean_candidate(candidate)
                    if self._looks_like_medication(cleaned):
                        medications.append(cleaned)
        medications.extend(self._extract_catalog_medications(clinical_note, medications))
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
                prefix = clinical_note[max(0, match.start() - 16):match.start()].lower()
                if re.search(r"\b(no|denies|without)\s+$", prefix):
                    continue
                for candidate in self._split_clinical_list(match.group(1)):
                    cleaned = self._clean_candidate(candidate)
                    if cleaned and not cleaned.lower().startswith("no "):
                        diagnoses.append(cleaned)
        diagnosis_catalog = [
            "atrial fibrillation",
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
        current_medications = [str(item) for item in extraction.get("medications", [])]
        allergies = [str(item).lower() for item in extraction.get("allergies", [])]
        symptom_text = " ".join(symptoms + diagnoses)

        clinical_impression: list[str] = []
        possible_conditions: list[str] = []
        recommended_tests: list[str] = []
        medications: list[str] = []
        medication_cautions: list[str] = [
            "Verify age, pregnancy status, weight, renal function, liver disease, allergies, current medicines, and local formulary before giving any medication.",
        ]
        red_flags: list[str] = []
        missing_information: list[str] = [
            "Age, sex, pregnancy status when relevant, symptom duration, severity, onset, comorbidities, current medications, allergies, and focused exam findings.",
        ]
        follow_ups: list[str] = []

        has_stroke_features = any(
            term in symptom_text
            for term in [
                "right-sided weakness",
                "right-sided motor weakness",
                "facial droop",
                "slurred speech",
                "difficulty understanding commands",
            ]
        ) or (
            any(term in symptom_text for term in ["weakness", "numbness"])
            and any(term in symptom_text for term in ["slurred speech", "facial droop", "difficulty understanding commands"])
        )

        if has_stroke_features:
            clinical_impression.append("Acute focal neurologic deficit documented; treat as suspected acute stroke/TIA until proven otherwise, especially with facial droop, unilateral weakness, speech/language difficulty, hypertension, and atrial fibrillation.")
            possible_conditions.extend([
                "acute ischemic stroke, including cardioembolic stroke related to atrial fibrillation",
                "intracranial hemorrhage requiring urgent exclusion",
                "transient ischemic attack if symptoms resolve",
                "stroke mimics such as hypoglycemia, seizure/postictal deficit, migraine, infection, or metabolic disturbance",
            ])
            recommended_tests.extend([
                "activate emergency stroke pathway/EMS transfer to a stroke-capable emergency department immediately",
                "establish last known well time and symptom onset time; do not rely only on time found or 'since morning'",
                "check point-of-care blood glucose immediately",
                "urgent non-contrast CT head or MRI brain to exclude hemorrhage before antithrombotic or thrombolytic decisions",
                "CTA head/neck or MRA if large vessel occlusion is suspected and thrombectomy eligibility needs assessment",
                "ECG/telemetry for atrial fibrillation, CBC, platelets, PT/INR, aPTT, electrolytes, renal function, and troponin per stroke protocol",
                "formal neurologic severity score such as NIHSS and swallow screen before oral intake",
            ])
            medications.extend([
                "do not give outpatient pain, vertigo, antihypertensive, antiplatelet, anticoagulant, or sedating medication before emergency stroke evaluation unless directed by the stroke team",
                "alteplase or tenecteplase may be considered only by a stroke team if ischemic stroke is confirmed, treatment window criteria are met, BP and bleeding risk are acceptable, and anticoagulant status/coagulation tests allow it",
                "aspirin is generally considered only after intracranial hemorrhage is excluded and thrombolysis decisions are complete, per clinician/stroke protocol",
                "anticoagulant reversal may be needed urgently if intracranial hemorrhage is present or anticoagulant effect is clinically significant; agent depends on the anticoagulant used",
            ])
            medication_cautions.extend([
                "Uncertain anticoagulant compliance is critical: identify warfarin vs DOAC, last dose time, INR/anti-Xa or relevant coagulation status where available.",
                "Avoid rapid BP lowering unless directed by stroke protocol; BP targets differ for thrombolysis candidates, non-thrombolysis ischemic stroke, and hemorrhagic stroke.",
                "Do not start or resume anticoagulation during the initial emergency evaluation without neuroimaging and stroke specialist input.",
            ])
            red_flags.append("Sudden unilateral weakness, facial droop, slurred speech, aphasia/confusion, severe headache, reduced consciousness, seizure, very high BP, anticoagulant use, or worsening deficits are emergency stroke red flags.")
            missing_information.append("Last known well time, exact onset time, current NIHSS/severity, glucose, anticoagulant name/dose/last dose, INR or relevant anticoagulant assay, CT/MRI result, CTA/LVO status, baseline function, and goals of care.")
            follow_ups.append("do not manage as routine clinic follow-up; arrange immediate emergency transfer/stroke-team evaluation")

        respiratory_terms = ["cough", "sore throat", "runny nose", "rhinorrhea", "wheezing", "shortness of breath", "dyspnea", "cold"]
        has_respiratory_symptoms = any(term in symptom_text for term in respiratory_terms)
        if has_respiratory_symptoms:
            clinical_impression.append("Respiratory/ENT symptom cluster documented; distinguish viral URI, influenza/COVID-19, pneumonia, asthma/COPD flare, and bacterial pharyngitis/sinusitis using vitals and exam.")
            possible_conditions.extend([
                "viral upper respiratory infection",
                "influenza-like illness",
                "COVID-19 or other respiratory infection",
                "pneumonia if abnormal vitals, focal lung findings, hypoxia, pleuritic pain, or high-risk patient",
            ])
            recommended_tests.extend([
                "COVID-19 antigen or PCR test",
                "influenza test if locally circulating",
                "chest X-ray if hypoxia, chest pain, abnormal lung exam, or persistent fever",
                "rapid strep test if sore throat with Centor/McIsaac features",
            ])
            medications.extend([
                "acetaminophen/paracetamol for fever or pain if clinically appropriate",
                "ibuprofen or naproxen for fever, myalgia, or throat pain if no renal disease, GI bleeding risk, anticoagulation, severe heart failure, NSAID allergy, or pregnancy concern",
                "dextromethorphan for troublesome dry cough in appropriate patients",
                "guaifenesin for thick productive cough if hydration is adequate",
                "pseudoephedrine or saline nasal spray for nasal congestion; avoid pseudoephedrine in uncontrolled hypertension or significant arrhythmia risk",
            ])
            medication_cautions.extend([
                "Do not use antibiotics for uncomplicated viral cold or acute bronchitis.",
                "Use amoxicillin or penicillin V only for confirmed group A strep pharyngitis after clinician assessment; use alternatives only after allergy review.",
            ])
            red_flags.extend([
                "Shortness of breath at rest, oxygen saturation below expected range, chest pain, cyanosis, confusion, severe dehydration, persistent high fever, or immunocompromised status.",
            ])
            missing_information.append("Respiratory exam, oxygen saturation, temperature, exposure history, vaccination status, sputum, chest pain, wheeze, and high-risk conditions.")

        if "fever" in symptom_text and not has_respiratory_symptoms:
            clinical_impression.append("Fever is documented without a clear source; look for respiratory, urinary, abdominal, skin/soft tissue, neurologic, medication, and travel-related causes.")
            possible_conditions.append("undifferentiated febrile illness requiring source assessment")
            recommended_tests.extend(["confirm temperature trend", "targeted infection workup based on history and exam"])
            medications.append("acetaminophen/paracetamol for fever if clinically appropriate")
            red_flags.append("Persistent high fever, rigors, hypotension, confusion, neck stiffness, non-blanching rash, dehydration, or sepsis concern.")
            missing_information.append("Temperature value, duration, travel/exposure history, localizing symptoms, immune status, and vital signs.")

        if any(term in symptom_text for term in ["chest pain", "palpitations", "syncope"]):
            clinical_impression.append("Cardiopulmonary risk symptoms documented; urgent triage is needed before outpatient medication decisions.")
            possible_conditions.extend(["acute coronary syndrome consideration", "arrhythmia", "anxiety or non-cardiac chest pain"])
            recommended_tests.extend(["ECG", "troponin if ACS is possible", "basic metabolic panel and CBC"])
            medications.append("avoid empiric outpatient pain-only treatment for concerning chest pain until ECG/ACS evaluation is complete")
            red_flags.append("Active, severe, exertional, crushing, or radiating chest pain; syncope; diaphoresis; dyspnea; abnormal vitals; or known cardiac disease.")
            missing_information.append("Pain character, exertional component, radiation, associated dyspnea/diaphoresis/nausea, cardiac risk factors, ECG, and vitals.")
            follow_ups.append("urgent evaluation if chest pain is active, severe, exertional, or associated with dyspnea or syncope")

        if any(term in symptom_text for term in ["abdominal pain", "vomiting", "nausea", "diarrhea", "constipation", "blood in stool"]):
            clinical_impression.append("Gastrointestinal symptom cluster documented; assess hydration, abdominal exam, stool features, pregnancy possibility, and surgical abdomen signs.")
            possible_conditions.extend(["gastroenteritis", "gastritis or peptic disease", "acute abdominal pathology depending on exam"])
            recommended_tests.extend(["CBC and electrolytes if dehydration or persistent symptoms", "liver enzymes and lipase for upper abdominal pain", "stool testing if severe or bloody diarrhea"])
            medications.extend([
                "oral rehydration solution (ORS) for vomiting or diarrhea if tolerated",
                "ondansetron for significant nausea/vomiting if clinician confirms suitability and QT/drug-interaction risk is acceptable",
                "loperamide for non-bloody diarrhea only when fever, dysentery, and suspected invasive infection are absent",
                "omeprazole or pantoprazole for reflux/gastritis-type symptoms if clinically appropriate",
            ])
            medication_cautions.append("Avoid loperamide with bloody diarrhea, high fever, suspected C. difficile, or severe colitis.")
            red_flags.append("Severe localized abdominal pain, guarding, persistent vomiting, bloody stool, black stool, dehydration, pregnancy, syncope, or hypotension.")
            missing_information.append("Pain location, peritoneal signs, stool blood, hydration status, pregnancy possibility, medication/NSAID use, travel, and food exposure.")

        has_ankle_trauma = any(term in symptom_text for term in ["ankle pain", "twisting injury", "difficulty bearing weight", "lateral malleolus"]) or (
            "ankle" in symptom_text and any(term in symptom_text for term in ["pain", "swelling", "tenderness"])
        )

        if any(term in symptom_text for term in ["pain", "body ache", "body aches", "headache", "back pain", "joint pain"]) and not has_ankle_trauma:
            medications.extend([
                "acetaminophen/paracetamol for pain or fever if no severe liver disease or overdose risk",
                "ibuprofen or naproxen for inflammatory pain if no renal disease, ulcer/GI bleed risk, anticoagulant use, severe heart failure, NSAID allergy, or pregnancy concern",
            ])

        if any(term in symptom_text for term in ["headache", "dizziness", "weakness", "numbness", "confusion", "seizure", "blurred vision"]) and not has_stroke_features:
            clinical_impression.append("Neurologic symptoms documented; screen for focal deficits, altered mental status, meningitis, stroke/TIA, hypoglycemia, and dangerous headache features.")
            possible_conditions.extend(["migraine or tension headache", "vestibular syndrome", "neurologic event requiring exclusion if focal deficits are present"])
            recommended_tests.extend(["focused neurologic exam", "blood glucose", "neuroimaging if red flags or focal neurologic findings"])
            medications.extend([
                "acetaminophen/paracetamol or ibuprofen for uncomplicated headache after red flags are excluded",
                "meclizine for vertigo only when clinician assessment supports a peripheral vestibular cause",
            ])
            red_flags.append("Sudden worst headache, focal weakness/numbness, seizure, confusion, neck stiffness, fever with headache, new vision loss, head trauma, or pregnancy/postpartum status.")
            missing_information.append("Neurologic exam, onset timing, headache severity, trauma, fever/neck stiffness, vision symptoms, glucose, and blood pressure.")
            follow_ups.append("urgent evaluation for sudden severe headache, new focal weakness, seizure, confusion, or vision loss")

        if any(term in symptom_text for term in ["dysuria", "hematuria", "urinary", "burning urination", "flank pain"]):
            clinical_impression.append("Urinary symptom cluster documented; distinguish uncomplicated cystitis from pyelonephritis, stone, STI, pregnancy-related UTI, and complicated UTI.")
            possible_conditions.extend(["urinary tract infection", "pyelonephritis consideration", "renal stone consideration"])
            recommended_tests.extend(["urinalysis", "urine culture when indicated", "renal imaging if severe flank pain or obstruction concern"])
            medications.extend([
                "nitrofurantoin, trimethoprim-sulfamethoxazole (TMP-SMX), or fosfomycin for confirmed uncomplicated cystitis only when patient factors and local resistance support use",
                "phenazopyridine for short-term dysuria relief only if clinician confirms suitability and renal function is acceptable",
            ])
            medication_cautions.extend([
                "Do not use nitrofurantoin for suspected pyelonephritis or systemic infection.",
                "Avoid TMP-SMX when allergy, important interactions, pregnancy concerns, renal issues, or high local resistance make it unsuitable.",
            ])
            red_flags.append("Flank pain, fever with rigors, vomiting, pregnancy, male patient with UTI symptoms, immunocompromised status, kidney disease, obstruction concern, or sepsis signs.")
            missing_information.append("Urinary frequency/urgency, flank pain, fever value, pregnancy status, sex, prior UTI/resistance history, renal function, and urinalysis results.")

        if has_ankle_trauma:
            clinical_impression.append("Acute ankle trauma documented; lateral malleolar tenderness and difficulty bearing weight raise concern for fracture and clinically significant sprain.")
            possible_conditions.extend([
                "lateral ankle sprain involving anterior talofibular ligament or calcaneofibular ligament",
                "distal fibula/lateral malleolus fracture requiring exclusion",
                "syndesmotic/high ankle sprain if pain is above the ankle joint or external rotation stress is positive",
                "peroneal tendon injury or osteochondral injury if pain, instability, locking, or persistent symptoms continue",
            ])
            recommended_tests.extend([
                "ankle X-ray series is indicated by Ottawa Ankle Rules because of lateral malleolus tenderness and difficulty bearing weight",
                "assess ability to take four steps, posterior edge/tip malleolar tenderness, base of fifth metatarsal tenderness, and navicular tenderness",
                "neurovascular exam including pulses, capillary refill, sensation, and motor function",
                "consider orthopedic/sports medicine review if X-ray is positive, instability is severe, or weight bearing remains difficult",
            ])
            medications.extend([
                "acetaminophen/paracetamol for pain if no severe liver disease or overdose risk",
                "ibuprofen or naproxen for pain and swelling if no renal disease, ulcer/GI bleed risk, anticoagulant use, severe heart failure, NSAID allergy, or pregnancy concern",
                "topical diclofenac gel for localized ankle pain if skin is intact and NSAIDs are suitable",
            ])
            medication_cautions.extend([
                "Do not mask severe pain and return to sport before fracture, tendon injury, and instability are assessed.",
                "Avoid NSAIDs when contraindications are present; consider acetaminophen/paracetamol instead after clinician review.",
            ])
            red_flags.extend([
                "Inability to bear weight, deformity, rapidly increasing swelling, severe uncontrolled pain, numbness, pale/cold foot, weak pulses, open wound, or suspected dislocation.",
            ])
            missing_information.append("Exact ability to bear four steps, foot tenderness at base of fifth metatarsal/navicular, neurovascular status, deformity, bruising, instability tests, and X-ray result.")
            follow_ups.append("rest, ice, compression, elevation, ankle support/brace, and crutches as needed until fracture is excluded and clinician-directed rehabilitation begins")

        has_skin_allergy_features = any(term in symptom_text for term in ["rash", "itching"]) or (
            "swelling" in symptom_text and any(term in symptom_text for term in ["facial", "tongue", "lip", "hives", "urticaria", "allergic"])
        )
        if has_skin_allergy_features:
            clinical_impression.append("Skin/allergy symptoms documented; distinguish urticaria/allergic reaction, dermatitis, viral exanthem, cellulitis, and severe drug reaction.")
            possible_conditions.extend(["allergic reaction", "viral exanthem", "dermatitis or cellulitis depending on exam"])
            recommended_tests.extend(["skin exam with distribution documentation", "CBC or inflammatory markers if cellulitis/systemic illness suspected"])
            medications.extend([
                "cetirizine or loratadine for urticaria/itching if clinically appropriate",
                "diphenhydramine for severe itching only with sedation/fall-risk precautions",
                "topical hydrocortisone for limited inflammatory dermatitis when infection is not suspected",
                "mupirocin, cephalexin, or other antibiotic therapy only if bacterial skin infection is diagnosed and allergy risk is checked",
            ])
            medication_cautions.append("Avoid topical steroids on undiagnosed infected lesions unless clinician assessment supports use.")
            red_flags.append("Facial/tongue swelling, breathing difficulty, mucosal involvement, blistering, fever with rash, rapidly spreading redness, severe pain, or hypotension.")
            missing_information.append("Rash distribution, mucosal involvement, new drugs/foods/exposures, fever, pain, warmth, drainage, and airway symptoms.")
            follow_ups.append("urgent evaluation for facial swelling, breathing difficulty, mucosal involvement, or rapidly spreading rash")

        if any("diabetes" in term for term in diagnoses):
            clinical_impression.append("Diabetes history documented; review glycemic control, acute illness effect, kidney function, and hypoglycemia risk.")
            possible_conditions.append("diabetes requiring ongoing monitoring")
            recommended_tests.extend(["blood glucose review", "HbA1c if not recently checked", "renal function and urine albumin screening when indicated"])
            medications.append("metformin, insulin, GLP-1 receptor agonists, SGLT2 inhibitors, or sulfonylureas should be continued/adjusted only after clinician review of glucose pattern, renal function, and acute illness status")
            medication_cautions.append("Hold or adjust diabetes medicines only under clinician direction; review hypoglycemia, dehydration, renal function, and sick-day rules.")
            missing_information.append("Diabetes type, current diabetes medicines, glucose readings, HbA1c, renal function, hypoglycemia episodes, and oral intake.")

        if any("hypertension" in term for term in diagnoses):
            clinical_impression.append("Hypertension history documented; assess current blood pressure, end-organ symptoms, renal function, and adherence before medication changes.")
            possible_conditions.append("hypertension requiring risk assessment")
            recommended_tests.extend(["repeat blood pressure measurement", "renal function and electrolytes when indicated"])
            if not has_stroke_features:
                medications.append("amlodipine, lisinopril/losartan, hydrochlorothiazide/chlorthalidone, or other antihypertensives require clinician selection based on BP, comorbidities, renal function, potassium, pregnancy status, and contraindications")
            medication_cautions.append("Do not start ACE inhibitors/ARBs, diuretics, or beta blockers without checking contraindications, renal function, potassium, pregnancy status, heart rate, and comorbidities.")
            missing_information.append("Actual BP readings, home BP trend, symptoms of emergency, kidney function, potassium, pregnancy status, and current antihypertensive adherence.")

        if any(term in symptom_text for term in ["wheezing", "shortness of breath", "dyspnea", "asthma", "copd"]):
            clinical_impression.append("Wheeze or dyspnea documented; assess severity, oxygenation, triggers, inhaler use, and need for urgent bronchodilator/oxygen support.")
            possible_conditions.extend(["asthma exacerbation or reactive airway bronchospasm", "COPD exacerbation if COPD history is present", "pneumonia, pulmonary embolism, heart failure, or other dyspnea cause depending on exam and risk factors"])
            recommended_tests.extend(["oxygen saturation trend", "peak expiratory flow or spirometry when feasible", "chest X-ray if hypoxia, fever, chest pain, focal findings, or poor response to bronchodilator"])
            medications.append("albuterol/salbutamol inhaler for wheeze or bronchospasm if clinically appropriate")
            medications.append("ipratropium or systemic corticosteroid therapy only for clinician-confirmed asthma/COPD exacerbation when indicated")
            medication_cautions.append("Assess oxygen saturation and severity before treating wheeze as outpatient.")
            red_flags.append("Severe breathlessness, inability to speak full sentences, silent chest, cyanosis, exhaustion, altered mental status, low oxygen saturation, or poor response to bronchodilator.")
            missing_information.append("Baseline asthma/COPD severity, inhaler use, triggers, prior ICU/intubation, oxygen saturation trend, lung exam, and response to bronchodilator.")

        if not possible_conditions:
            clinical_impression.append("The note is too sparse for a reliable problem-specific assessment; obtain a focused history, exam, and vitals before choosing medications.")
            possible_conditions.append("insufficient information for a narrow differential")
            recommended_tests.extend(["focused history and physical exam", "targeted labs or imaging based on clinician assessment"])
            medications.append("medication plan requires more clinical detail and licensed clinician review")
            red_flags.append("Any severe, worsening, unexplained, or life-threatening symptom should trigger urgent evaluation.")

        follow_ups.append("clinician should review recommendations, assess red flags, allergies, contraindications, and local protocols")

        if current_medications:
            medication_cautions.append(
                "Medication reconciliation needed for documented medicines: "
                + ", ".join(current_medications[:8])
            )
        if allergies:
            medication_cautions.append("Avoid documented allergy triggers unless clinician verifies the entry is not a true allergy: " + ", ".join(allergies[:8]))

        return {
            "clinical_impression": self._dedupe(clinical_impression)[:8],
            "possible_conditions": self._dedupe(possible_conditions)[:8],
            "recommended_tests": self._dedupe(recommended_tests)[:10],
            "medications": self._dedupe(medications)[:12],
            "medication_cautions": self._dedupe(medication_cautions)[:10],
            "red_flags": self._dedupe(red_flags)[:10],
            "missing_information": self._dedupe(missing_information)[:10],
            "follow_ups": self._dedupe(follow_ups)[:6],
            "safety_notes": [
                "AI recommendations require clinician review and are not a final diagnosis.",
                "Medication options are examples, not prescriptions; dose, route, duration, and suitability require licensed clinician verification.",
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

    def _split_medication_list(self, value: str) -> list[str]:
        value = re.sub(r"\b(?:for|because of|due to)\s+[^,;\n.]+", "", value, flags=re.I)
        return [item.strip() for item in re.split(r",|;|\band\b|\bwith\b|\+", value, flags=re.I)]

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

    def _looks_like_medication(self, value: str) -> bool:
        lowered = value.lower().strip()
        if not lowered:
            return False
        blocked = {
            "none",
            "nil",
            "no medications",
            "not taking medications",
            "no current medications",
            "no regular medications",
        }
        if lowered in blocked or lowered.startswith(("no ", "denies ")):
            return False
        non_medical_terms = [
            "follow up",
            "review",
            "investigation",
            "test",
            "x-ray",
            "xray",
            "blood work",
            "diet",
            "exercise",
        ]
        if any(term in lowered for term in non_medical_terms):
            return False
        medication_cues = [
            "mg",
            "mcg",
            "g ",
            "ml",
            "tablet",
            "tab ",
            "tab.",
            "capsule",
            "cap ",
            "cap.",
            "syrup",
            "inj",
            "injection",
            "drops",
            "cream",
            "ointment",
            "spray",
        ]
        known = self._known_medication_terms()
        return any(cue in f"{lowered} " for cue in medication_cues) or any(
            re.search(rf"\b{re.escape(term)}\b", lowered) for term in known
        )

    def _extract_catalog_medications(self, clinical_note: str, existing: list[str]) -> list[str]:
        note = clinical_note.lower()
        medications: list[str] = []
        existing_text = " ".join(existing).lower()
        for term in self._known_medication_terms():
            if term in existing_text:
                continue
            for match in re.finditer(rf"\b{re.escape(term)}\b", note):
                prefix = note[max(0, match.start() - 28):match.start()]
                if re.search(r"\b(allergic to|allergy to|allerg(?:y|ies):|no|denies|without)\s*$", prefix):
                    continue
                medications.append(term)
                break
        return medications

    def _known_medication_terms(self) -> list[str]:
        return [
            "acetaminophen",
            "albuterol",
            "amoxicillin",
            "anticoagulant therapy",
            "apixaban",
            "aspirin",
            "atorvastatin",
            "azithromycin",
            "cetirizine",
            "dabigatran",
            "dolo",
            "edoxaban",
            "enoxaparin",
            "heparin",
            "ibuprofen",
            "insulin",
            "levothyroxine",
            "lisinopril",
            "metformin",
            "omeprazole",
            "ondansetron",
            "pantoprazole",
            "paracetamol",
            "rivaroxaban",
            "salbutamol",
            "warfarin",
        ]

    def _is_negated(self, note: str, term: str) -> bool:
        index = note.find(term)
        if index < 0:
            return False
        prefix = note[max(0, index - 40):index]
        return bool(re.search(r"\b(no|denies|without|negative for|no history of)\b[^.;,\n]*$", prefix))

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
            impression = self._format_bullets(
                recommendations.get("clinical_impression", []),
                "Clinical impression requires more documented information.",
            )
            conditions = self._format_bullets(
                recommendations.get("possible_conditions", []),
                "Requires clinician assessment.",
            )
            tests = self._format_bullets(
                recommendations.get("recommended_tests", []),
                "Targeted testing based on exam.",
            )
            meds = self._format_bullets(
                recommendations.get("medications", []),
                "Medication plan requires clinician review.",
            )
            cautions = self._format_bullets(
                recommendations.get("medication_cautions", []),
                "Check allergies, contraindications, interactions, pregnancy status, renal function, and liver disease.",
            )
            red_flags = self._format_bullets(
                recommendations.get("red_flags", []),
                "Escalate severe or worsening symptoms.",
            )
            missing = self._format_bullets(
                recommendations.get("missing_information", []),
                "Collect complete history, exam, vitals, allergies, and current medications.",
            )
            follow_ups = self._format_bullets(
                recommendations.get("follow_ups", []),
                "Follow up based on symptom severity.",
            )
            safety = self._format_bullets(
                recommendations.get("safety_notes", []),
                "Recommendations require licensed clinician review and do not replace diagnosis or local care protocols.",
            )
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
                "Clinical Impression\n"
                f"{impression}\n\n"
                "Differential Diagnosis\n"
                f"{conditions}\n\n"
                "Tests/Workup\n"
                f"{tests}\n"
                "\nMedication Options\n"
                f"{meds}\n"
                "\nMedication Safety Checks\n"
                f"{cautions}\n\n"
                "Red Flags\n"
                f"{red_flags}\n\n"
                "Missing Information\n"
                f"{missing}\n\n"
                "Follow-up\n"
                f"{follow_ups}\n\n"
                "Safety Notes\n"
                f"{safety}"
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

    def _format_bullets(self, values: list[str], fallback: str) -> str:
        items = [str(item).strip() for item in values if str(item).strip()]
        if not items:
            items = [fallback]
        return "\n".join(f"- {item}" for item in items)
