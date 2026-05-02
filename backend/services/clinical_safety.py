import re
from typing import Any


class ClinicalSafetyValidator:
    keys = [
        "clinical_impression",
        "possible_conditions",
        "recommended_tests",
        "medications",
        "medication_cautions",
        "red_flags",
        "missing_information",
        "follow_ups",
        "safety_notes",
    ]

    low_acuity_medication_terms = [
        "meclizine",
        "dextromethorphan",
        "guaifenesin",
        "pseudoephedrine",
        "topical hydrocortisone",
        "cetirizine",
        "loratadine",
        "diphenhydramine",
        "loperamide",
    ]

    def validate(
        self,
        clinical_note: str,
        extraction: dict[str, Any],
        recommendations: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        checked = {key: list(recommendations.get(key, [])) for key in self.keys}
        text = self._clinical_text(clinical_note, extraction)

        if self._has_stroke_red_flags(text):
            self._apply_stroke_pathway(checked)

        if self._has_chest_pain_red_flags(text):
            self._apply_chest_pain_pathway(checked)

        if self._has_anaphylaxis_red_flags(text):
            self._apply_anaphylaxis_pathway(checked)

        if self._has_sepsis_red_flags(text):
            self._apply_sepsis_pathway(checked)

        self._ensure_core_safety(checked)
        return {key: self._dedupe(checked[key]) for key in self.keys}

    def _clinical_text(self, clinical_note: str, extraction: dict[str, Any]) -> str:
        extracted_parts: list[str] = []
        for key in ["symptoms", "diagnoses", "medications", "allergies", "relevant_negatives"]:
            extracted_parts.extend(str(item) for item in extraction.get(key, []))
        return f"{clinical_note} {' '.join(extracted_parts)}".lower()

    def _has_stroke_red_flags(self, text: str) -> bool:
        focal_deficit = any(
            term in text
            for term in [
                "facial droop",
                "slurred speech",
                "aphasia",
                "difficulty understanding",
                "right-sided weakness",
                "left-sided weakness",
                "unilateral weakness",
                "motor weakness",
            ]
        )
        sudden_or_high_risk = any(
            term in text
            for term in ["sudden", "since morning", "atrial fibrillation", "anticoagulant"]
        )
        return focal_deficit and sudden_or_high_risk

    def _has_chest_pain_red_flags(self, text: str) -> bool:
        if "chest pain" not in text:
            return False
        return any(
            term in text
            for term in [
                "radiating",
                "exertional",
                "diaphoresis",
                "shortness of breath",
                "syncope",
                "hypotension",
                "crushing",
                "pressure",
            ]
        )

    def _has_anaphylaxis_red_flags(self, text: str) -> bool:
        swelling = any(term in text for term in ["tongue swelling", "lip swelling", "facial swelling", "throat swelling"])
        breathing_or_shock = any(term in text for term in ["wheezing", "shortness of breath", "hypotension", "syncope"])
        allergy_context = any(term in text for term in ["hives", "urticaria", "allergic", "allergy", "anaphylaxis"])
        return (swelling and breathing_or_shock) or (allergy_context and breathing_or_shock)

    def _has_sepsis_red_flags(self, text: str) -> bool:
        infection_signal = any(term in text for term in ["fever", "sepsis", "infection", "rigors"])
        systemic_risk = any(
            term in text
            for term in ["confusion", "hypotension", "tachycardia", "rapid breathing", "altered mental status"]
        )
        return infection_signal and systemic_risk

    def _apply_stroke_pathway(self, recommendations: dict[str, list[str]]) -> None:
        self._remove_low_acuity_medications(recommendations)
        recommendations["clinical_impression"].insert(
            0,
            "Emergency neurologic presentation: treat as suspected acute stroke/TIA until proven otherwise.",
        )
        recommendations["possible_conditions"].extend(
            [
                "acute ischemic stroke",
                "intracranial hemorrhage requiring urgent exclusion",
                "transient ischemic attack if deficits resolve",
                "stroke mimic such as hypoglycemia, seizure/postictal deficit, migraine, infection, or metabolic disturbance",
            ]
        )
        recommendations["recommended_tests"].extend(
            [
                "activate emergency stroke pathway/EMS transfer to stroke-capable emergency department immediately",
                "establish exact last known well and symptom onset time",
                "point-of-care blood glucose immediately",
                "urgent non-contrast CT head or MRI brain before antithrombotic/thrombolytic decisions",
                "CTA head/neck or MRA if large vessel occlusion is suspected",
                "ECG/telemetry, CBC, platelets, PT/INR, aPTT, electrolytes, renal function, and troponin per stroke protocol",
                "NIHSS and swallow screen before oral intake",
            ]
        )
        recommendations["medications"].extend(
            [
                "do not give outpatient pain, vertigo, antihypertensive, antiplatelet, anticoagulant, or sedating medication before emergency stroke evaluation unless directed by the stroke team",
                "alteplase or tenecteplase may be considered only by a stroke team if ischemic stroke is confirmed and eligibility criteria are met",
                "aspirin is generally considered only after intracranial hemorrhage is excluded and thrombolysis decisions are complete",
                "anticoagulant reversal may be needed urgently if intracranial hemorrhage or clinically significant anticoagulant effect is present",
            ]
        )
        recommendations["medication_cautions"].extend(
            [
                "Identify anticoagulant name, dose, last dose time, adherence, and INR/anti-Xa or relevant coagulation status where available.",
                "Avoid rapid BP lowering unless directed by stroke protocol; BP targets differ by thrombolysis eligibility and hemorrhage status.",
            ]
        )
        recommendations["red_flags"].append(
            "Sudden unilateral weakness, facial droop, slurred speech, aphasia/confusion, severe headache, reduced consciousness, seizure, very high BP, anticoagulant use, or worsening deficits require emergency stroke evaluation."
        )
        recommendations["missing_information"].append(
            "Last known well time, exact onset time, NIHSS/severity, glucose, anticoagulant details, coagulation results, CT/MRI result, CTA/LVO status, baseline function, and goals of care."
        )
        recommendations["follow_ups"].append(
            "do not manage as routine clinic follow-up; arrange immediate emergency transfer/stroke-team evaluation"
        )

    def _apply_chest_pain_pathway(self, recommendations: dict[str, list[str]]) -> None:
        recommendations["clinical_impression"].append(
            "Potential high-risk chest pain; exclude acute coronary syndrome and other life-threatening cardiopulmonary causes."
        )
        recommendations["possible_conditions"].extend(
            ["acute coronary syndrome", "pulmonary embolism", "aortic dissection", "pneumothorax or pneumonia depending on exam"]
        )
        recommendations["recommended_tests"].extend(
            ["urgent ECG", "serial troponins", "vital sign reassessment", "chest X-ray and further testing based on risk assessment"]
        )
        recommendations["medications"].append(
            "do not provide routine outpatient chest-pain medication advice until ACS and other emergencies are excluded"
        )
        recommendations["red_flags"].append(
            "Active, severe, exertional, radiating, pressure-like chest pain, syncope, dyspnea, diaphoresis, hypotension, or abnormal ECG requires emergency evaluation."
        )

    def _apply_anaphylaxis_pathway(self, recommendations: dict[str, list[str]]) -> None:
        self._remove_low_acuity_medications(recommendations)
        recommendations["clinical_impression"].append(
            "Possible anaphylaxis or airway-threatening allergic reaction; prioritize airway, breathing, circulation, and emergency treatment."
        )
        recommendations["possible_conditions"].extend(["anaphylaxis", "angioedema", "severe allergic reaction"])
        recommendations["recommended_tests"].extend(
            ["immediate airway and vital sign assessment", "oxygen saturation and respiratory exam", "identify trigger and exposure timing"]
        )
        recommendations["medications"].extend(
            [
                "intramuscular epinephrine is first-line for anaphylaxis and should not be delayed when criteria are met",
                "antihistamines and corticosteroids are adjuncts only and must not replace epinephrine for anaphylaxis",
            ]
        )
        recommendations["red_flags"].append(
            "Tongue/lip/throat swelling, wheeze, dyspnea, hypotension, syncope, or rapidly progressive hives/swelling requires emergency treatment."
        )

    def _apply_sepsis_pathway(self, recommendations: dict[str, list[str]]) -> None:
        recommendations["clinical_impression"].append(
            "Possible systemic infection/sepsis; assess urgently rather than treating as a routine outpatient infection."
        )
        recommendations["possible_conditions"].extend(["sepsis or serious systemic infection", "infection with altered mental status or hemodynamic risk"])
        recommendations["recommended_tests"].extend(
            ["urgent full vital sign reassessment", "CBC, CMP, lactate when indicated, cultures when indicated, urinalysis/chest imaging based on source"]
        )
        recommendations["medications"].append(
            "empiric antibiotics and fluids require urgent clinician-directed sepsis assessment and local protocol selection"
        )
        recommendations["red_flags"].append(
            "Fever with confusion, hypotension, rigors, rapid breathing, dehydration, or worsening clinical status requires urgent sepsis evaluation."
        )

    def _remove_low_acuity_medications(self, recommendations: dict[str, list[str]]) -> None:
        recommendations["medications"] = [
            item
            for item in recommendations.get("medications", [])
            if not any(term in item.lower() for term in self.low_acuity_medication_terms)
        ]

    def _ensure_core_safety(self, recommendations: dict[str, list[str]]) -> None:
        recommendations["safety_notes"].extend(
            [
                "AI recommendations are decision support only and require licensed clinician review.",
                "Medication options are not prescriptions; dose, route, duration, and suitability require patient-specific verification.",
            ]
        )
        if not recommendations["possible_conditions"]:
            recommendations["possible_conditions"].append("insufficient information for a narrow differential")
        if not recommendations["missing_information"]:
            recommendations["missing_information"].append(
                "Age, sex, pregnancy status when relevant, symptom duration, severity, vitals, comorbidities, current medications, allergies, and focused exam findings."
            )

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: list[str] = []
        result: list[str] = []
        for value in values:
            cleaned = str(value).strip()
            key = self._dedupe_key(cleaned)
            if cleaned and not any(self._is_similar(key, existing) for existing in seen):
                seen.append(key)
                result.append(cleaned)
        return result

    def _dedupe_key(self, value: str) -> str:
        key = value.lower()
        key = re.sub(r"[^a-z0-9\s]", " ", key)
        key = re.sub(r"\b(a|an|the|to|a stroke capable)\b", " ", key)
        return re.sub(r"\s+", " ", key).strip()

    def _is_similar(self, current: str, existing: str) -> bool:
        if current == existing:
            return True
        if len(current) >= 15 and current in existing:
            return True
        if len(existing) >= 15 and existing in current:
            return True
        current_tokens = set(current.split())
        existing_tokens = set(existing.split())
        if len(current_tokens) < 4 or len(existing_tokens) < 4:
            return False
        overlap = len(current_tokens & existing_tokens) / min(
            len(current_tokens),
            len(existing_tokens),
        )
        return overlap >= 0.82
