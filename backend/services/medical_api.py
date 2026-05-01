from models.schemas import MedicalReference


class MockMedicalAPI:
    def __init__(self):
        self.catalog = {
            "fever": MedicalReference(
                term="fever",
                category="symptom",
                clinical_hint="May indicate infection, inflammation, heat illness, or other systemic process.",
            ),
            "cough": MedicalReference(
                term="cough",
                category="symptom",
                clinical_hint="Common in upper respiratory infection, bronchitis, pneumonia, asthma, and reflux.",
            ),
            "fatigue": MedicalReference(
                term="fatigue",
                category="symptom",
                clinical_hint="Nonspecific symptom; consider infection, anemia, endocrine, sleep, and medication causes.",
            ),
            "hypertension": MedicalReference(
                term="hypertension",
                category="diagnosis",
                clinical_hint="Confirm with repeated blood pressure measurements and assess cardiovascular risk.",
            ),
            "diabetes": MedicalReference(
                term="diabetes",
                category="diagnosis",
                clinical_hint="Assess glucose control, renal function, complications, and medication plan.",
            ),
            "chest pain": MedicalReference(
                term="chest pain",
                category="symptom",
                clinical_hint="Assess cardiac, pulmonary, gastrointestinal, and musculoskeletal causes; urgent evaluation may be needed.",
            ),
            "palpitations": MedicalReference(
                term="palpitations",
                category="symptom",
                clinical_hint="Consider rhythm assessment, stimulant/medication review, thyroid disease, anemia, and anxiety.",
            ),
            "shortness of breath": MedicalReference(
                term="shortness of breath",
                category="symptom",
                clinical_hint="Assess oxygenation, cardiopulmonary exam, infection, asthma/COPD, heart failure, and embolic risk.",
            ),
            "abdominal pain": MedicalReference(
                term="abdominal pain",
                category="symptom",
                clinical_hint="Location, severity, peritoneal signs, pregnancy status, and associated GI/GU symptoms guide workup.",
            ),
            "vomiting": MedicalReference(
                term="vomiting",
                category="symptom",
                clinical_hint="Assess hydration, electrolyte risk, abdominal exam, neurologic red flags, and medication/toxin exposure.",
            ),
            "diarrhea": MedicalReference(
                term="diarrhea",
                category="symptom",
                clinical_hint="Assess hydration, blood, fever, travel, antibiotic exposure, and outbreak risk.",
            ),
            "headache": MedicalReference(
                term="headache",
                category="symptom",
                clinical_hint="Screen for sudden onset, neurologic deficit, fever, trauma, pregnancy, immunosuppression, and visual symptoms.",
            ),
            "dizziness": MedicalReference(
                term="dizziness",
                category="symptom",
                clinical_hint="Clarify vertigo vs presyncope, check vitals, glucose, neurologic exam, medication effects, and hydration.",
            ),
            "dysuria": MedicalReference(
                term="dysuria",
                category="symptom",
                clinical_hint="Consider urinary tract infection, sexually transmitted infection, stones, and local irritation.",
            ),
            "flank pain": MedicalReference(
                term="flank pain",
                category="symptom",
                clinical_hint="Consider renal stone, pyelonephritis, musculoskeletal pain, and abdominal pathology.",
            ),
            "rash": MedicalReference(
                term="rash",
                category="symptom",
                clinical_hint="Assess distribution, mucosal involvement, fever, medication exposure, allergy, and infection signs.",
            ),
            "metformin": MedicalReference(
                term="metformin",
                category="medication",
                clinical_hint="Common diabetes medication; review renal function, GI effects, dosing, and contraindications.",
            ),
            "insulin": MedicalReference(
                term="insulin",
                category="medication",
                clinical_hint="Review glucose pattern, dose timing, hypoglycemia risk, nutrition, and sick-day plan.",
            ),
            "amoxicillin": MedicalReference(
                term="amoxicillin",
                category="medication",
                clinical_hint="Beta-lactam antibiotic; verify indication, dose, duration, allergy history, and local guidance.",
            ),
        }

    def lookup_terms(self, terms: list[str]) -> list[MedicalReference]:
        matches: dict[str, MedicalReference] = {}
        for term in terms:
            lowered = term.lower()
            found_catalog_match = False
            for key, reference in self.catalog.items():
                if key in lowered:
                    matches[key] = reference
                    found_catalog_match = True
            if lowered and not found_catalog_match and lowered not in matches and len(lowered) <= 60:
                matches[lowered] = MedicalReference(
                    term=term,
                    category="clinical_term",
                    clinical_hint="Review this extracted term in context and verify significance during clinician assessment.",
                )
        return list(matches.values())
