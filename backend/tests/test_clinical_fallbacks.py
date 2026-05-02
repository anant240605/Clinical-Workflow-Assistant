import unittest

from agents.extraction_agent import MedicalExtractionAgent
from agents.pipeline import ClinicalWorkflowPipeline
from agents.reasoning_agent import ReasoningAgent
from services.clinical_safety import ClinicalSafetyValidator
from services.llm_service import LLMService
from services.medical_api import MockMedicalAPI
from services.prompt_loader import PromptLoader


class ClinicalFallbackTests(unittest.TestCase):
    def setUp(self):
        self.llm = LLMService()

    def test_extracts_prescription_style_medications(self):
        note = "Rx: Tab Paracetamol 500 mg, syrup cetirizine. Patient has fever and cold."

        extraction = self.llm.heuristic_extract(note)

        medications = " ".join(extraction["medications"]).lower()
        self.assertIn("paracetamol", medications)
        self.assertIn("cetirizine", medications)

    def test_sparse_urinary_note_gets_specific_recommendations(self):
        extraction = self.llm.heuristic_extract("c/o burning urination and fever for 2 days")
        recommendations = self.llm.heuristic_recommendations(extraction)

        conditions = " ".join(recommendations["possible_conditions"]).lower()
        medications = " ".join(recommendations["medications"]).lower()
        self.assertIn("urinary tract infection", conditions)
        self.assertIn("nitrofurantoin", medications)
        self.assertIn("trimethoprim-sulfamethoxazole", medications)
        self.assertIn("fosfomycin", medications)
        self.assertTrue(recommendations["red_flags"])
        self.assertTrue(recommendations["missing_information"])

    def test_allergy_mentions_are_not_active_medications(self):
        extraction = self.llm.heuristic_extract("Allergy: amoxicillin. No current medications.")

        self.assertEqual(extraction["medications"], [])
        self.assertIn("amoxicillin", " ".join(extraction["allergies"]).lower())

    def test_wheeze_gets_bronchodilator_specific_guidance(self):
        extraction = self.llm.heuristic_extract("Patient has wheezing and shortness of breath. History of asthma.")
        recommendations = self.llm.heuristic_recommendations(extraction)

        conditions = " ".join(recommendations["possible_conditions"]).lower()
        medications = " ".join(recommendations["medications"]).lower()
        self.assertIn("asthma exacerbation", conditions)
        self.assertIn("albuterol", medications)

    def test_ankle_trauma_gets_fracture_workup_not_rash_guidance(self):
        note = (
            "27-year-old male with left ankle pain after twisting injury. "
            "Immediate swelling and difficulty bearing weight. "
            "Tenderness over lateral malleolus."
        )
        extraction = self.llm.heuristic_extract(note)
        recommendations = self.llm.heuristic_recommendations(extraction)

        conditions = " ".join(recommendations["possible_conditions"]).lower()
        tests = " ".join(recommendations["recommended_tests"]).lower()
        medications = " ".join(recommendations["medications"]).lower()
        self.assertEqual(extraction["medications"], [])
        self.assertIn("lateral ankle sprain", conditions)
        self.assertIn("lateral malleolus fracture", conditions)
        self.assertIn("ankle x-ray", tests)
        self.assertIn("acetaminophen", medications)
        self.assertIn("ibuprofen", medications)
        self.assertNotIn("urticaria", conditions)

    def test_sudden_focal_deficit_gets_stroke_emergency_guidance(self):
        note = (
            "65-year-old female with sudden right-sided weakness, slurred speech, "
            "difficulty in understanding commands, facial droop. No history of head injury. "
            "Past medical history includes hypertension and atrial fibrillation. "
            "Patient is on anticoagulant therapy but compliance is uncertain. "
            "BP 170/105, HR 88 bpm irregular."
        )
        extraction = self.llm.heuristic_extract(note)
        recommendations = self.llm.heuristic_recommendations(extraction)

        diagnoses = " ".join(extraction["diagnoses"]).lower()
        conditions = " ".join(recommendations["possible_conditions"]).lower()
        tests = " ".join(recommendations["recommended_tests"]).lower()
        medications = " ".join(recommendations["medications"]).lower()
        self.assertIn("atrial fibrillation", diagnoses)
        self.assertNotIn("head injury", diagnoses)
        self.assertIn("acute ischemic stroke", conditions)
        self.assertIn("intracranial hemorrhage", conditions)
        self.assertIn("non-contrast ct", tests)
        self.assertIn("last known well", tests)
        self.assertIn("alteplase", medications)
        self.assertIn("tenecteplase", medications)
        self.assertIn("do not give outpatient", medications)
        self.assertNotIn("meclizine", medications)

    def test_heart_failure_note_gets_congestion_guidance_not_skin_guidance(self):
        note = (
            "62-year-old male with swelling in both legs and breathlessness for the past 2 weeks. "
            "Symptoms worsen on lying flat and improve on sitting up. "
            "Fatigue and reduced exercise tolerance. History of hypertension and coronary artery disease. "
            "BP 140/90, HR 95 bpm, SpO2 94%. Bilateral pitting edema and basal lung crackles."
        )
        extraction = self.llm.heuristic_extract(note)
        recommendations = ClinicalSafetyValidator().validate(
            note,
            extraction,
            self.llm.heuristic_recommendations(extraction),
        )

        symptoms = " ".join(extraction["symptoms"]).lower()
        conditions = " ".join(recommendations["possible_conditions"]).lower()
        tests = " ".join(recommendations["recommended_tests"]).lower()
        medications = " ".join(recommendations["medications"]).lower()
        self.assertIn("orthopnea", symptoms)
        self.assertIn("pitting edema", symptoms)
        self.assertIn("basal lung crackles", symptoms)
        self.assertIn("decompensated heart failure", conditions)
        self.assertIn("bnp", tests)
        self.assertIn("echocardiography", tests)
        self.assertIn("furosemide", medications)
        self.assertNotIn("allergic reaction", conditions)
        self.assertNotIn("cetirizine", medications)

    def test_heart_failure_pipeline_report_has_extracted_details(self):
        async def run_pipeline():
            note = (
                "Patient has swelling in both legs and breathlessness for 2 weeks. "
                "Symptoms worsen on lying flat and improve on sitting up. "
                "Past history of hypertension and coronary artery disease. "
                "Vital signs: BP 140/90 mmHg, HR 95 bpm, SpO2 94%. "
                "Bilateral pitting edema and basal lung crackles noted."
            )
            return await ClinicalWorkflowPipeline().run(note)

        import asyncio

        report = asyncio.run(run_pipeline())

        self.assertIn("orthopnea", [item.lower() for item in report.extracted_info.symptoms])
        self.assertTrue(report.extracted_info.vital_signs)
        self.assertIn("heart failure", " ".join(report.recommendations.possible_conditions).lower())
        self.assertNotIn("the documented symptoms", report.report.lower())

    def test_reasoning_merge_drops_insufficient_when_fallback_is_specific(self):
        agent = ReasoningAgent(self.llm, prompt_loader=None)
        primary = {
            "possible_conditions": ["insufficient information for a narrow differential"],
            "recommended_tests": [],
            "medications": [],
            "follow_ups": [],
            "safety_notes": [],
        }
        fallback = {
            "possible_conditions": ["viral upper respiratory infection"],
            "recommended_tests": ["COVID-19 antigen or PCR test"],
            "medications": ["antipyretic if clinically appropriate"],
            "follow_ups": ["clinician should review recommendations"],
            "safety_notes": ["AI recommendations require clinician review."],
        }

        merged = agent._merge_recommendations(primary, fallback)

        self.assertEqual(merged["possible_conditions"], ["viral upper respiratory infection"])
        self.assertIn("antipyretic if clinically appropriate", merged["medications"])

    def test_safety_validator_corrects_weak_stroke_recommendations(self):
        validator = ClinicalSafetyValidator()
        note = (
            "Sudden right-sided weakness with facial droop and slurred speech. "
            "History of atrial fibrillation and anticoagulant use."
        )
        extraction = {
            "symptoms": ["weakness", "facial droop", "slurred speech"],
            "diagnoses": ["atrial fibrillation"],
            "medications": ["anticoagulant therapy"],
            "allergies": [],
            "relevant_negatives": [],
        }
        weak_recommendations = {
            "clinical_impression": [],
            "possible_conditions": ["migraine or tension headache"],
            "recommended_tests": ["focused neurologic exam"],
            "medications": ["meclizine for vertigo", "acetaminophen for headache"],
            "medication_cautions": [],
            "red_flags": [],
            "missing_information": [],
            "follow_ups": [],
            "safety_notes": [],
        }

        corrected = validator.validate(note, extraction, weak_recommendations)

        conditions = " ".join(corrected["possible_conditions"]).lower()
        tests = " ".join(corrected["recommended_tests"]).lower()
        medications = " ".join(corrected["medications"]).lower()
        self.assertIn("acute ischemic stroke", conditions)
        self.assertIn("non-contrast ct", tests)
        self.assertIn("do not give outpatient", medications)
        self.assertNotIn("meclizine", medications)

    def test_extraction_removes_llm_items_that_are_negated(self):
        agent = MedicalExtractionAgent(
            self.llm,
            PromptLoader(),
            MockMedicalAPI(),
        )
        merged = agent._merge_extraction(
            {
                "symptoms": ["seizure"],
                "diagnoses": ["head injury"],
                "medications": [],
                "allergies": [],
                "relevant_negatives": ["No history of head injury", "No seizure activity observed"],
                "vital_signs": [],
            },
            {
                "symptoms": [],
                "diagnoses": [],
                "medications": [],
                "allergies": [],
                "relevant_negatives": [],
                "vital_signs": [],
            },
        )

        self.assertNotIn("head injury", [item.lower() for item in merged["diagnoses"]])
        self.assertNotIn("seizure", [item.lower() for item in merged["symptoms"]])

    def test_extraction_references_stay_prompt_parseable(self):
        async def run_agent():
            state = {
                "cleaned_note": "Rx: Tab Paracetamol 500 mg. Patient has fever."
            }
            agent = MedicalExtractionAgent(
                self.llm,
                PromptLoader(),
                MockMedicalAPI(),
            )
            return await agent.run(state)

        import asyncio

        state = asyncio.run(run_agent())
        prompt = f"Extracted information:\n{state['extracted_info']}"

        parsed = self.llm._extract_dict_from_prompt(prompt)

        self.assertIn("fever", parsed["symptoms"])
        self.assertIsInstance(parsed["medical_references"][0], dict)


if __name__ == "__main__":
    unittest.main()
