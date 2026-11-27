"""
Comprehensive verification test suite for modernized MDAgents framework.
Validates dataset loading, complexity triage, recruitment graphs,
pipelines, entropy calculations, baselines, decision maker, and evaluation metrics.
"""

import unittest
from unittest.mock import patch

from data import load_dataset, sample_dataset, create_question_text, get_few_shot_examples
from complexity import determine_complexity
from recruiter import recruit_pcc, recruit_mdt, recruit_ict, recruit
from pipelines.low import run_low_pipeline
from pipelines.moderate import run_moderate_pipeline, calculate_consensus_entropy
from pipelines.high import run_high_pipeline
from decide import extract_answer_choice, make_final_decision_with_ensemble
from eval.metrics import compute_accuracy, compute_consensus_entropy, estimate_cost, significance_test
from eval.analysis import analyze_results, format_summary_table

class TestMDAgentsModernized(unittest.TestCase):

    def test_01_dataset_loaders_and_schema(self):
        """Test dataset loading, schema standardization, and deterministic sampling."""
        dataset = load_dataset("medqa", split="test")
        self.assertGreater(len(dataset), 0)
        sample = dataset[0]
        
        required_keys = ["id", "question", "options", "answer", "context", "image_path", "video_path", "modality"]
        for key in required_keys:
            self.assertIn(key, sample)

        sampled_1 = sample_dataset(dataset, n_samples=2, seed=42)
        sampled_2 = sample_dataset(dataset, n_samples=2, seed=42)
        self.assertEqual([s["id"] for s in sampled_1], [s["id"] for s in sampled_2])

        q_text, media = create_question_text(sample, shuffle_options=False)
        self.assertIn("Options:", q_text)

    def test_02_complexity_triage(self):
        """Test complexity moderator classification and forced modes."""
        self.assertEqual(determine_complexity("query", forced_difficulty="low"), "low")
        self.assertEqual(determine_complexity("query", forced_difficulty="moderate"), "moderate")
        self.assertEqual(determine_complexity("query", forced_difficulty="high"), "high")

        with patch("complexity.chat", return_value="This case is moderate difficulty. Option 2) moderate."):
            comp = determine_complexity("Complex clinical query", forced_difficulty="adaptive")
            self.assertEqual(comp, "moderate")

    def test_03_expert_recruitment_and_graph(self):
        """Test expert recruitment for PCC, MDT, and ICT."""
        pcc_plan = recruit_pcc()
        self.assertEqual(pcc_plan.complexity, "low")
        self.assertEqual(len(pcc_plan.agents), 1)

        mdt_plan = recruit_mdt("Patient with dyspnea", num_agents=3)
        self.assertEqual(mdt_plan.complexity, "moderate")
        self.assertGreaterEqual(len(mdt_plan.agents), 2)
        self.assertIn(mdt_plan.agents[0].role, mdt_plan.communication_graph)

        ict_plan = recruit_ict("Complex multiorgan failure")
        self.assertEqual(ict_plan.complexity, "high")
        self.assertGreaterEqual(len(ict_plan.teams), 2)

    @patch("agents.chat")
    def test_04_low_complexity_pipeline(self, mock_chat):
        """Test Primary Care Clinician execution."""
        mock_chat.return_value = "Based on clinical review, the correct option is (B) Hashimoto's thyroiditis."
        plan = recruit_pcc()
        res = run_low_pipeline("Thyroid clinical case", plan)
        self.assertEqual(res["complexity"], "low")
        self.assertIn("Hashimoto", res["final_answer"])
        choice = extract_answer_choice(res["final_answer"])
        self.assertEqual(choice, "B")

    @patch("pipelines.moderate.chat")
    @patch("agents.chat")
    def test_05_moderate_complexity_pipeline(self, mock_agent_chat, mock_mod_chat):
        """Test MDT multi-round discussion, consensus check, and entropy."""
        mock_agent_chat.return_value = "Diagnosis indicates Answer: (A)."
        mock_mod_chat.return_value = "Moderator confirms consensus on Answer: (A)."
        
        plan = recruit_mdt("Patient case", num_agents=3)
        res = run_moderate_pipeline("Patient case", plan, max_rounds=2, max_turns=2)
        self.assertEqual(res["complexity"], "moderate")
        self.assertIn("entropy_history", res)
        self.assertGreaterEqual(len(res["entropy_history"]), 1)
        
        choice = extract_answer_choice(res["final_answer"])
        self.assertEqual(choice, "A")

    @patch("pipelines.high.chat")
    @patch("agents.chat")
    def test_06_high_complexity_pipeline(self, mock_agent_chat, mock_lead_chat):
        """Test ICT team investigation and accumulated report synthesis."""
        mock_agent_chat.return_value = "Specialist findings documented."
        mock_lead_chat.return_value = "Comprehensive diagnosis establishes Answer: (C)."

        plan = recruit_ict("High complexity case")
        res = run_high_pipeline("High complexity case", plan)
        self.assertEqual(res["complexity"], "high")
        self.assertIn("accumulated_reports", res)
        self.assertGreaterEqual(len(res["accumulated_reports"]), 2)
        choice = extract_answer_choice(res["final_answer"])
        self.assertEqual(choice, "C")

    def test_07_evaluation_metrics_and_entropy(self):
        """Test accuracy, entropy, cost, and significance calculations."""
        preds = ["A", "B", "C", "D"]
        gts = ["A", "B", "A", "D"]
        acc = compute_accuracy(preds, gts)
        self.assertEqual(acc["accuracy"], 0.75)
        self.assertEqual(acc["correct"], 3)

        h_zero = compute_consensus_entropy(["A", "A", "A"])
        self.assertEqual(h_zero, 0.0)

        h_div = compute_consensus_entropy(["A", "B", "C", "D"])
        self.assertAlmostEqual(h_div, 2.0, places=2)

        cost = estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 0.75, places=2)

        sig = significance_test([1, 1, 1, 1], [0, 0, 1, 0])
        self.assertIn("p_value", sig)

if __name__ == "__main__":
    unittest.main()
