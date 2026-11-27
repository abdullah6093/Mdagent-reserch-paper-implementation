"""
Prompt templates for the MDAgents framework.
Based on: Kim et al., "MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making", NeurIPS 2024.
"""

# ---------------------------------------------------------------------------
# 1. Complexity / Difficulty Assessment
# ---------------------------------------------------------------------------

COMPLEXITY_CHECK_PROMPT = (
    "You are an experienced medical professional capable of assessing the complexity of medical queries.\n\n"
    "Now, given the medical query as below, you need to decide the difficulty/complexity of it:\n"
    "{question}\n\n"
    "Please indicate the difficulty/complexity among:\n"
    "1) low - The question can be answered by a single medical expert with general knowledge. It does not require interdisciplinary consultation.\n"
    "2) moderate - The question requires collaboration among a small group of medical experts from different specialties through a Multi-disciplinary Team (MDT) discussion.\n"
    "3) high - The question is highly complex and requires an integrated care team (ICT) approach involving multiple groups of specialists, including an Initial Assessment Team (IAT), a Differential & Expert Team (DET), and a Final Review & Decision Team (FRDT).\n\n"
    "Please respond with exactly one of: low, moderate, high."
)

# ---------------------------------------------------------------------------
# 2. Recruiter Prompt – MDT (moderate)
# ---------------------------------------------------------------------------

RECRUITER_PROMPT_MDT = (
    "You are a medical recruiter responsible for assembling a Multi-disciplinary Team (MDT) of medical experts to discuss the following medical query:\n\n"
    "{question}\n\n"
    "Based on the nature of the query, recruit a suitable team of medical experts. Each expert should have a distinct role and clear description of their expertise.\n\n"
    "Please provide the team in the following format. Use the == operator to separate the role from the description, and use the > operator to separate different experts.\n\n"
    "Example:\n"
    "Cardiologist == A medical doctor specializing in heart diseases and cardiovascular conditions. > "
    "Radiologist == A medical doctor who specializes in diagnosing and treating diseases and injuries using medical imaging techniques. > "
    "Internist == A physician who specializes in internal medicine, focusing on adult diseases.\n\n"
    "Now recruit the appropriate MDT team for the query above."
)

# ---------------------------------------------------------------------------
# 3. Recruiter Prompt – ICT (high)
# ---------------------------------------------------------------------------

RECRUITER_PROMPT_ICT = (
    "You are a medical recruiter responsible for assembling an Integrated Care Team (ICT) to address the following highly complex medical query:\n\n"
    "{question}\n\n"
    "The ICT consists of three groups:\n"
    "1. Initial Assessment Team (IAT) - Performs the preliminary evaluation and gathers baseline findings.\n"
    "2. Differential & Expert Team (DET) - Provides in-depth specialist analysis and differential diagnosis.\n"
    "3. Final Review & Decision Team (FRDT) - Synthesizes all findings and makes the final clinical decision.\n\n"
    "For each group, recruit suitable medical experts. Use the following format:\n\n"
    "[Initial Assessment Team (IAT)]\n"
    "- Member: General Practitioner == A physician providing primary care and initial evaluation.\n"
    "- Member: Emergency Medicine Physician == A doctor specializing in acute care and rapid assessment.\n\n"
    "[Differential & Expert Team (DET)]\n"
    "- Member: Neurologist == A specialist in disorders of the nervous system.\n"
    "- Member: Oncologist == A physician who specializes in diagnosing and treating cancer.\n\n"
    "[Final Review & Decision Team (FRDT)]\n"
    "- Member: Attending Physician == A senior physician overseeing patient care and definitive decision.\n"
    "- Member: Clinical Pharmacologist == A specialist in drug therapy, interactions, and safety.\n\n"
    "Now recruit the appropriate ICT for the query above."
)

# ---------------------------------------------------------------------------
# 4. Agent System Prompt
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = (
    "You are a {role} who {description}. "
    "Your job is to collaborate with other medical experts in a team to address the medical query at hand. "
    "Provide responses grounded in your area of expertise, and engage constructively with the opinions of other team members."
)

# ---------------------------------------------------------------------------
# 5. Initial Assessment Prompt
# ---------------------------------------------------------------------------

INITIAL_ASSESSMENT_PROMPT = (
    "You are a {role}.\n\n"
    "Below are examples of how a medical expert analyzes a question and provides an answer:\n\n"
    "{few_shot_examples}\n\n"
    "Now, given the following medical query, provide your initial assessment.\n\n"
    "Question:\n{question}\n\n"
    "{answer_choices}\n\n"
    "Your assessment:"
)

# ---------------------------------------------------------------------------
# 6. Participation Prompt
# ---------------------------------------------------------------------------

PARTICIPATION_PROMPT = (
    "You are a {role} participating in a multidisciplinary team discussion.\n\n"
    "Medical query under discussion:\n{question}\n\n"
    "Opinions from other team members:\n{opinions}\n\n"
    "Given the opinions from other medical agents in your team, please indicate whether you want to talk to any expert (yes/no). "
    "If not, provide your consolidated opinion."
)

# ---------------------------------------------------------------------------
# 7. Target Selection Prompt
# ---------------------------------------------------------------------------

TARGET_SELECTION_PROMPT = (
    "From the following list of available agents, indicate the agent you want to talk to:\n{agent_list}\n\n"
    "Please respond with the name/role of the expert you wish to consult and state what you would like to discuss with them."
)

# ---------------------------------------------------------------------------
# 8. Persuasion Prompt
# ---------------------------------------------------------------------------

PERSUASION_PROMPT = (
    "Remind your medical expertise and leave your opinion to an expert you chose. You are a {role}.\n\n"
    "The medical query under discussion is:\n{question}\n\n"
    "You are now addressing {target_agent}.\n\n"
    "Deliver your opinion once you are confident enough and in a way to convince other expert with a short reason. "
    "Be concise, cite relevant clinical evidence, and state your position clearly."
)

# ---------------------------------------------------------------------------
# 9. Moderator Feedback Prompt
# ---------------------------------------------------------------------------

MODERATOR_FEEDBACK_PROMPT = (
    "You are the moderator of a multi-disciplinary medical team discussion.\n\n"
    "Medical query:\n{question}\n\n"
    "The following discussion has taken place among team members:\n"
    "{discussion_history}\n\n"
    "As the moderator, review the discussion and provide constructive feedback to each agent to resolve discrepancies and reach clinical consensus."
)

# ---------------------------------------------------------------------------
# 10. Report Synthesis Prompt
# ---------------------------------------------------------------------------

REPORT_SYNTHESIS_PROMPT = (
    "You are a medical report synthesizer.\n\n"
    "Given the following reports and opinions from multiple medical experts:\n"
    "{agent_answers}\n\n"
    "Please complete the following steps:\n"
    "1. Take careful consideration of each expert's report.\n"
    "2. Extract key knowledge, findings, and clinical insights.\n"
    "3. Derive a comprehensive analysis integrating the perspectives of all experts.\n"
    "4. Generate a refined synthesized report with a coherent conclusion.\n\n"
    "Synthesized Report:"
)

# ---------------------------------------------------------------------------
# 11. Final Decision Prompt
# ---------------------------------------------------------------------------

FINAL_DECISION_PROMPT = (
    "You are a final medical decision maker who reviews all opinions from different medical experts and makes the final decision.\n\n"
    "The medical query is:\n{question}\n\n"
    "{answer_choices}\n\n"
    "The following expert opinions and reports have been provided:\n"
    "{agent_answers}\n\n"
    "Based on a careful review of all expert opinions above, provide your final decision. State your chosen answer and brief clinical justification.\n\n"
    "Final Decision:"
)

# ---------------------------------------------------------------------------
# 12. Few-Shot Base Prompt
# ---------------------------------------------------------------------------

FEWSHOT_BASE_PROMPT = (
    "{instruction}\n\n"
    "{few_shot_examples}\n\n"
    "{context}\n\n"
    "Question:\n{question}\n\n"
    "{answer_choices}\n\n"
    "Answer:"
)

# ---------------------------------------------------------------------------
# 13. Baseline Prompts
# ---------------------------------------------------------------------------

ZERO_SHOT_PROMPT = (
    "You are a highly knowledgeable medical expert.\n\n"
    "{context}\n\n"
    "Question:\n{question}\n\n"
    "{answer_choices}\n\n"
    "Please provide your answer directly.\n\n"
    "Answer:"
)

COT_PROMPT = (
    "You are a highly knowledgeable medical expert.\n\n"
    "{context}\n\n"
    "Question:\n{question}\n\n"
    "{answer_choices}\n\n"
    "Let's think step by step.\n"
    "First, carefully analyze the clinical information provided. Then reason through each answer choice systematically before arriving at your final answer.\n\n"
    "Step-by-step reasoning:"
)

COT_SC_PROMPT = (
    "You are a highly knowledgeable medical expert.\n\n"
    "{context}\n\n"
    "Question:\n{question}\n\n"
    "{answer_choices}\n\n"
    "Let's think step by step. Provide multiple independent chains of reasoning for this question and determine the most consistent answer."
)
