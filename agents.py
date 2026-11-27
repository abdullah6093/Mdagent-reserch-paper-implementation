"""
Agent representation for MDAgents.
Provides conversation memory, role-grounded prompts, and multimodal chat.
"""

from typing import List, Dict, Optional, Any, Union
from pathlib import Path

from llm import chat
from prompts import AGENT_SYSTEM_PROMPT
from config import DEFAULT_TEMPERATURE, DEFAULT_MODEL

class Agent:
    """An expert medical agent with memory, role, and chat capabilities."""

    def __init__(
        self,
        role: str,
        description: str,
        model: str = DEFAULT_MODEL,
        examplers: Optional[List[Dict[str, Any]]] = None,
        custom_system_prompt: Optional[str] = None
    ):
        self.role = role
        self.description = description
        self.model = model
        self.memory: List[Dict[str, Any]] = []
        self.current_answer: Optional[str] = None

        if custom_system_prompt:
            system_content = custom_system_prompt
        else:
            system_content = AGENT_SYSTEM_PROMPT.format(role=self.role, description=self.description)

        self.memory.append({"role": "system", "content": system_content})

        if examplers:
            for ex in examplers:
                self.memory.append({"role": "user", "content": ex["question"]})
                self.memory.append({"role": "assistant", "content": f"{ex['answer']}\n\n{ex.get('reason', '')}"})

    def chat(
        self,
        message: str,
        temperature: float = DEFAULT_TEMPERATURE,
        images: Optional[List[Union[str, Path]]] = None
    ) -> str:
        self.memory.append({"role": "user", "content": message})
        response = chat(
            messages=self.memory,
            model=self.model,
            temperature=temperature,
            images=images
        )
        self.memory.append({"role": "assistant", "content": response})
        return response

    def reset_memory(self, keep_system: bool = True):
        if keep_system and self.memory:
            self.memory = [self.memory[0]]
        else:
            self.memory = []

class AgentTeam:
    """A multidisciplinary or integrated care team with a lead and assisting clinicians."""

    def __init__(
        self,
        name: str,
        goal: str,
        members: List[Agent],
        lead: Optional[Agent] = None
    ):
        self.name = name
        self.goal = goal
        self.members = members
        if lead:
            self.lead = lead
        elif members:
            self.lead = members[0]
        else:
            self.lead = None
        self.assistants = [m for m in self.members if m != self.lead]

    def synthesize_team_report(
        self,
        question: str,
        prior_reports: str = ""
    ) -> str:
        if not self.lead:
            return "No lead clinician available."

        directive_prompt = (
            f"You are the lead of {self.name}. Team goal: {self.goal}.\n"
            f"Prior reports from earlier teams:\n{prior_reports}\n\n"
            f"Medical query: {question}\n\n"
            f"Provide specific clinical investigation questions for your assistant clinicians:\n"
            + "\n".join([f"- {a.role}" for a in self.assistants])
        )
        directive = self.lead.chat(directive_prompt)

        investigations = []
        for assist in self.assistants:
            assist_prompt = (
                f"You are in {self.name}. Your lead gave the following directions:\n"
                f"{directive}\n\n"
                f"Medical query:\n{question}\n\n"
                f"Please apply your expertise ({assist.role}) and provide your concise findings."
            )
            findings = assist.chat(assist_prompt)
            investigations.append(f"[{assist.role} Findings]:\n{findings}")

        gathered_str = "\n\n".join(investigations)

        synthesis_prompt = (
            f"You are the lead of {self.name}.\n"
            f"The investigations gathered from your team are as follows:\n\n"
            f"{gathered_str}\n\n"
            f"Question under evaluation: {question}\n\n"
            f"Synthesize these findings into a unified, rigorous medical report."
        )
        team_report = self.lead.chat(synthesis_prompt)
        return team_report
