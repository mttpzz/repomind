from crewai import Agent, Task, Crew, Process, LLM
from src.config import SETTINGS

class RepoMindCrew:
    def __init__(self, tools):
        self.tools = tools

    def run_analysis(self, query: str):
        # is_litellm=True forces routing through LiteLLM (instead of crewai's native Anthropic SDK)
        # so calls pick up the LITELLM_SUCCESS_CALLBACKS/LITELLM_FAILURE_CALLBACKS Langfuse tracing
        llm = LLM(model=SETTINGS["default_model"], is_litellm=True)

        # Agent 1: Code Analyst
        code_analyst = Agent(
            role='Senior Codebase Analyst',
            goal='Analyze the codebase and find the relevant files and functions for the user request.',
            backstory="You are a software engineer with decades of experience in analyzing complex enterprise architectures.",
            tools=self.tools,
            llm=llm,
            verbose=True,
            memory=True
        )

        # Agent 2: Technical Writer / Documentator
        tech_writer = Agent(
            role='Technical Documentation Specialist',
            goal='Synthesize the findings into a clear, clean, and well-structured technical explanation.',
            backstory="You excel at translating complex code logic into high-level documentation for developers.",
            llm=llm,
            verbose=True
        )

        # Tasks definition
        task_search = Task(
            description=f"Find and analyze the code sections relevant to this query: {query}",
            expected_output="A detailed report with relevant code snippets and their logical explanation.",
            agent=code_analyst
        )

        task_write = Task(
            description="Take the analyst's report and write a final, professional response ready for technical documentation.",
            expected_output="A perfectly formatted Markdown technical guide.",
            agent=tech_writer
        )

        # Crew assembly and execution
        crew = Crew(
            agents=[code_analyst, tech_writer],
            tasks=[task_search, task_write],
            process=Process.sequential,
            verbose=True,
            embedder={
                "provider": "ollama",
                "config": {"model": SETTINGS["embedding_model"]}
            }
        )

        return crew.kickoff()