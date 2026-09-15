from unittest.mock import MagicMock, patch

from src.agents.crew_manager import RepoMindCrew
from src.config import SETTINGS


@patch("src.agents.crew_manager.Crew")
@patch("src.agents.crew_manager.Task")
@patch("src.agents.crew_manager.Agent")
@patch("src.agents.crew_manager.LLM")
def test_run_analysis_forces_litellm_routing(mock_llm_cls, mock_agent_cls, mock_task_cls, mock_crew_cls):
    """LLM must be built with is_litellm=True so calls go through LiteLLM (and its Langfuse
    callbacks) instead of crewai's native Anthropic SDK, which bypasses tracing entirely."""
    mock_llm_instance = MagicMock()
    mock_llm_cls.return_value = mock_llm_instance
    mock_crew_cls.return_value.kickoff.return_value = "final result"

    crew_manager = RepoMindCrew(tools=[])
    result = crew_manager.run_analysis(query="test query")

    mock_llm_cls.assert_called_once_with(model=SETTINGS["default_model"], is_litellm=True)
    assert result == "final result"


@patch("src.agents.crew_manager.Crew")
@patch("src.agents.crew_manager.Task")
@patch("src.agents.crew_manager.Agent")
@patch("src.agents.crew_manager.LLM")
def test_run_analysis_passes_llm_to_both_agents(mock_llm_cls, mock_agent_cls, mock_task_cls, mock_crew_cls):
    """Both agents (code_analyst and tech_writer) must share the explicit llm= — without it,
    crewai falls back to an OpenAI default and crashes with no OPENAI_API_KEY set."""
    mock_llm_instance = MagicMock()
    mock_llm_cls.return_value = mock_llm_instance

    RepoMindCrew(tools=[]).run_analysis(query="test query")

    assert mock_agent_cls.call_count == 2
    for call in mock_agent_cls.call_args_list:
        assert call.kwargs["llm"] is mock_llm_instance


@patch("src.agents.crew_manager.Crew")
@patch("src.agents.crew_manager.Task")
@patch("src.agents.crew_manager.Agent")
@patch("src.agents.crew_manager.LLM")
def test_run_analysis_configures_ollama_embedder(mock_llm_cls, mock_agent_cls, mock_task_cls, mock_crew_cls):
    """Crew's memory embedder must point at the local Ollama model (bare name, no provider
    prefix), not crewai's default OpenAI embedder which also requires OPENAI_API_KEY."""
    RepoMindCrew(tools=[]).run_analysis(query="test query")

    _, crew_kwargs = mock_crew_cls.call_args
    assert crew_kwargs["embedder"] == {
        "provider": "ollama",
        "config": {"model": SETTINGS["embedding_model"]},
    }


@patch("src.agents.crew_manager.Crew")
@patch("src.agents.crew_manager.Task")
@patch("src.agents.crew_manager.Agent")
@patch("src.agents.crew_manager.LLM")
def test_run_analysis_returns_kickoff_result(mock_llm_cls, mock_agent_cls, mock_task_cls, mock_crew_cls):
    mock_crew_cls.return_value.kickoff.return_value = "analysis output"

    result = RepoMindCrew(tools=["some_tool"]).run_analysis(query="how does auth work?")

    mock_crew_cls.return_value.kickoff.assert_called_once()
    assert result == "analysis output"
