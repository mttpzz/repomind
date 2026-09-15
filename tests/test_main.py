import pytest
from unittest.mock import MagicMock, patch

from src.main import main


@patch("src.main.AdvancedRAGPipeline")
def test_ingest_command_calls_pipeline(mock_pipeline_cls, monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "ingest", "--dir", "./some/codebase"])
    mock_pipeline = mock_pipeline_cls.return_value

    main()

    mock_pipeline_cls.assert_called_once_with(target_dir="./some/codebase")
    mock_pipeline.ingest_codebase.assert_called_once()


def test_ingest_command_requires_dir(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "ingest"])

    with pytest.raises(SystemExit):
        main()


@patch("src.main.RepoMindCrew")
@patch("src.main.LlamaIndexTool")
@patch("src.main.AdvancedRAGPipeline")
def test_query_command_wires_pipeline_into_crew(
    mock_pipeline_cls, mock_tool_cls, mock_crew_cls, monkeypatch, capsys
):
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "query", "--target-dir", "./some/codebase", "--prompt", "How does X work?"],
    )
    mock_pipeline = mock_pipeline_cls.return_value
    mock_query_engine = mock_pipeline.get_query_engine.return_value
    mock_rag_tool = mock_tool_cls.from_query_engine.return_value
    mock_crew_instance = mock_crew_cls.return_value
    mock_crew_instance.run_analysis.return_value = "final analysis"

    main()

    mock_pipeline_cls.assert_called_once_with(target_dir="./some/codebase")
    mock_tool_cls.from_query_engine.assert_called_once_with(
        query_engine=mock_query_engine,
        name="Codebase RAG Search",
        description=(
            "Searches the indexed codebase via advanced RAG and cross-encoder "
            "reranking to retrieve accurate code snippets."
        ),
    )
    mock_crew_cls.assert_called_once_with(tools=[mock_rag_tool])
    mock_crew_instance.run_analysis.assert_called_once_with(query="How does X work?")
    assert "final analysis" in capsys.readouterr().out


def test_query_command_requires_prompt(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "query", "--target-dir", "./some/codebase"])

    with pytest.raises(SystemExit):
        main()


def test_no_command_prints_help_and_exits(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["main.py"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert "usage" in capsys.readouterr().out.lower()
