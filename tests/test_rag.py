import pytest
from pathlib import Path
from src.rag.pipeline import AdvancedRAGPipeline

def test_pipeline_initialization(tmp_path):
    """Test that the AdvancedRAGPipeline initializes correctly with a target directory."""
    # Create a temporary test directory
    test_dir = tmp_path / "test_codebase"
    test_dir.mkdir()
    
    # Initialize the RAG pipeline
    pipeline = AdvancedRAGPipeline(target_dir=str(test_dir))
    
    # Assertions to verify correct setup
    assert pipeline.target_dir == test_dir
    assert isinstance(pipeline.target_dir, Path)

def test_pipeline_missing_directory():
    """Test that initializing the pipeline with a non-existent directory works (ingest will fail later)."""
    non_existent_path = "./non_existent_codebase_dir_12345"
    pipeline = AdvancedRAGPipeline(target_dir=non_existent_path)
    
    assert pipeline.target_dir == Path(non_existent_path)