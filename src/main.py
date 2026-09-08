import argparse
import sys
from src.rag.pipeline import AdvancedRAGPipeline
from src.agents.crew_manager import RepoMindCrew
from crewai_tools import LlamaIndexTool

def main():
    """Main CLI entry point for RepoMind."""
    parser = argparse.ArgumentParser(description="RepoMind: Advanced Multi-Agent RAG Codebase Assistant")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest and index a target codebase directory")
    ingest_parser.add_argument("--dir", type=str, required=True, help="Path to the target codebase directory")

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Query the codebase using multi-agent workflow and advanced RAG")
    query_parser.add_argument("--target-dir", type=str, required=True, help="Path to the target codebase directory")
    query_parser.add_argument("--prompt", type=str, required=True, help="Question or task regarding the codebase")

    args = parser.parse_args()

    if args.command == "ingest":
        print(f"[*] Initializing codebase ingestion for directory: {args.dir}")
        pipeline = AdvancedRAGPipeline(target_dir=args.dir)
        pipeline.ingest_codebase()
        print("[+] Ingestion completed successfully.")

    elif args.command == "query":
        print(f"[*] Initializing RepoMind pipeline for: {args.target_dir}")
        pipeline = AdvancedRAGPipeline(target_dir=args.target_dir)
        
        # 1. Get LlamaIndex query engine equipped with Cross-Encoder Reranker
        query_engine = pipeline.get_query_engine()
        
        # 2. Wrap LlamaIndex query engine into a CrewAI-compatible tool
        rag_tool = LlamaIndexTool.from_query_engine(
            query_engine=query_engine,
            name="Codebase RAG Search",
            description="Searches the indexed codebase via advanced RAG and cross-encoder reranking to retrieve accurate code snippets."
        )
        
        # 3. Initialize and execute multi-agent crew workflow
        print(f"[*] Running multi-agent analysis for prompt: '{args.prompt}'...")
        crew_manager = RepoMindCrew(tools=[rag_tool])
        result = crew_manager.run_analysis(query=args.prompt)
        
        print("\n" + "="*60)
        print(" FINAL MULTI-AGENT ANALYSIS RESULT ")
        print("="*60)
        print(result)

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()