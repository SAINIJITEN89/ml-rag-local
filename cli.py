#!/usr/bin/env python3

import argparse
import os
import sys
from rag_system import RAGSystem


def main():
    parser = argparse.ArgumentParser(description="RAG-based Local Knowledge Search")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add document command
    add_parser = subparsers.add_parser('add', help='Add document to knowledge base')
    add_parser.add_argument('file_path', help='Path to document file')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Search knowledge base')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--results', '-r', type=int, default=5, 
                             help='Number of context chunks to retrieve (default: 5)')
    query_parser.add_argument('--model', '-m', type=str, default='auto',
                             help='LLM model to use (default: auto-select based on resources)')
    
    # Interactive mode
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize RAG system with appropriate model
    model = getattr(args, 'model', 'auto')
    rag = RAGSystem(llm_model=model)
    
    if args.command == 'add':
        if not os.path.exists(args.file_path):
            print(f"Error: File {args.file_path} not found")
            sys.exit(1)
        
        try:
            rag.add_document(args.file_path)
            print(f"Successfully added {args.file_path} to knowledge base")
        except Exception as e:
            print(f"Error adding document: {e}")
            sys.exit(1)
    
    elif args.command == 'query':
        try:
            result = rag.query(args.question, args.results)
            print(f"Question: {result['question']}")
            print(f"Answer: {result['answer']}")
            print(f"\nSources ({result['context_chunks']} chunks):")
            for source in set(result['sources']):
                print(f"  - {source}")
        except Exception as e:
            print(f"Error querying: {e}")
            sys.exit(1)
    
    elif args.command == 'interactive':
        print("RAG Knowledge Search - Interactive Mode")
        print("Type 'quit' or 'exit' to leave, 'help' for commands")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("Commands:")
                    print("  add <file_path>  - Add document to knowledge base")
                    print("  <question>       - Ask a question")
                    print("  quit/exit        - Exit interactive mode")
                    continue
                
                if user_input.startswith('add '):
                    file_path = user_input[4:].strip()
                    if not os.path.exists(file_path):
                        print(f"Error: File {file_path} not found")
                        continue
                    
                    try:
                        rag.add_document(file_path)
                        print(f"Successfully added {file_path}")
                    except Exception as e:
                        print(f"Error: {e}")
                    continue
                
                if not user_input:
                    continue
                
                # Treat as query
                try:
                    result = rag.query(user_input)
                    print(f"\nAnswer: {result['answer']}")
                    print(f"Sources: {', '.join(set(result['sources']))}")
                except Exception as e:
                    print(f"Error: {e}")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break


if __name__ == "__main__":
    main()