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
    
    # Analyze document command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze document extraction and chunking')
    analyze_parser.add_argument('file_path', help='Path to document file')
    
    # Remove document command
    remove_parser = subparsers.add_parser('remove', help='Remove document from knowledge base')
    remove_parser.add_argument('file_path', help='Path to document file to remove')
    
    # List documents command
    list_parser = subparsers.add_parser('list', help='List all documents in knowledge base')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear all documents from knowledge base')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Search knowledge base')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--results', '-r', type=int, default=15, 
                             help='Number of context chunks to retrieve (default: 15)')
    query_parser.add_argument('--model', '-m', type=str, default='auto',
                             help='LLM model to use (default: auto-select based on resources)')
    query_parser.add_argument('--debug', '-d', action='store_true',
                             help='Enable debug mode to show retrieved chunks')
    query_parser.add_argument('--timeout', '-t', type=int, default=600,
                             help='Request timeout in seconds (default: 600)')
    
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
    
    elif args.command == 'analyze':
        if not os.path.exists(args.file_path):
            print(f"Error: File {args.file_path} not found")
            sys.exit(1)
        
        try:
            rag.analyze_document(args.file_path)
        except Exception as e:
            print(f"Error analyzing document: {e}")
            sys.exit(1)
    
    elif args.command == 'remove':
        try:
            rag.remove_document(args.file_path)
        except Exception as e:
            print(f"Error removing document: {e}")
            sys.exit(1)
    
    elif args.command == 'list':
        try:
            rag.show_documents()
        except Exception as e:
            print(f"Error listing documents: {e}")
            sys.exit(1)
    
    elif args.command == 'clear':
        try:
            response = input("Are you sure you want to clear all documents? (y/N): ")
            if response.lower() in ['y', 'yes']:
                rag.clear_knowledge_base()
            else:
                print("Operation cancelled.")
        except Exception as e:
            print(f"Error clearing knowledge base: {e}")
            sys.exit(1)
    
    elif args.command == 'query':
        try:
            result = rag.query(args.question, args.results, debug=args.debug, timeout=args.timeout)
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
                    print("  add <file_path>    - Add document to knowledge base")
                    print("  remove <file_path> - Remove document from knowledge base")
                    print("  list               - Show all documents in knowledge base")
                    print("  clear_docs         - Clear all documents (with confirmation)")
                    print("  reset              - Clear conversation history")
                    print("  history            - Show conversation history")
                    print("  <question>         - Ask a question")
                    print("  quit/exit          - Exit interactive mode")
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
                
                if user_input.lower() == 'reset':
                    rag.reset_conversation()
                    continue
                
                if user_input.lower() == 'history':
                    rag.show_conversation()
                    continue
                
                if user_input.startswith('remove '):
                    file_path = user_input[7:].strip()
                    try:
                        rag.remove_document(file_path)
                    except Exception as e:
                        print(f"Error: {e}")
                    continue
                
                if user_input.lower() == 'list':
                    try:
                        rag.show_documents()
                    except Exception as e:
                        print(f"Error: {e}")
                    continue
                
                if user_input.lower() == 'clear_docs':
                    try:
                        response = input("Are you sure you want to clear all documents? (y/N): ")
                        if response.lower() in ['y', 'yes']:
                            rag.clear_knowledge_base()
                        else:
                            print("Operation cancelled.")
                    except Exception as e:
                        print(f"Error: {e}")
                    continue
                
                if not user_input:
                    continue
                
                # Treat as query
                try:
                    result = rag.query(user_input, use_conversation=True, timeout=600)
                    print(f"\nAnswer: {result['answer']}")
                    print(f"Sources: {', '.join(set(result['sources']))}")
                    if result['conversation_mode']:
                        conv_info = rag.get_conversation_info()
                        print(f"Conversation: {conv_info['history_length']}/{conv_info['memory_capacity']} exchanges")
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
