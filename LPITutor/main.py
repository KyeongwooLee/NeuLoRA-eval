from __future__ import annotations

from document_processing import DocumentProcessor
from response_generation import ResponseGenerator
from user_view import QueryProcessor


FIXED_LEVEL = "intermediate"


def main() -> None:
    processor = DocumentProcessor(pdf_dir="./pdf_files")
    doc_count = processor.upload_to_vector_db()
    print(f"Indexed documents into Chroma: {doc_count}")

    query_processor = QueryProcessor()
    user_query = input("Please enter your question: ")
    processed_query_data = query_processor.process_query(user_query, FIXED_LEVEL)

    responder = ResponseGenerator()
    response = responder.respond_to_user(
        query=processed_query_data["query"],
        level=FIXED_LEVEL,
    )

    print("\nIntelligent Tutor Response:")
    print(response)


if __name__ == "__main__":
    main()
