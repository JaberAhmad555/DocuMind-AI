
"""Automated tests for the DocuMind AI RAG engine."""

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pypdf import PdfWriter
from langchain_core.documents import Document

from rag_engine import (
    get_identifier_pattern,
    remove_duplicates,
    prepare_sources,
    format_context,
    retrieve_documents,
    extract_pdf,
)


# ==========================================
# IDENTIFIER MATCHING TESTS
# ==========================================

@pytest.mark.parametrize(
    "course_code",
    ["CSE220", "CSE 220", "cse-220"]
)
def test_identifier_matching(course_code):

    pattern = get_identifier_pattern(
        "What is my grade in CSE220?"
    )

    assert pattern is not None
    assert pattern.search(course_code)


def test_identifier_not_found():

    pattern = get_identifier_pattern(
        "What is machine learning?"
    )

    assert pattern is None


def test_wrong_identifier():

    pattern = get_identifier_pattern("Find CSE220")

    assert not pattern.search("CSE221")
    assert not pattern.search("CSE2200")


# ==========================================
# DUPLICATE REMOVAL TESTS
# ==========================================

def test_remove_duplicates():

    doc1 = Document(
        page_content="Machine learning uses data.",
        metadata={"page": 1}
    )

    doc2 = Document(
        page_content="Machine learning uses data.",
        metadata={"page": 1}
    )

    doc3 = Document(
        page_content="Deep learning uses neural networks.",
        metadata={"page": 2}
    )

    result = remove_duplicates(
        [doc1, doc2, doc3]
    )

    assert len(result) == 2
    assert result[0] == doc1
    assert result[1] == doc3


def test_remove_duplicates_empty():

    assert remove_duplicates([]) == []


# ==========================================
# SOURCE FORMATTING TESTS
# ==========================================

def test_prepare_sources():

    documents = [
        Document(
            page_content="Artificial intelligence",
            metadata={"page": 3}
        )
    ]

    sources = prepare_sources(documents)

    assert len(sources) == 1
    assert sources[0]["page"] == 3
    assert sources[0]["text"] == "Artificial intelligence"


def test_format_context():

    sources = [
        {
            "page": 1,
            "text": "Introduction to AI"
        },
        {
            "page": 2,
            "text": "Machine learning concepts"
        }
    ]

    context = format_context(sources)

    assert "[p. 1]" in context
    assert "[p. 2]" in context
    assert "Introduction to AI" in context
    assert "Machine learning concepts" in context


def test_empty_context():

    assert format_context([]) == ""


# ==========================================
# HYBRID RETRIEVAL TESTS
# ==========================================

def test_hybrid_retrieval():

    vectorstore = MagicMock()

    semantic_doc = Document(
        page_content="CSE221 Algorithms",
        metadata={"page": 2}
    )

    retriever = vectorstore.as_retriever.return_value

    retriever.invoke.return_value = [
        semantic_doc
    ]

    vectorstore.get.return_value = {
        "documents": [
            "CSE220 Data Structures Grade A",
            "CSE221 Algorithms"
        ],
        "metadatas": [
            {"page": 1},
            {"page": 2}
        ]
    }

    results = retrieve_documents(
        vectorstore=vectorstore,
        question="What is the grade in CSE220?",
        chunk_count=2
    )

    assert len(results) == 2

    assert "CSE220" in results[0].page_content
    assert "Grade A" in results[0].page_content


def test_semantic_retrieval():

    vectorstore = MagicMock()

    document = Document(
        page_content="Machine learning learns from data.",
        metadata={"page": 1}
    )

    retriever = vectorstore.as_retriever.return_value

    retriever.invoke.return_value = [
        document
    ]

    result = retrieve_documents(
        vectorstore=vectorstore,
        question="What is machine learning?",
        chunk_count=1
    )

    assert len(result) == 1
    assert result[0] == document

    vectorstore.get.assert_not_called()


def test_empty_vectorstore():

    vectorstore = MagicMock()

    result = retrieve_documents(
        vectorstore=vectorstore,
        question="Test question",
        chunk_count=0
    )

    assert result == []

    vectorstore.as_retriever.assert_not_called()


# ==========================================
# PDF EXTRACTION TESTS
# ==========================================

def test_sample_pdf_extraction():

    pdf_path = (
        Path(__file__).resolve().parents[1]
        / "Intro_to_AI_Transcript.pdf"
    )

    documents, page_count = extract_pdf(
        pdf_path.read_bytes(),
        pdf_path.name
    )

    assert page_count == 6
    assert len(documents) > 0

    assert all(
        document.page_content.strip()
        for document in documents
    )

    assert documents[0].metadata["page"] == 1


def test_empty_pdf():

    writer = PdfWriter()

    writer.add_blank_page(
        width=612,
        height=792
    )

    buffer = BytesIO()

    writer.write(buffer)

    with pytest.raises(
        ValueError,
        match="No readable text found"
    ):

        extract_pdf(
            buffer.getvalue(),
            "empty.pdf"
        )
