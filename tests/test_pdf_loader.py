from src.pdf_loader import PDFLoader


def test_pdf_extraction():

    loader = PDFLoader(
        "data/pdfs/deeplearningfordummies.pdf"
    )

    text = loader.extract_text()

    assert len(text) > 0