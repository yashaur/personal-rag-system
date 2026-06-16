import fitz
from langchain_core.documents import Document
from pathlib import Path

def _load_pdf(path: str) -> list[Document]:
    with fitz.open(path) as docs:
        pages = []

        for page in docs:
            
            metadata = {
                        'source': Path(path).name,
                        'page_number': page.number + 1,
            }

            page_content = (
                            page.get_text()
                                .replace('•', '-')
                                # .replace('\n', '')
                                # .replace('\n\n', '')
                                # .replace('\t', ' ')
                                # .replace('\t\t', ' ')
                                .strip()
                            )
            if page_content:
                doc = Document(page_content = page_content, metadata = metadata)
                pages.append(doc)

    return pages


def _load_txt(path: str) -> list[Document]:

    text = Path(path).read_text(encoding="utf-8")

    return [Document(page_content = text, metadata = {'source': Path(path).name})]


def load_document(path: str) -> list[Document]:

    extension = Path(path).suffix.lower()

    LOADERS = {
                        '.txt': _load_txt,
                        '.pdf': _load_pdf
                        # More to be added later
    }

    if extension not in LOADERS:
        raise ValueError(f"Unsupported file format: {extension}. Please ensure document is one of these formats: {list(LOADERS.keys())}")
    else:
        load_func = LOADERS[extension]
        docs = load_func(path)
        return docs    

if __name__ == '__main__':
    path = './data/uploads/test.txt'
    docs = load_document(path)
    print(docs)
    print(len(docs))