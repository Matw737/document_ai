# Document AI

Prosty POC do przetwarzania dokumentów PDF.

### Input
Program przyjmuje plik PDF:

- PDF z warstwą tekstową
- skan / PDF bez warstwy tekstowej
- dokument mieszany

Uruchomienie:

```bash
python main.py path/to/document.pdf --output output/result.json

document_ai/
├── main.py
├── models/       # model OCR
├── ocr/          # OCR i postprocessing
├── pipeline/     # przetwarzanie dokumentu
├── output/       # zapis wyników
├── tests/
└── temp/         # pliki tymczasowe

