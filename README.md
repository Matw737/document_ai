1. Project Overview

Document AI is a personal learning project focused on building an OCR system and understanding the process of training and using deep learning models.

The main goal was to explore sequential learning and gain practical experience with its implementation.

The original idea was to build a small OCR model that could recognize text with limited computational resources. The initial design also included an additional step: documents or text regions with low recognition confidence would be passed to an LLM for further processing.

This approach was intended to keep the initial processing stage as lightweight as possible and use larger language models only when needed.

During development, the third version of the OCR model showed promising results. Because of this, the LLM escalation module was not implemented.

The project is still under development. Further improvements, including model fine-tuning, are planned.

2. Project Goals and Design Decisions

The OCR project started as an extension of a larger system concept called AI Inbox.

The original system was designed to read incoming emails, classify their intent using an embedding model combined with a GRU network, and route them to the appropriate department. The routing process was intended to use prediction confidence, with low-confidence cases sent for human review.

The OCR module was planned as the next part of this workflow. Its purpose was to process email attachments, extract their content, and make it available to other system components. An additional idea was to use an LLM to generate document summaries.

Over time, the OCR module became a separate project and received more development time than originally planned.

Model Architecture

The architecture was based on experiments from my previous CNN projects.

I decided to use two CNN branches with different receptive fields. The idea was to capture different types of visual features and provide the model with a broader view of the input. Based on my earlier experiments, I expected this approach to help distinguish visually similar characters.

For sequence processing, I chose a bidirectional GRU (BiGRU). The main reason was to balance model capacity and the number of parameters.

Later, I added an attention mechanism to give the model more context when processing text sequences.

Main Learning Goal

The main goal was to build a complete OCR pipeline and apply knowledge gained from previous deep learning experiments.

The project was not focused only on training a model. It also involved working with document processing, text detection, recognition, decoding, and postprocessing.

3. Features

### Document Input and Processing

The system accepts PDF and DOCX documents. At the beginning, it checks the document type and selects the appropriate processing method.

* **PDF with a text layer:** Text is extracted directly.
* **Scanned PDF:** DBNet detects individual words and returns their bounding boxes. The detected regions are then grouped into lines and passed to the OCR model.
* **DOCX:** Text is extracted from document paragraphs. Tables are not currently supported.

### OCR Recognition

The project uses a custom OCR model developed in PyTorch.

In earlier versions, the recognizer processed individual word crops returned by DBNet.

In version 3.0, I changed this approach. Instead of passing individual words, the system groups detected words into lines and passes larger image regions to the recognizer.

This gives the model more visual context and helps improve character recognition, especially letter capitalization.

### Text Postprocessing

After recognition, the predicted text goes through several correction steps:

* **Digit correction:** Handles common OCR mistakes, such as confusing `O` with `0` or `I` with `1`.
* **Capitalization correction:** Uses less ambiguous characters to estimate whether a word should use lowercase or uppercase letters.
* **Dictionary-based correction:** Uses Levenshtein distance to find possible spelling mistakes. Corrections are applied only when the distance is equal to 1.

These steps are designed to correct common recognition errors while avoiding unnecessary changes to the original text.

### Output

The system produces structured output organized by document pages.

The initial output format contained only the page number and extracted text. Later, I extended it with spatial information to preserve part of the original document layout.

For OCR results, each text line can contain:

* Recognized text.
* Bounding box coordinates (`bbox`).
* Vertical center position (`center_y`).
* Line height.
* Vertical spacing between consecutive lines (`gap_after` and `gap_ratio`).

This information can help identify paragraph boundaries and preserve parts of the original document structure.

It can also be useful in later stages of text processing, especially when preparing documents for text chunking in a RAG pipeline.


4. Architecture

### OCR Processing

Scanned PDF documents require additional processing because they do not contain an extractable text layer.

The processing steps are:

1. **Text Detection:** DBNet detects individual words and returns their bounding boxes.
2. **Line Grouping:** The detected word regions are grouped into larger crops representing text lines.
3. **Temporary Storage:** The generated image crops are saved as temporary files instead of keeping all images in RAM.
4. **Data Loading:** `OCRInferenceDataset` and PyTorch `DataLoader` load the images in batches.
5. **Text Recognition:** The OCR model processes the batches on the available device (GPU or CPU).
6. **Cleanup:** Temporary image files are removed after processing.

### Memory Management

One of the design decisions was to avoid loading all generated image crops into RAM at once.

Instead, image crops are temporarily stored on disk. The dataset and DataLoader load them when needed and prepare batches for the recognition model.

This approach separates image storage from model execution and helps control memory usage, especially when processing larger documents.

### Model Initialization

The DBNet detector and OCR recognizer are initialized once when the application starts.

The same model instances are reused when processing all pages of a document. This avoids loading the models again for every page.

### Text Line Cropping

DBNet detects individual words and returns their bounding boxes.

To provide more context for the OCR model, detected words are grouped into text lines. The crop boundaries are extended based on the first detected word in each line, with an additional margin around the text.

6. Processing Pipeline

The application uses a document router to select the appropriate processing method based on the input file.

### Document Input

The user provides a PDF or DOCX file through the command line.

The application checks whether the input file exists and initializes the OCR models.

### Document Routing

The document is passed to the document router, which selects the processing method.

There are three main processing paths:

### PDF with a text layer

Text is extracted directly from the PDF.
OCR recognition is not required.
The extracted text is passed to the output processing stage.

### Scanned PDF

DBNet detects individual words on each page.
Bounding boxes are used to group detected words into text lines.
Larger image crops are created from the original page.
Crops are temporarily saved on disk.
The dataset and DataLoader load images in batches.
The OCR recognizer processes the batches using the available device (GPU or CPU).

### DOCX

Text is extracted from document paragraphs.
Empty paragraphs are skipped.
The extracted text is passed to the output processing stage.
### Text Postprocessing

The extracted or recognized text is passed to the postprocessing module.

This stage includes:

Character correction.
Capitalization correction.
Dictionary-based spelling correction.
Grouping text regions into lines.
Collecting spatial information from OCR bounding boxes.
### Output Generation

The processed results are saved as a JSON file.

The output is organized by document pages and contains extracted or recognized text. For OCR results, additional spatial information can also be included.

Temporary image crops are removed after processing.

## 6. OCR Model

The OCR model went through several development stages. The first version was designed to be lightweight and relatively simple. Later versions introduced a more complex CNN architecture, a larger BiGRU layer, and an attention mechanism.

The current version is called **OCR v3.0**.

### 6.1 Model Overview

The model combines three main components:

* A custom CNN feature extractor based on residual blocks with two parallel convolution branches.
* A bidirectional GRU (BiGRU) for sequence processing.
* A Multi-Head Attention layer to provide additional context between sequence positions.

The model processes an image of a text line and converts its visual features into a sequence of character predictions.

The input image has a fixed height of 32 pixels, while its width depends on the aspect ratio of the original text line.

### 6.2 CNN Feature Extractor

The CNN is based on a custom architecture called `DualKernelResidualCNN2`.

The main idea behind this architecture is to use two parallel convolution branches with different receptive fields.

Each block contains:

* An initial convolution that projects the input into an internal feature space.
* A standard convolution branch using 3×3 kernels.
* A dilated convolution branch using 3×3 kernels with a dilation rate of 2.
* Residual connections inside both branches.
* A gating mechanism that controls the contribution of each branch.
* Feature concatenation followed by an activation function.

The two branches are designed to capture different types of visual information.

The standard branch focuses on local features, while the dilated branch has a wider receptive field. A 3×3 kernel with dilation 2 covers an effective 5×5 area of the input feature map.

The idea was to combine detailed local features with information from a wider area without simply increasing the size of the convolution kernels.

### 6.3 Channel Reduction and Feature Gating

The architecture also uses different channel configurations inside its blocks.

Depending on the `is_expanded` parameter, the internal channel count is either equal to `branch_channels` or reduced by half.

For example, when `branch_channels` is 64 and `is_expanded` is set to `False`, each branch processes 32 channels. After concatenation, the block produces 64 output channels.

This approach allows the model to use two parallel branches without unnecessarily increasing the number of parameters.

The gating mechanism operates on the shared feature representation and produces separate gates for both branches. These gates are applied to the branch outputs before concatenation.

The intention is to allow the network to adjust the contribution of each branch instead of treating both feature representations equally.

### 6.4 BiGRU Sequence Processing

After the CNN feature extraction stage, the resulting feature map is converted into a sequence.

The CNN reduces the spatial height from 32 to 8 pixels and reduces the width by a factor of four.

The feature map is then rearranged so that each position along the width becomes one sequence step.

The features from the remaining height and channel dimensions are combined into a single feature vector.

The resulting sequence is passed to a bidirectional GRU.

The current model uses:

* Input size: 512
* Hidden size: 256
* Bidirectional processing: enabled

The BiGRU processes the sequence in both directions. This allows the model to use information from both earlier and later positions when generating its character representations.

The choice of GRU was based on the intention to balance sequence modelling capacity and model size.

Packed sequences are used to handle different input lengths without processing unnecessary padded positions.

### 6.5 Multi-Head Attention

The current version also includes a Multi-Head Attention layer.

The attention layer uses:

* Embedding dimension: 512
* Number of attention heads: 4

It is applied after the BiGRU.

The main intention was to give the model an additional way to use information from different positions in the text sequence.

For example, recognizing a character may depend not only on its immediate visual features but also on the surrounding characters and the broader context of the text line.

The attention output is added to the original BiGRU output through a residual connection and then passed through Layer Normalization.

A padding mask is used to prevent the attention mechanism from treating padded sequence positions as valid input.

### 6.6 Output Layer

The final representation is passed through a linear layer that maps the 512-dimensional feature vectors to the model's character classes.

The current output layer contains 118 classes.

The model also calculates the output sequence lengths based on the width reduction performed by the CNN. This information is needed to handle variable-length inputs during training and decoding.

### 6.7 Model Size

The current OCR v3.0 model contains approximately **2,996,374 trainable parameters**.

The first version was much smaller, with approximately 1.7 million parameters.

The increase in model size came mainly from expanding the BiGRU hidden dimension from 128 to 256 and introducing the attention mechanism.

The goal was to increase the model's ability to process more complex visual and sequential information while keeping the overall architecture relatively compact.

### 6.8 Architecture Note

One implementation detail is worth mentioning.

My original intention was to use a 1×1 convolution for the initial channel projection in the CNN blocks. However, the implemented version uses a 3×3 kernel instead.

I noticed this difference only after training the model.

Since the current architecture produces promising results, I decided to keep it unchanged rather than modify the model and retrain it.

This means that the initial convolution also performs local feature extraction instead of only projecting the input into a higher-dimensional feature space.


## 7. Text Decoding and Dictionary Correction

The OCR model produces a sequence of character predictions. These predictions need to be converted into readable text and processed to reduce common OCR errors.

The project uses a greedy CTC decoder followed by several text correction steps.

### 7.1 Greedy CTC Decoding

The decoder uses a greedy approach.

For each output position, it selects the class with the highest predicted value using `argmax`.

The resulting sequence is then processed by:

* Removing consecutive repeated predictions.
* Removing blank tokens.
* Converting the remaining class indices into characters using the model alphabet.

This produces the initial text output.

Greedy decoding is relatively simple and computationally efficient. However, it does not explore alternative character sequences when the model is uncertain.

### 7.2 Rule-Based Text Correction

After decoding, the text passes through several correction functions.

The first stage handles common character recognition errors.

It includes:

* Correcting confusion between visually similar letters and digits, such as `O/0` and `I/1`.
* Detecting incorrect capitalization.
* Handling selected OCR-specific character errors, such as `9` recognized instead of `g`.
* Correcting the paragraph symbol when `§` is followed by an incorrectly recognized character.

Some rules use the surrounding characters to decide whether a correction is safe.

### 7.3 Dictionary-Based Correction

The project also includes a dictionary-based correction module.

The dictionary is built from clean text extracted from training documents. Instead of storing only a list of words, it also keeps additional information, such as word frequency and capitalization patterns.

The correction process uses Levenshtein distance to find words that are similar to an incorrectly recognized token.

The main design principle is conservative correction: it is better to leave an uncertain word unchanged than to replace a correct word with an incorrect one.

To reduce unwanted changes, the module uses several restrictions:

* Tokens containing digits are not processed by dictionary correction.
* Short abbreviations written in uppercase are protected.
* Capitalized words are checked against capitalization information from the training data.
* Ambiguous candidates are rejected.
* Word frequency is used to distinguish between possible candidates.
* Some valid word endings and inflection differences are protected from correction.
* Compound words separated by hyphens or slashes can be processed in separate parts.

The dictionary is optional. If the dictionary file is not available, the decoder can still process text using the other correction rules.

### 7.4 Sentence-Level Capitalization

Some capitalization errors cannot be reliably corrected by looking at individual words.

For this reason, the project also includes a sentence-level correction function.

It analyses the surrounding text and attempts to correct words that are incorrectly capitalized in the middle of a sentence.

The implementation includes additional protection for abbreviations, proper names, and defined terms commonly used in formal documents.

### 7.5 AI-Assisted Development

The decoding and dictionary correction modules were developed with significant assistance from Large Language Models (LLMs).

The modules contain more advanced logic than I could implement independently at the time of development. They were introduced to improve the readability of OCR output and address errors encountered during testing.

I consider these modules an important part of the project, but also an area for further learning. I plan to study their implementation in detail and gain a deeper understanding of the algorithms and design decisions behind them.

## 8. Project Structure

The project is divided into separate modules to keep document processing, text detection, OCR recognition, and output generation independent.

```text
document_ai/
│
├── main.py
├── requirements.txt
├── README.md
│
├── detector/
│   └── dbnet.py
│
├── models/
│   ├── ocr_v2.1_best.pt
│   └── ocr_v3.0_best.pt
│
├── ocr/
│   ├── alphabet.py
│   ├── confidence.py
│   ├── dataset.py
│   ├── decoder.py
│   ├── dict_correct.py
│   ├── model.py
│   ├── recognizer.py
│   └── slownik.json
│
├── pipeline/
│   ├── decision.py
│   ├── docx_processor.py
│   ├── pdf_processor.py
│   ├── postprocessor.py
│   ├── router.py
│   ├── scan_processor.py
│   ├── scan_processor_by_line.py
│   └── text_processor.py
│
├── output/
│   └── output.py
│
├── schemas/
│   └── output.py
│
├── utils/
│   └── image.py
│
└── tests/
    ├── test_detector.py
    ├── test_ocr.py
    └── test_pipeline.py
```

### Main Components

* **`main.py`** – Application entry point. Handles command-line arguments, initializes the models and starts document processing.

* **`detector/`** – Contains the DBNet text detection module responsible for locating text regions in scanned documents.

* **`models/`** – Stores trained OCR model checkpoints. The repository currently includes versions 2.1 and 3.0.

* **`ocr/`** – Contains the character recognition system:

  * `model.py` – OCR neural network architecture.
  * `dataset.py` – Dataset and DataLoader implementation.
  * `recognizer.py` – Handles model initialization and inference.
  * `alphabet.py` – Character set and character-to-index mappings.
  * `decoder.py` – Greedy CTC decoding and text correction.
  * `dict_correct.py` – Dictionary-based correction logic.
  * `slownik.json` – Dictionary data used during text correction.
  * `confidence.py` – Confidence-related functionality.

* **`pipeline/`** – Contains the document processing workflow:

  * `router.py` – Selects the appropriate processing path based on document type.
  * `pdf_processor.py` – Handles PDF processing and text extraction.
  * `docx_processor.py` – Extracts text from DOCX documents.
  * `scan_processor.py` and `scan_processor_by_line.py` – Handle scanned document processing and OCR region preparation.
  * `text_processor.py` – Processes extracted text.
  * `postprocessor.py` – Applies text corrections and collects additional information.
  * `decision.py` – Contains processing decision logic.

* **`output/`** – Handles formatting and saving processing results.

* **`schemas/`** – Defines the data structures used for document processing results.

* **`utils/`** – Contains shared utility functions, including image processing helpers.

* **`tests/`** – Contains tests for the detector, OCR recognizer, and document processing pipeline.

### Additional Files

* **`requirements.txt`** – Lists the Python dependencies required by the project.
* **`output*.json`** – Example processing outputs generated during development.
* **`temp/`** – Temporary directory used to store image crops during scanned document processing. Its contents are generated at runtime and are not part of the application source code.

## 9. Installation

### Requirements

The project was developed and tested in the following environment:

* Python 3.12.5
* PyTorch 2.12.0
* torchvision 0.27.0
* NumPy 2.2.6

For GPU acceleration, an NVIDIA GPU with a compatible driver and CUDA-enabled PyTorch installation is required.

The project can also run on a CPU, although OCR inference may be slower.

### 1. Clone the Repository

```bash
git clone https://github.com/Matw737/document_ai.git
cd document_ai
```

### 2. Create a Virtual Environment

It is recommended to use a separate virtual environment for the project.

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch

PyTorch and torchvision are installed separately from the other dependencies to allow choosing between CPU and GPU versions.

**CPU installation:**

```bash
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU (CUDA 12.6):**

```bash
pip install torch==2.12.0+cu126 torchvision==0.27.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

The CUDA installation requires a compatible NVIDIA driver.

### 4. Install Project Dependencies

After installing PyTorch, install the remaining dependencies:

```bash
pip install -r requirements.txt
```

### 5. Verify the Installation

To check whether PyTorch can access the GPU, run:

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

If CUDA is not available, the application automatically uses the CPU.

## 10. Usage

### Basic Usage

The application is started from the project root directory using `main.py`.

```bash
python main.py path/to/document.pdf
```

The application currently supports two input formats:

* PDF
* DOCX

You can provide either a PDF document with an existing text layer or a scanned PDF document.

### Specifying the Output Path

By default, the processing results are saved to `output.json` in the project root directory.

You can also specify a custom output path using the `--output` argument:

```bash
python main.py path/to/document.pdf --output results/document.json
```

The application creates the output directory if it does not already exist.

### Examples

**Process a PDF document:**

```bash
python main.py documents/sample.pdf
```

**Process a DOCX document:**

```bash
python main.py documents/sample.docx
```

**Save results to a custom location:**

```bash
python main.py documents/scanned_document.pdf --output results/scanned_document.json
```

### Output

After processing, the application:

1. Prints the number of detected text regions to the terminal.
2. Displays the processed text.
3. Saves the results as a JSON file.

The JSON output contains extracted text and additional information about text regions, such as page numbers and spatial coordinates when available.

The output file can be used for further processing, document analysis, or integration with other applications.

## 11. Output Format

The application saves processing results as a JSON file.

The output is organized by pages. Each page contains the extracted text and a list of text lines with additional spatial information.

### 11.1 JSON Structure

The output is a list of page objects.

Each page contains:

* `page` – page number.
* `text` – complete extracted text from the page, with line breaks.
* `lines` – a list of text lines with their spatial information.

Example:

```json
[
  {
    "page": 1,
    "text": "First recognized line\nSecond recognized line",
    "lines": [
      {
        "text": "First recognized line",
        "bbox": [120, 150, 850, 185],
        "center_y": 167.5,
        "height": 35,
        "gap_after": 12,
        "gap_ratio": 0.34
      },
      {
        "text": "Second recognized line",
        "bbox": [120, 197, 760, 232],
        "center_y": 214.5,
        "height": 35
      }
    ]
  }
]
```

*The values above are illustrative and do not represent an actual processing result.*

### 11.2 Page Information

Each page is represented as a separate object.

The `page` field preserves the original page number, while `text` contains the recognized content in reading order.

Text lines are separated by newline characters. This makes the output easier to read and provides a basic representation of the original document structure.

### 11.3 Bounding Boxes

Each line contains a `bbox` field representing its bounding box:

```json
"bbox": [x1, y1, x2, y2]
```

The coordinates represent:

* `x1`, `y1` – top-left corner.
* `x2`, `y2` – bottom-right corner.

For scanned documents, these coordinates are based on the rendered page image. For PDFs with an existing text layer, they come from the PDF text extraction process.

Bounding boxes are combined when individual text regions are grouped into lines.

### 11.4 Spatial Metadata

Each line can contain additional information:

* `center_y` – vertical center of the line.
* `height` – height of the line's bounding box.
* `gap_after` – vertical distance between the current line and the next line.
* `gap_ratio` – the distance between lines relative to their average height.

The `gap_ratio` is calculated using the heights of two consecutive lines. It can help identify larger vertical gaps, which may indicate paragraph boundaries or other document structure.

The last line on a page does not have `gap_after` or `gap_ratio`, because there is no following line to compare it with.

### 11.5 Region Metadata

During processing, individual text regions are represented by a `TextRegion` data structure.

It contains fields for:

* Page number.
* Recognized text.
* Bounding box.
* Source information.
* Optional confidence value.
* Optional crop ID.
* Optional line ID.

These fields are used internally during processing. The final JSON currently focuses on page-level text and line-level spatial information rather than exporting all internal region metadata.

## 12. Limitations

The current implementation is a working OCR pipeline, but it still has several limitations related to document layout analysis, text recognition, and spatial information.

### Text Detection and Line Grouping

* Line grouping is based on a relatively simple geometric approach using the bounding boxes returned by DBNet.
* Line crops are created by extending the vertical boundaries of the first detected word. This approach does not account for text rotation or curved baselines.
* Incorrect grouping or cropping may result in missing parts of words or including fragments of neighbouring lines.

### Document Layout and Reading Order

* The current implementation does not perform full document layout analysis.
* Multi-column documents may produce an incorrect reading order.
* Paragraph boundaries, tables, headers, and other structural elements are not reliably reconstructed.

### Image Geometry and Quality

* There is no dedicated deskewing or perspective correction stage.
* Curved pages and heavily skewed scans may negatively affect text detection and recognition.
* Recognition quality depends heavily on image resolution, contrast, noise, and overall document quality.

### Input Format Limitations

* DOCX processing currently extracts text from paragraphs only. Tables are not supported.
* DOCX output does not contain spatial information such as bounding boxes.
* PDF documents with an existing text layer are processed through direct text extraction. This path does not currently preserve the same spatial metadata as the OCR pipeline.

### Text Recognition and Confidence

* The OCR model can still make character-level mistakes, including confusion between visually similar letters and digits.
* Polish diacritical characters may also be incorrectly recognized or omitted.
* The current implementation does not provide a reliable confidence score for individual predictions or complete text regions.

These limitations are known and provide a basis for further development.

## 13. Future Development

The project is still under development. The following improvements are planned for future versions.

### Document Processing and Layout Analysis

* **Improved geometric line grouping** – develop a more reliable method for grouping detected words into text lines.
* **Handling rotated and skewed text lines** – improve line detection and cropping to account for text that is not perfectly horizontal.
* **Better reading order reconstruction** – improve the ordering of text regions, especially in multi-column documents.
* **Improved document structure preservation** – retain additional spatial metadata for PDFs with an existing text layer and DOCX documents.
* **Extended DOCX support** – add extraction of text from tables and improve the handling of document structure.

### OCR Model Improvements

* **Dataset expansion** – increase the diversity of training data to improve recognition across different document types and text styles.
* **Extended character support** – expand the character set to include additional special characters and symbols.
* **Improved recognition of text styles** – investigate better recognition of italic text, special formatting, and other typographic variations.
* **Model fine-tuning** – continue training the existing model on more diverse data and evaluate its ability to generalize to documents outside the original training distribution.
* **Recognition quality improvements** – investigate remaining character-level errors and improve recognition of visually similar characters.

The main objective is to improve the reliability of the entire document processing pipeline, rather than focusing only on OCR model accuracy.

14. AI-Assisted Development

Large Language Models (LLMs) were used as supporting tools throughout the development of this project.

Most of the main ideas, architectural decisions, experiments, and development directions came from my own work. I used LLMs primarily as a learning and problem-solving aid rather than asking them to generate complete project components.

My typical workflow involved discussing implementation ideas, exploring possible solutions, understanding unfamiliar concepts, debugging problems, and working through implementation steps with AI assistance.

One notable exception is the text decoder and dictionary correction module. These components were implemented with substantial assistance from an LLM, including code generation. I consider them part of the project, but I do not claim full independent authorship of their implementation. Understanding and reviewing this code in greater depth remains part of my learning goals.

I believe that using AI tools responsibly is an important part of modern software development. At the same time, I consider it important to be transparent about where and how they contributed to my work.

15. Related Projects