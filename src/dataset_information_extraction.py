import datetime
import json
import os
import time
import openai
import tenacity
import tiktoken
from termcolor import colored
import fitz
import logging


class Paper:
    SECTION_NAMES_LIST = [
        "Abstract",
        "Introduction",
        "Methodology",
        "Related Work",
        "Basic definitions and properties",
        "Background",
        "Background and Related Work",
        "Background and Motivation",
        "Background and Preliminaries",
        "Preliminaries",
        "Model Representation",
        "Motivation",
        "Motivations",
        "System Model",
        "Problem Setting",
        "Proposed Method",
        "Methodology for Studying the Question",
        "Basic Concepts and Method",
        "Methods",
        "Implementation",
        "Proofs",
        "Proof",
        "Algorithm Description" "Discussion",
        "Conclusion and Future Work",
        "Future Research Direction",
        "Conclusion",
        "References",
        "Acknowledgements",
        "Appendix",
    ]
    DATASET_RELATED_SECTION_NAMES_LIST = [
        "Evaluation and Results",
        "Experiments",
        "Experiment",
        "Results",
        "Evaluation",
        "Evaluation and Discussion",
        "Results and Discussion",
        "Experiment and Result",
        "Performance Evaluation",
        "Numerical Results",
        "Benchmark",
        "Experiments and Results",
        "Data Analysis",
        "Experimental Results",
        "Numerical Examples",
        "Data",
        "Dataset",
        "Datasets",
        "Data Collection",
        "Materials and Methods",
    ]

    def __init__(self, path, arxiv_metadata=None):
        self.path = path
        self.section_names = []
        self.section_start_pages = {}
        self.section_texts = {}
        self.all_section_names = (
            self.SECTION_NAMES_LIST + self.DATASET_RELATED_SECTION_NAMES_LIST
        )
        if arxiv_metadata is not None:
            self.arxiv_id = arxiv_metadata["id"]
            self.authors = arxiv_metadata["authors"]
            self.title = arxiv_metadata["title"].replace("\n", " ")
            self.abstract = arxiv_metadata["abstract"].replace("\n", " ")
        else:
            self.arxiv_id = None
            self.authors = None
            self.title = None
            self.abstract = self.section_texts["Abstract"]
        self.parse()

    def parse(self):
        self.pdf_doc = fitz.open(self.path)
        self.doc_texts = [page.get_text() for page in self.pdf_doc]
        self.section_start_pages = self.find_section_start_pages()
        self.set_section_text()

    def find_section_start_pages(self):
        section_start_pages = {}
        found_section_names = set()

        for page_index, page in enumerate(self.pdf_doc):
            page_text = page.get_text()

            for section_name in self.all_section_names:
                if section_name not in found_section_names and (
                    section_name + "\n" in page_text
                    or section_name.upper() in page_text
                ):
                    found_section_names.add(section_name)
                    section_start_pages[section_name] = page_index

        self.section_names.extend(list(found_section_names))
        self.section_start_pages.update(section_start_pages)

        return self.section_start_pages

    def find_text(self, text, page_index, default=None):
        if text is None:
            return default
        index = self.doc_texts[page_index].find(text + "\n")
        if index == -1:
            index = self.doc_texts[page_index].find(text.upper())
        return index if index != -1 else default

    def get_section_text_range(self, section_name, next_section_name, page_index):
        start_index = self.find_text(section_name, page_index, default=0)
        end_index = self.find_text(
            next_section_name, page_index, default=len(self.doc_texts[page_index])
        )
        return start_index, end_index

    def set_section_text(self):
        """
        Extracts text from each section of a PDF document.

        This method iterates over the sections defined in the `section_names` attribute.
        For each section, it determines the start and end pages using `section_start_pages`.
        It then concatenates the text from these pages, considering only the relevant portions
        of the start and end pages based on the `get_section_text_range` method.

        The extracted text for each section is stored in the `section_texts` dictionary,
        with section names as keys and the corresponding extracted text as values.

        Note:
            - Assumes that `self.pdf_doc` contains the PDF document pages.
            - `self.doc_texts` should contain the text of each page in the document.
            - This method does not return any value but updates `self.section_texts`.
        """
        for index, section_name in enumerate(self.section_names):
            start_page = self.section_start_pages[section_name]
            end_page = (
                self.section_start_pages.get(
                    self.section_names[index + 1], len(self.pdf_doc)
                )
                - 1
                if index + 1 < len(self.section_names)
                else len(self.pdf_doc) - 1
            )

            section_text = ""
            for page_index in range(start_page, end_page + 1):
                if (
                    start_page == end_page
                    or page_index == start_page
                    or page_index == end_page
                ):
                    next_section_name = (
                        self.section_names[index + 1]
                        if index + 1 < len(self.section_names)
                        else None
                    )
                    start_index, end_index = self.get_section_text_range(
                        section_name, next_section_name, page_index
                    )
                else:
                    start_index, end_index = 0, len(self.doc_texts[page_index])

                section_text += self.doc_texts[page_index][start_index:end_index]

            self.section_texts[section_name] = section_text

    def log_section_contents(self):
        """
        Logs the content of each section in the document.

        This method iterates through all sections specified in `self.section_names` and logs their content.
        It uses a logging framework instead of direct print statements for better control and flexibility.
        """
        separator = "*" * 100  # 可以根据需要更改分隔符
        for section_name in self.section_names:
            if section_name in self.section_texts:
                logging.info(colored(f"{section_name}", "cyan"))
                logging.info(self.section_texts[section_name])
                logging.info(separator)
            else:
                logging.error("Section '%s' not found in section_texts.", section_name)

    def have_dataset_related_section(self):
        # 现在可以使用类变量而不是重新定义列表
        for section in self.section_names:
            if section in self.DATASET_RELATED_SECTION_NAMES_LIST:
                return True
        return False

    def get_dataset_related_section_texts(self):
        dataset_related_section_texts = "\n".join(
            self.section_texts[section_name]
            for section_name in self.section_names
            if section_name in self.DATASET_RELATED_SECTION_NAMES_LIST
        )
        if self.title is not None:
            title = "Title:" + self.title + "\n"
        if self.authors is not None:
            authors = "Authors:" + self.authors + "\n"
        if self.abstract is not None:
            abstract = "Abstract:" + self.abstract + "\n"
        return title + authors + abstract + dataset_related_section_texts


def process_file(file_dir, file):
    try:
        print(colored(f"Extracting information from {file}", "green"))
        paper = Paper(os.path.join(file_dir, file))
        if paper.have_dataset_related_section():
            print(colored(f"{file} has dataset related section", "green"))
            return (True, file)  # 返回一个元组，指示文件是否与数据集相关
        else:
            print(colored(f"{file} has no dataset related section", "red"))
            return (False, file)
    except Exception as e:
        print(colored(f"An error occurred with file {file}: {str(e)}", "red"))
        return (False, file)  # 假设出现错误的文件没有与数据集相关的部分


class LLM_Extract:
    def __init__(self):
        openai_api = json.load(open("openai_api.json", encoding="utf-8"))
        self.chat_model = openai_api["chat_gpt_model"]
        openai.api_base = openai_api.get("api_base", "https://api.openai.com")
        openai.api_key = openai_api["api_key"]
        if openai.api_base == "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx":
            raise Exception("Please set your OpenAI API key in openai_api.json")

    @staticmethod
    def get_truncated_text(original_text, max_tokens, encoding):
        """
        Use binary search to find the truncation position of the text
        so that the number of encoded tokens does not exceed max_tokens
        """
        low = 0
        high = len(original_text)

        while low < high:
            mid = (low + high) // 2
            truncated_text = original_text[:mid]
            encoded_text = encoding.encode(truncated_text)

            if len(encoded_text) > max_tokens:
                high = mid
            else:
                low = mid + 1

        # 当low和high收敛时，low-1将给出最长的文本，该文本的编码长度不超过max_tokens
        return original_text[: low - 1]

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(7),
        reraise=True,
    )
    def llm_extract_dataset_infomation(self, paper_text):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        # 210表示系统和助手的对话内容的token数
        print("start to get paper text")
        paper_text = self.get_truncated_text(paper_text, 2940, encoding)
        print("start llm extract")
        messages = [
            {
                "role": "system",
                "content": "You're a researcher in the Computer Science who is good at extrating information about datasets from papers using concise statements.",
            },
            {
                "role": "assistant",
                "content": f"This is the context related to the dataset information extraction task and I need you read and extract the formation about dataset:\n{paper_text}\n",
            },
            {
                "role": "user",
                "content": """
                1. Mine the dataset information used in this paper.
                2. Generate the following structured information
                Note: Sometimes multiple data sets may be involved in the paper, please answer by dataset
                Follow this json format strictly, if some information cannot be obtained from the provided materials, fill in null
                {
                    "Dataset Name": "xxx",
                    "Dataset Summary": "xxx",
                    "Dataset Publicly Available": "xxx",
                    "Data Type": "xxx",
                    "Task": "xxx",
                    "Dateset Published By": "xxx",
                    "Location": "xxx",
                    "Research Paper": "xxx",
                    "Authors": "xxx",
                    "Other Useful Information about the Dataset": "xxx"
                },
                """,
            },
        ]

        response = openai.ChatCompletion.create(
            model=self.chat_model,
            messages=messages,
        )
        print(response)
        print("finish chat")
        print("*" * 100)
        return "".join(choice.message.content for choice in response.choices)


def find_and_create_paper(file, file_dir, file_metadata_json):
    # 构造完整的文件路径
    full_file_path = os.path.join(file_dir, file)

    # 去除文件名的扩展名，这里假设扩展名是'.pdf'（4个字符长）
    file_id = file[:-4]

    # 在元数据中搜索匹配的ID
    for metadata in file_metadata_json:
        if metadata["id"] == file_id:
            # 如果找到了匹配的元数据，用它来创建一个Paper对象
            return Paper(full_file_path, metadata)

    # 如果没有找到匹配的元数据，使用默认值创建一个Paper对象
    return Paper(full_file_path)


class PaperExtractor:
    def __init__(self, file_dir, metadata_path, output_path):
        self.file_dir = file_dir
        self.metadata_path = metadata_path
        self.output_path = output_path
        self.file_list = os.listdir(file_dir)
        self.file_metadata_json = json.load(open(metadata_path, encoding="utf-8"))
        self.extracted_papers_list = self.load_extracted_papers()

    def load_extracted_papers(self):
        # 尝试加载已提取的论文列表，如果文件为空或不存在，则初始化为空字典
        try:
            with open(self.output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def extract_papers(self):
        extractor = LLM_Extract()
        for index, file in enumerate(self.file_list):
            print(f"Extracting information from {file}")
            if file[:-4] in self.extracted_papers_list:
                print(f"{file} has been extracted")
                continue
            paper = find_and_create_paper(file, self.file_dir, self.file_metadata_json)
            if paper.have_dataset_related_section():
                print(colored(f"{file} has dataset related section", "green"))
                self.process_paper(file, paper, extractor)

    def process_paper(self, file, paper, extractor):
        try:
            results = extractor.llm_extract_dataset_infomation(
                paper.get_dataset_related_section_texts()
            )
            with open(
                "results_" + self.file_dir[2:] + ".txt", "a", encoding="utf-8"
            ) as f:
                f.write(results + "\n")
            self.update_extracted_list(paper)
            print(colored(f"Extracted information from {file} successfully.", "green"))
        except Exception as e:
            print(
                colored(
                    f"An error occurred when llm extracted file {file}: {str(e)}", "red"
                )
            )

    def update_extracted_list(self, paper):
        self.extracted_papers_list[paper.arxiv_id] = {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "extract_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "extract_valid": True,
        }
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.extracted_papers_list, f, indent=4)


if __name__ == "__main__":
    paper_extractor = PaperExtractor(
        "./demo_papers_cs",
        "./demo_papers_info/demo_cs_objects.json",
        "./demo_papers_info/extracted_papers_list.json",
    )
    paper_extractor.extract_papers()
