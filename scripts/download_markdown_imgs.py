import re
from dataclasses import dataclass
from pathlib import Path

import requests
from loguru import logger
from nlp_utils.file_utils import JsonFactory
from nlp_utils.log_utils import set_logger_level
from tqdm import tqdm


@dataclass
class Note:
    id: str
    title: str
    time: str
    tags: list


COOKIE = None
BASIC_URL = "https://notes.sjtu.edu.cn/"
HISTORY_URL = "https://notes.sjtu.edu.cn/history"


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, default="./notes_online_version")
    parser.add_argument("--output_dir", "-o", type=str, default="./notes_local_version")
    # parser.add_argument("--cookie", type=str, default=None)
    return parser.parse_args()


class MarkdownImageDownloader:
    def __init__(self, args) -> None:
        self.args = args
        self.history_path = "sjtu_notes_history.json"

    def download_history(self):
        history = requests.get(HISTORY_URL, headers={"Cookie": COOKIE}).json()
        logger.debug(f"Downloaded history, length: {len(history['history'])}")
        JsonFactory.write_json(history, self.history_path)

    def parse_history(self):
        history = JsonFactory.load_json(self.history_path)
        notes = []
        for note in history["history"]:
            notes.append(
                Note(
                    id=note["id"],
                    title=note["title"],
                    time=note["time"],
                    tags=note["tags"],
                )
            )
        logger.debug(f"Parsed history, length: {len(notes)}")
        return notes

    def extract_images_from_markdown_path(
        self,
        markdown_path: str,
        output_dir: str = "./output",
    ):
        if not Path(markdown_path).exists():
            raise FileNotFoundError
        markdown_str = Path(markdown_path).read_text()
        self.extract_images_from_markdown(markdown_str, markdown_path.stem, output_dir)

    def extract_images_from_markdown(
        self,
        markdown_str,
        markdown_name: str,
        output_dir: str = "./output",
    ):
        output_dir = Path(output_dir)
        image_dir = output_dir / "images"
        image_dir.mkdir(exist_ok=True, parents=True)

        image_urls = []
        # match pattern1: ![](.*/uploads/upload_161173a348b6b6e26b77dde023653460.png)
        pattern1 = re.compile(r"!\[\]\(.*/uploads/upload_.*?\)")
        for match in pattern1.finditer(markdown_str):
            image_str = match.group()
            image_url = image_str[4:-1]
            if not image_url.startswith(BASIC_URL):
                image_url = BASIC_URL + image_url
            image_name = image_url.split("/")[-1]
            logger.debug(f"image_url: {image_url} image_name: {image_name}")
            image_urls.append(image_url)
            # replace with local path
            local_path = Path(image_dir).joinpath(image_name).relative_to(output_dir)
            markdown_str = markdown_str.replace(image_str, f"![]({local_path})")
        # match pattern2: <img src=".*/uploads/upload_161173a348b6b6e26b77dde023653460.png"
        pattern2 = re.compile(r"<img src=\".*/uploads/upload_.*?\"")
        for match in pattern2.finditer(markdown_str):
            image_str = match.group()
            image_url = image_str[10:-1]
            if not image_url.startswith(BASIC_URL):
                image_url = BASIC_URL + image_url
            image_name = image_url.split("/")[-1]
            logger.debug(f"image_url: {image_url} image_name: {image_name}")
            image_urls.append(image_url)
            # replace with local path
            local_path = Path(image_dir) / image_name
            markdown_str = markdown_str.replace(image_str, f'<img src="{local_path}"')
        logger.debug(f"image_urls: {image_urls}")

        # download images
        self.download_images(image_urls, image_dir=image_dir)
        logger.info(f"Downloaded {len(image_urls)} images")

        # save markdown
        output_markdown_path = output_dir / f"{markdown_name}.md"
        output_markdown_path.write_text(markdown_str)
        logger.info(f"Saved markdown to {output_markdown_path}")

    def download_images(self, image_urls, image_dir):
        Path(image_dir).mkdir(exist_ok=True)
        for image_url in tqdm(image_urls, desc="Downloading images"):
            image_name = image_url.split("/")[-1]
            image_path = Path(image_dir) / image_name
            if image_path.exists():
                logger.debug(f"Skip downloading {image_url}")
                continue
            logger.debug(f"Downloading {image_url}")
            r = requests.get(image_url, headers={"Cookie": COOKIE})
            with open(image_path, "wb") as f:
                f.write(r.content)
            logger.debug(f"Downloaded {image_url}")


if __name__ == "__main__":
    """
    Usage:
        python sjtu_notes_downloader.py -i ./notes_online_version -o ./notes_local_version
        python sjtu_notes_downloader.py -i notes-20240407.zip -o ./notes_local_version
    """
    set_logger_level("INFO")
    args = parse_args()
    downloader = MarkdownImageDownloader(args)
    # downloader.download_history()
    # downloader.parse_history()
    if Path(args.input_dir).is_file() and Path(args.input_dir).suffix == ".zip":
        from zipfile import ZipFile

        with ZipFile(args.input_dir, "r") as zip_ref:
            # traverse each file in zip file
            markdown_list = filter(lambda x: x.endswith(".md"), zip_ref.namelist())
            for file_name in markdown_list:
                with zip_ref.open(file_name, "r") as markdown_file:
                    markdown_content = markdown_file.read().decode("utf-8")
                    downloader.extract_images_from_markdown(
                        markdown_content,
                        markdown_name=Path(file_name).stem,
                        output_dir=args.output_dir,
                    )
    else:
        markdown_list = list(Path(args.input_dir).glob("*.md"))
        for markdown_path in tqdm(markdown_list, desc="Extracting images"):
            downloader.extract_images_from_markdown_path(
                markdown_path, output_dir=args.output_dir
            )
